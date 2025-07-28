import os
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR

import torchvision.transforms as T

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def set_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CocoChocolateDataset(Dataset):
    def __init__(self, img_dir, coco_json, class_names, img_size=(512,512), mode='train'):
        coco = json.load(open(coco_json))
        self.img_dir = img_dir
        self.id2fn = {im['id']: im['file_name'] for im in coco['images']}
        self.samples = [
            (ann['image_id'], tuple(map(int, ann['bbox'])), ann['category_id'] - 1)
            for ann in coco['annotations']
        ]
        self.class_names = class_names
        self.img_size = img_size
        self.mode = mode
        self._setup_transforms()

    def _setup_transforms(self):
        if self.mode == 'train':
            self.tf = T.Compose([
                T.Resize(self.img_size),
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(self.img_size, scale=(0.6,1.0), ratio=(0.75,1.33)),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                T.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.8,1.2), shear=10),
                T.ToTensor(),
                T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
                T.RandomErasing(p=0.5, scale=(0.02,0.2), ratio=(0.3,3.3))
            ])
        else:
            self.tf = T.Compose([
                T.Resize(self.img_size),
                T.ToTensor(),
                T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, (x, y, w, h), cls = self.samples[idx]
        fn = self.id2fn[img_id]
        img = Image.open(os.path.join(self.img_dir, fn)).convert('RGB')
        patch = img.crop((x, y, x+w, y+h))
        return self.tf(patch), torch.tensor(cls, dtype=torch.long), fn, (x, y, w, h)

class RepeatDataset(Dataset):
    def __init__(self, ds, times):
        self.ds = ds
        self.times = times

    def __len__(self):
        return len(self.ds) * self.times

    def __getitem__(self, idx):
        return self.ds[idx % len(self.ds)]

class ChocolateNet(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(512,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512,512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

def train(img_dir, coco_json, class_names, epochs=20, batch_size=32, lr=1e-3, val_split=0.2,  
          device='cuda', repeat_times=4, resume_ckpt=None): #change PARAMETERS
    print(f"Training on {device}")
    base_ds = CocoChocolateDataset(img_dir, coco_json, class_names, mode='train')
    full_ds = RepeatDataset(base_ds, times=repeat_times)
    n_val = int(len(full_ds) * val_split)
    train_ds, val_ds = random_split(full_ds, [len(full_ds)-n_val, n_val])

    val_ds.dataset.ds.mode = 'val'
    val_ds.dataset.ds._setup_transforms()

    all_train_labels = [train_ds.dataset.ds.samples[i % len(train_ds.dataset.ds.samples)][2] for i in train_ds.indices]
    counts = np.bincount(all_train_labels, minlength=len(class_names))
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(class_names)
    weight_tensor = torch.tensor(weights, device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weight_tensor)
    sample_weights = [weights[l] for l in all_train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = ChocolateNet(len(class_names)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs, pct_start=0.1, div_factor=10)

    start_epoch, best_acc = 1, 0.0
    if resume_ckpt and os.path.isfile(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        scheduler.load_state_dict(ckpt['sched_state'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']

    for epoch in range(start_epoch, epochs+1):
        model.train()
        running_loss = 0.0
        for imgs, labels, *_ in train_loader:
            mixed, la, lb, lam = mixup_data(imgs, labels, alpha=0.4)
            mixed, la, lb = mixed.to(device), la.to(device), lb.to(device)
            out = model(mixed)
            loss = lam * criterion(out, la) + (1 - lam) * criterion(out, lb)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
            running_loss += loss.item() * mixed.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        model.eval(); val_loss = 0.0; y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels, *_ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_loss += criterion(out, labels).item() * imgs.size(0)
                y_pred.extend(out.argmax(1).cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(y_true, y_pred)
        print(f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'sched_state': scheduler.state_dict(),
                'best_acc': best_acc,
            }, "best_checkpoint_cnn.pth")

    return model, val_loader






 