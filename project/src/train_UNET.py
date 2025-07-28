## YOUR CODE
## YOUR CODE
# train_segmentation.py

import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter, map_coordinates
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

def set_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ========== MASK GENERATION ==========
def generate_masks(json_path, output_mask_dir, fixed_radius=290):
    os.makedirs(output_mask_dir, exist_ok=True)
    with open(json_path, "r") as f:
        coco = json.load(f)

    image_id_to_info = {img["id"]: img for img in coco["images"]}
    image_id_to_annotations = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        image_id_to_annotations.setdefault(img_id, []).append(ann)

    for image_id in tqdm(image_id_to_annotations, desc="Generating circular masks"):
        img_info = image_id_to_info[image_id]
        width, height = img_info["width"], img_info["height"]
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        for ann in image_id_to_annotations[image_id]:
            bbox = ann["bbox"]
            x_center = bbox[0] + bbox[2] // 2
            y_center = bbox[1] + bbox[3] // 2
            draw.ellipse(
                (x_center - fixed_radius, y_center - fixed_radius,
                 x_center + fixed_radius, y_center + fixed_radius),
                fill=255
            )

        base_name = os.path.splitext(img_info["file_name"])[0]
        mask.save(os.path.join(output_mask_dir, f"{base_name}_mask.png"))

# ========== UNET MODEL ==========
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.final(d1))

# ========== DATASET CLASS ==========
def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    image_np = np.array(image)
    shape = image_np.shape[:2]
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    if image_np.ndim == 3:
        channels = [map_coordinates(image_np[..., c], indices, order=1, mode='reflect').reshape(shape)
                    for c in range(image_np.shape[2])]
        distorted = np.stack(channels, axis=-1)
    else:
        distorted = map_coordinates(image_np, indices, order=1, mode='reflect').reshape(shape)

    return Image.fromarray(np.uint8(distorted))


def random_cutout(image, mask, num_holes=3, max_size=40):
    w, h = image.size
    for _ in range(num_holes):
        cutout_w = random.randint(10, max_size)
        cutout_h = random.randint(10, max_size)
        x0 = random.randint(0, w - cutout_w)
        y0 = random.randint(0, h - cutout_h)
        image.paste((0, 0, 0), (x0, y0, x0 + cutout_w, y0 + cutout_h))
    return image, mask


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size=(256, 256), augmentations=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
        self.resize = T.Resize(size)

        base_images = [
            img for img in os.listdir(image_dir)
            if img.endswith(('.jpg', '.png')) and os.path.exists(
                os.path.join(mask_dir, os.path.splitext(img)[0] + "_mask.png"))
        ]

        self.available_augmentations = {
            "hflip": lambda img, msk: (TF.hflip(img), TF.hflip(msk)),
            "vflip": lambda img, msk: (TF.vflip(img), TF.vflip(msk)),
            "rotation": lambda img, msk: (TF.rotate(img, angle := random.uniform(-15, 15)),
                                          TF.rotate(msk, angle)),
            "color_jitter": lambda img, msk: (T.ColorJitter(0.2, 0.2, 0.2, 0.05)(img), msk),
            "blur": lambda img, msk: (img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.5))), msk),
            "elastic": lambda img, msk: (elastic_transform(img, 20, 4), elastic_transform(msk, 20, 4)),
            "cutout": lambda img, msk: random_cutout(img, msk, 10, 30),
            "cutout2": lambda img, msk: random_cutout(img, msk, 3, 120),
            "cutout3": lambda img, msk: random_cutout(img, msk, 50, 20),
        }

        self.augmentations = augmentations if augmentations else []
        self.samples = []

        if augmentations:
            for img in base_images:
                self.samples.append((img, None))
                for aug_name, prob in augmentations:
                    if random.random() < prob:
                        self.samples.append((img, aug_name))
        else:
            for img in base_images:
                self.samples.append((img, None))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, _ = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, os.path.splitext(img_name)[0] + "_mask.png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image, mask = self.resize(image), self.resize(mask)

        for aug_name, prob in self.augmentations:
            if random.random() < prob and aug_name in self.available_augmentations:
                image, mask = self.available_augmentations[aug_name](image, mask)

        image = T.ToTensor()(image)
        mask = T.ToTensor()(mask)
        mask = (mask > 0.5).float()
        return image, mask

# ========== MAIN TRAINING ==========
def train_Unet(json_path,image_dir):
    
    output_mask_dir = image_dir+"/train_mask"
    generate_masks(json_path, output_mask_dir)

    dataset = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=output_mask_dir,
        size=(256, 256),
        augmentations=[
        ("hflip", 0.1),
        ("vflip", 0.1),
        ("rotation", 0.3),
        ("color_jitter", 0.1),
        ("blur", 0.1),
        ("elastic", 0.2),
        ("cutout", 0.5),
        ("cutout2", 0.5),
    ]
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Starting training...")
    for epoch in range(25):
        model.train()
        total_loss = 0
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/20 - Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "best_model_unet.pth")
    print("Model saved as best_model_unet.pth")