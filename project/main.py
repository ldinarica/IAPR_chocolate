## YOUR CODE
#To change the test data path, go line 441
#To run this file 
#train mode only: python main.py --mode train
#inference mode only:  python main.py --mode inference
#train + inference: python main.py --mode both




import os
import csv
import json
import random
import argparse

import numpy as np
import pandas as pd


from PIL import Image, ImageDraw, ImageFilter
import cv2
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from tqdm import tqdm

from src.train_UNET import *
from src.train_CNN import *



def set_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def split_blob_mask(mask_pil, min_peak_distance=10, min_distance_value=10):
    """
    Applies conservative watershed to split clearly overlapping blobs in a binary mask.

    Args:
        mask_pil (PIL.Image): Binary circular mask image.
        min_peak_distance (int): Minimum distance (in pixels) between peaks.
        min_distance_value (float): Minimum distance transform value to consider a peak.

    Returns:
        labels (np.ndarray): Labeled image with split blobs.
    """
    # Convert to binary array
    mask = np.array(mask_pil) > 0

    # Compute distance transform
    distance = ndi.distance_transform_edt(mask)

    # Threshold: remove weak peaks
    distance_thresh = distance.copy()
    distance_thresh[distance_thresh < min_distance_value] = 0

    # Detect local maxima (less sensitive due to large min_distance)
    coords = peak_local_max(
        distance_thresh,
        labels=mask,
        min_distance=min_peak_distance,
        footprint=np.ones((min_peak_distance, min_peak_distance)),
        exclude_border=False
    )

    mask_peaks = np.zeros_like(distance, dtype=bool)
    mask_peaks[tuple(coords.T)] = True

    # Label peak markers
    markers = label(mask_peaks)

    # Watershed segmentation (on inverted distance)
    labels = watershed(-distance, markers, mask=mask)

    return labels


def save_formatted_csv(output_path, dataset, class_names):
    # Step 1: Build CSV content in memory
    rows = []

    for idx in tqdm(range(len(dataset)), desc="Processing images"):
        img, fname, preds = dataset[idx]

        # Convert predicted class names to indices
        pred_classes = [class_names.index(pred['pred']) for pred in preds]
        counts = [pred_classes.count(i) for i in range(len(class_names))]

        # Remove extension and first character from filename
        image_id = os.path.splitext(fname)[0][1:]

        # Optional: convert to int if needed
        try:
            image_id_int = int(''.join(filter(str.isdigit, image_id)))
        except ValueError:
            image_id_int = image_id

        rows.append([image_id] + counts)

    # Step 2: Create DataFrame
    df = pd.DataFrame(rows, columns=["id"] + class_names)

    # Step 3: Rename columns to chocolate names
    type_to_class = {
        "Type1": "Amandina",
        "Type2": "Arabia",
        "Type3": "Comtesse",
        "Type4": "Crème brulée",
        "Type5": "Jelly Black",
        "Type6": "Jelly Milk",
        "Type7": "Jelly White",
        "Type8": "Noblesse",
        "Type9": "Noir authentique",
        "Type10": "Passion au lait",
        "Type11": "Stracciatella",
        "Type12": "Tentation noir",
        "Type13": "Triangolo"
    }

    df = df.rename(columns=type_to_class)

    # Step 4: Desired column order
    desired_order = [
        "id",
        "Jelly White", "Jelly Milk", "Jelly Black", "Amandina", "Crème brulée", "Triangolo",
        "Tentation noir", "Comtesse", "Noblesse", "Noir authentique", "Passion au lait",
        "Arabia", "Stracciatella"
    ]

    # Step 5: Save to final CSV
    df_reordered = df[desired_order]
    df_reordered = df_reordered.astype(int)

    df_reordered.to_csv(output_path, index=False)

    print(f"Formatted CSV saved to: {output_path}")



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
        x = self.features(x)
        return self.classifier(x)

# 2) The Pipeline Dataset
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage.measure    import label, regionprops
from skimage.morphology import remove_small_objects
import torchvision.transforms as T
import torch.nn as nn
from torch.utils.data import Dataset
import csv

class ChocolateInferenceDataset(Dataset):
    def __init__(self,
                 image_dir: str,
                 seg_model: nn.Module,
                 cls_model: nn.Module,
                 class_names: list,
                 device: torch.device,
                 seg_size=(256, 256),
                 cls_size=(512, 512),     
                 seg_thresh=0.5,
                 min_blob_size=100):
        self.image_dir   = image_dir
        self.fnames      = sorted(os.listdir(image_dir))
        self.seg_model   = seg_model.to(device).eval()
        self.cls_model   = cls_model.to(device).eval()
        self.class_names = class_names
        self.device      = device

        self.seg_tf = T.Compose([
            T.Resize(seg_size),
            T.ToTensor(),
        ])
        self.cls_tf = T.Compose([
            T.Resize(cls_size),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],
                        [0.229,0.224,0.225])
        ])

        self.seg_thresh    = seg_thresh
        self.min_blob_size = min_blob_size

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img   = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
        W, H  = img.size

        # 1) SEGMENTATION INFERENCE at 256 resolution
        x512 = self.seg_tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out512 = self.seg_model(x512)[0, 0].cpu().numpy()  # [256, 256]

        # 2) Threshold & cleanup
        mask512 = out512 > self.seg_thresh
        

        # 3) Upsample to original image size
        mask_img  = Image.fromarray((mask512 * 255).astype(np.uint8))
        mask_full = np.array(mask_img.resize((W, H), Image.NEAREST)) > 0


        # 4) Extract blobs & classify
        results = []
        labels = split_blob_mask(mask_full, min_peak_distance=200, min_distance_value=200)

        for blob in regionprops(labels):
            minr, minc, maxr, maxc = blob.bbox
            height = maxr - minr
            width = maxc - minc
            max_side = max(height, width)
            center_r = (minr + maxr) // 2
            center_c = (minc + maxc) // 2

            half_side = max_side // 2
            minr_sq = max(center_r - half_side, 0)
            maxr_sq = min(center_r + half_side, img.height)
            minc_sq = max(center_c - half_side, 0)
            maxc_sq = min(center_c + half_side, img.width)

            patch = img.crop((minc_sq, minr_sq, maxc_sq, maxr_sq))
            
            xp = self.cls_tf(patch).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred_idx = self.cls_model(xp).argmax(1).item()
                


            results.append({
                "bbox": (minc_sq, minr_sq, maxc_sq - minc_sq, maxr_sq - minr_sq),
                "pred": self.class_names[pred_idx]
            })

        return img, fname, results

def Inference(TEST_DIR):
        # 3) Usage + Visualization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_names =  class_names = [
            "Amandina",
            "Arabia",
            "Comtesse",
            "Crème brulée",
            "Jelly Black",
            "Jelly Milk",
            "Jelly White",
            "Noblesse",
            "Noir authentique",
            "Passion au lait",
            "Stracciatella",
            "Tentation noir",
            "Triangolo",

        ]
        # instantiate models and load weights
        seg_model = UNet()
        seg_model.load_state_dict(torch.load("best_model_unet.pth", map_location=device))
        cls_model = ChocolateNet(len(class_names))

        cls_model.load_state_dict(torch.load("best_checkpoint_cnn.pth", map_location=device)["model_state"])
        #best_choco_cnn
        # dataset
        dataset = ChocolateInferenceDataset(
            image_dir=TEST_DIR,
            seg_model=seg_model,
            cls_model=cls_model,
            class_names=class_names,
            device=device
        )
        save_formatted_csv("submission.csv", dataset, class_names)

    


def save_formatted_csv(output_path, dataset, class_names):
    # Step 1: Build CSV content in memory
    rows = []

    for idx in tqdm(range(len(dataset)), desc="Processing images"):
        img, fname, preds = dataset[idx]

        # Convert predicted class names to indices
        pred_classes = [class_names.index(pred['pred']) for pred in preds]
        counts = [pred_classes.count(i) for i in range(len(class_names))]

        # Remove extension and first character from filename
        image_id = os.path.splitext(fname)[0][1:]

        # Optional: convert to int if needed
        try:
            image_id_int = int(''.join(filter(str.isdigit, image_id)))
        except ValueError:
            image_id_int = image_id

        rows.append([image_id] + counts)

    # Step 2: Create DataFrame
    df = pd.DataFrame(rows, columns=["id"] + class_names)

    # Step 3: Rename columns to chocolate names
    type_to_class = {
        "Type1": "Amandina",
        "Type2": "Arabia",
        "Type3": "Comtesse",
        "Type4": "Crème brulée",
        "Type5": "Jelly Black",
        "Type6": "Jelly Milk",
        "Type7": "Jelly White",
        "Type8": "Noblesse",
        "Type9": "Noir authentique",
        "Type10": "Passion au lait",
        "Type11": "Stracciatella",
        "Type12": "Tentation noir",
        "Type13": "Triangolo"
    }

    df = df.rename(columns=type_to_class)

    # Step 4: Desired column order
    desired_order = [
        "id",
        "Jelly White", "Jelly Milk", "Jelly Black", "Amandina", "Crème brulée", "Triangolo",
        "Tentation noir", "Comtesse", "Noblesse", "Noir authentique", "Passion au lait",
        "Arabia", "Stracciatella"
    ]

    # Step 5: Save to final CSV
    df_reordered = df[desired_order]
    df_reordered.to_csv(output_path, index=False)
    print(f"Formatted CSV saved to: {output_path}")


# ========== RUN SCRIPT ==========
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Chocolate Classification Pipeline")
    parser.add_argument("--mode", choices=["train", "inference", "both"], default="inference",
                        help="Specify whether to run training, inference, or both.")
    args = parser.parse_args()
    set_seed(100)
    TRAIN_DIR = "train"
    TEST_DIR = "test"
    JSON_PATH = "train/_annotations.coco.json"
    CLASS_NAMES = [
        "Amandina", "Arabia", "Comtesse", "Crème brulée", "Jelly Black",
        "Jelly Milk", "Jelly White", "Noblesse", "Noir authentique",
        "Passion au lait", "Stracciatella", "Tentation noir", "Triangolo"
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #train
    if args.mode in ["train", "both"]:
    #Unet
        train_Unet(JSON_PATH,TRAIN_DIR)

    #CNN
        model, val_loader = train(
                TRAIN_DIR, JSON_PATH, CLASS_NAMES,
                epochs=15, batch_size=8, lr=1e-3,
                val_split=0.2, device=device,
                repeat_times=4,
            )
    if args.mode in ["inference", "both"]:
    #inference
        Inference(TEST_DIR)


 