
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main5.py - Unified Federated Learning Benchmark
-----------------------------------------------
Datasets:
 - Clinical CSVs: lung (Stage), heart (DEATH_EVENT), diabetes (diabetes)
 - MNIST, CIFAR-10 (+ optional binary reductions)

Methods:
 - FedAvg, FedProx, FedSGD, Adaptive (PBFT + kNN-Shapley + intent-aware)

Features:
 - Non-IID Dirichlet client partitions
 - Differential Privacy (Gaussian noise + RDP accountant)
 - PBFT validators and ledger
 - kNN-Shapley (client-model–centric) contribution scores
 - Parallel validator evaluation (joblib)
 - Overlay plots and CSV logs
 - Run a single method or all methods in one call (--run_all)

Examples:
  # Run all methods on MNIST (10-class)
  python main5.py --dataset mnist --rounds 5 --clients 8 --run_all --out_dir ./results

  # Adaptive on lung CSVs with DP + stronger validators
  python main5.py --dataset lung --lung_csv data/lung_cancer_data.csv \
      --rounds 10 --validators 6 --validator_val_size 600 \
      --base_noise 0.1 --clip_norm 2.0 --method adaptive --out_dir ./results

    MedMnist Citation
        @article{medmnistv2,
    title={MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
    author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
    journal={Scientific Data},
    volume={10},
    number={1},
    pages={41},
    year={2023},
    publisher={Nature Publishing Group UK London}
}

@inproceedings{medmnistv1,
    title={MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis},
    author={Yang, Jiancheng and Shi, Rui and Ni, Bingbing},
    booktitle={IEEE 18th International Symposium on Biomedical Imaging (ISBI)},
    pages={191--195},
    year={2021}
}
"""

import argparse, os, time, math, random, copy, warnings
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from joblib import Parallel, delayed, parallel_backend
import torch.nn.functional as F

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from tqdm import tqdm
try:
    from medmnist import PathMNIST, ChestMNIST, OCTMNIST, TissueMNIST, OrganAMNIST, OrganSMNIST
except ImportError:
    PathMNIST = ChestMNIST = OCTMNIST = TissueMNIST = OrganAMNIST = OrganSMNIST = None

def require_medmnist():
    if PathMNIST is None:
        raise ImportError("medmnist is required for PathMNIST/TissueMNIST/OrganMNIST datasets. Install with: pip install medmnist")
from sklearn.metrics import pairwise_distances
from collections import defaultdict
from sklearn.model_selection import train_test_split
from collections import defaultdict, deque

# print(medmnist.__version__)

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------
# Reproducibility
# -------------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Models
# -------------------------------
class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden=256, num_classes=2):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)

class MNISTCNN(nn.Module):
    """Small CNN for MNIST (1x28x28)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 → 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 → 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14 → 14x14
            nn.ReLU(),
            nn.MaxPool2d(2)   # 14 → 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),  # 64 feature maps of size 7x7
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
    
class SimpleCNN(nn.Module):
    """Small CNN for CIFAR-10."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))
    
class SimpleCNN_MedMNIST(nn.Module):
    """Small CNN for CIFAR-10."""
    def __init__(self,in_channels=3, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class SimpleCNN_MedMNIST2(nn.Module):
    def __init__(self, in_channels=3, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # 3x28x28 -> 32x28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 32x14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64x14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 64x7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class DeepCNN_MedMNIST(nn.Module):
    def __init__(self, in_channels=3, num_classes=9):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),       # 28 -> 14
            # nn.Dropout(0.25)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),        # 14 -> 7
            # nn.Dropout(0.25)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),        # 7 -> 3
            # nn.Dropout(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.classifier(x)



# Basic residual block
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

# ResNet18 Modified for 28x28 MedMNIST
class ResNet18_MedMNIST(nn.Module):
    def __init__(self, in_channels=3, num_classes=9):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)  # <-- Smaller kernel than default
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # No maxpool here because input is small (28x28)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # 28x28
        x = self.layer1(x)                      # 28x28
        x = self.layer2(x)                      # 14x14
        x = self.layer3(x)                      # 7x7
        x = self.layer4(x)                      # 4x4
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# Inverted Residual Block
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden_dim = inp * expand_ratio
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # Point-wise expansion
            layers.append(nn.Conv2d(inp, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Point-wise linear projection
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# MobileNetV2 for MedMNIST
class MobileNetV2_MedMNIST(nn.Module):
    def __init__(self, in_channels=3, num_classes=9, width_mult=1.0):
        super().__init__()
        # First layer
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * width_mult)

        self.features = [nn.Conv2d(in_channels, input_channel, 3, 2, 1, bias=False),
                         nn.BatchNorm2d(input_channel),
                         nn.ReLU6(inplace=True)]

        # Inverted residual settings (t, c, n, s)
        block = InvertedResidual
        settings = [
            # expand, out, blocks, stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Build inverted residual blocks
        for t, c, n, s in settings:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(block(input_channel, output_channel, stride, t))
                input_channel = output_channel

        # Last layers
        self.features.append(nn.Conv2d(input_channel, last_channel, 1, bias=False))
        self.features.append(nn.BatchNorm2d(last_channel))
        self.features.append(nn.ReLU6(inplace=True))
        self.features = nn.Sequential(*self.features)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.size(0), -1)
        return self.classifier(x)

class SimpleCNN2(nn.Module):
    """Small CNN for CIFAR-10."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
            # nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 256), 
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class CIFAR10CNN2(nn.Module):
    """Deeper CNN for CIFAR-10 (replacement for SimpleCNN)."""
    def __init__(self, num_classes=10):
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 → 16x16

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 → 8x8

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 8x8 → 4x4
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10CNN, self).__init__()
        # Convolutional blocks
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # after 3 pools, input is 32→16→8→4
        self.fc2=nn.Linear(256,256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)   # 32→16

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)   # 16→8

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)   # 8→4

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x



# -------------------------------
# Dataset loaders
# -------------------------------
class ClinicalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def load_csv_generic(path, target_col, test_size=0.1, random_state=42):
    """
    Load CSV into (X_train, X_test, y_train, y_test).
    - Handles numeric + categorical features.
    - Encodes categorical features with one-hot encoding.
    - Encodes categorical targets into integers if needed.
    """
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Target {target_col} not in {path}")

    # --- Extract target y ---
    y_raw = df[target_col]
    if y_raw.dtype.kind in "OUS":  # string/object → categorical ints
        y = pd.factorize(y_raw.astype(str).str.strip())[0]
    else:
        y = pd.to_numeric(y_raw, errors='coerce').fillna(0).astype(int).values

    # --- Extract features X ---
    Xdf = df.drop(columns=[target_col])

    # Separate numeric + categorical features
    numeric_feats = Xdf.select_dtypes(include=[np.number])
    categorical_feats = Xdf.select_dtypes(exclude=[np.number])

    # One-hot encode categoricals if present
    if not categorical_feats.empty:
        categorical_encoded = pd.get_dummies(categorical_feats.astype(str), drop_first=False)
        Xdf = pd.concat([numeric_feats, categorical_encoded], axis=1)
    else:
        Xdf = numeric_feats

    # Convert to NumPy
    X = Xdf.values.astype(float)

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, list(Xdf.columns)

# def load_csv_generic(path, target_col):
#     df = pd.read_csv(path)
#     if target_col not in df.columns:
#         raise ValueError(f"Target {target_col} not in {path}")
#     # drop non-numerics except target, then cast target to categorical ints if needed
#     y_raw = df[target_col]
#     if y_raw.dtype.kind in "OUS":  # string / object → factorize
#         y = pd.factorize(y_raw.astype(str).str.strip())[0]
#     else:
#         y = pd.to_numeric(y_raw, errors='coerce').fillna(0).astype(int).values
#     Xdf = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
#     X = Xdf.values.astype(float)
#     return X, y

def load_mnist(binary=False):
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test  = datasets.MNIST("./data", train=False, download=True, transform=transform)
    if binary:
        train_idx = [i for i,t in enumerate(train.targets) if t in [0,1]]
        test_idx  = [i for i,t in enumerate(test.targets) if t in [0,1]]
        train.data, train.targets = train.data[train_idx], train.targets[train_idx]
        test.data, test.targets   = test.data[test_idx], test.targets[test_idx]
        num_classes = 2
    else:
        num_classes = 10
    input_dim = 28*28
    return train, test, input_dim, num_classes

def load_cifar10(binary=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    # transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    test  = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    if binary:
        train_idx = [i for i,t in enumerate(train.targets) if t in [3,5]] # cat=3, dog=5
        test_idx  = [i for i,t in enumerate(test.targets) if t in [3,5]]
        train.data, train.targets = train.data[train_idx], np.array(train.targets)[train_idx]
        test.data, test.targets   = test.data[test_idx], np.array(test.targets)[test_idx]
        num_classes = 2
    else:
        num_classes = 10
    input_dim = 3*32*32
    return train, test, input_dim, num_classes


def load_pathmnist(binary=False, image_size=28):
    require_medmnist()
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Force RGB
    transforms.Normalize(mean=[.5], std=[.5])
    ])
     # ✅ Fix target shape here using target_transform
    target_transform = transforms.Lambda(lambda y: int(y.squeeze().item()))

    # transform = transforms.Compose([
    #     transforms.Resize((image_size, image_size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[.5], std=[.5])
    # ])

    train = PathMNIST(split="train", download=True, transform=transform, target_transform=target_transform)
    test  = PathMNIST(split="test", download=True, transform=transform, target_transform=target_transform)


    if binary:
        # Keep only classes 0 and 1
        train_idx = np.where((train.labels == 0) | (train.labels == 1))[0]
        test_idx  = np.where((test.labels == 0) | (test.labels == 1))[0]
        train.imgs, train.labels = train.imgs[train_idx], train.labels[train_idx]
        test.imgs, test.labels   = test.imgs[test_idx], test.labels[test_idx]
        num_classes = 2
    else:
        num_classes = len(np.unique(train.labels))

    # Get input shape
    x0, _ = train[0]
    in_channels = x0.shape[0]
    img_size = x0.shape[1]

    return train, test, img_size, num_classes, in_channels

def load_organamnist(image_size=28):
    require_medmnist()
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # Output shape = [1, 28, 28] (grayscale)
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # Target transform (convert to integer class index)
    # target_transform = transforms.Lambda(lambda y: int(y))
    target_transform = transforms.Lambda(lambda y: int(y.squeeze().item()))

    # Load dataset
    train = OrganAMNIST(split="train", download=True, transform=transform, target_transform=target_transform)
    test  = OrganAMNIST(split="test", download=True, transform=transform, target_transform=target_transform)

    # Metadata
    num_classes = len(np.unique(train.labels))  # 11 classes
    in_channels = 1  # grayscale images
    img_size = image_size

    return train, test, img_size, num_classes, in_channels

def load_tissuemnist(image_size=28):
    require_medmnist()
    # Transform for grayscale medical images
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # Output shape = [1, H, W]
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # Fix target format: convert to integer labels
    # target_transform = transforms.Lambda(lambda y: int(y))
    target_transform = transforms.Lambda(lambda y: int(y.squeeze().item()))

    # Load dataset splits
    train = TissueMNIST(split="train", download=True, transform=transform, target_transform=target_transform)
    test  = TissueMNIST(split="test",  download=True, transform=transform, target_transform=target_transform)

    num_classes = len(np.unique(train.labels))  # Should return 8
    in_channels = 1  # Grayscale images

    return train, test, image_size, num_classes, in_channels

def load_organsmnist(image_size=28):
    require_medmnist()
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # Output shape = [1, 28, 28] (grayscale)
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # Target transform (convert to integer class index)
    # target_transform = transforms.Lambda(lambda y: int(y))
    target_transform = transforms.Lambda(lambda y: int(y.squeeze().item()))

    # Load dataset
    train = OrganSMNIST(split="train", download=True, transform=transform, target_transform=target_transform)
    test  = OrganSMNIST(split="test", download=True, transform=transform, target_transform=target_transform)

    # Metadata
    num_classes = len(np.unique(train.labels))  # 11 classes
    in_channels = 1  # grayscale images
    img_size = image_size

    return train, test, img_size, num_classes, in_channels


# -------------------------------
# Partitioning
# -------------------------------
def dirichlet_partition(y, num_clients, alpha=0.5, min_size=10):
    y = np.array(y)
    labels = np.unique(y)
    idx_by_class = {c: np.where(y==c)[0].tolist() for c in labels}
    client_indices = [[] for _ in range(num_clients)]
    for c in labels:
        idx_c = idx_by_class[c]
        if len(idx_c)==0: continue
        proportions = np.random.dirichlet([alpha]*num_clients)
        counts = (proportions*len(idx_c)).astype(int)
        while counts.sum() < len(idx_c):
            counts[np.argmax(proportions)] += 1
        start = 0
        for i, cnt in enumerate(counts):
            client_indices[i].extend(idx_c[start:start+cnt]); start += cnt
    # ensure min
    n = len(y)
    for i in range(num_clients):
        while len(client_indices[i]) < min_size:
            donor = max(range(num_clients), key=lambda j: len(client_indices[j]))
            if donor==i or len(client_indices[donor])<=min_size: break
            client_indices[i].append(client_indices[donor].pop())
        if len(client_indices[i])==0:
            client_indices[i] = list(np.random.choice(n, min_size, replace=False))
    return client_indices

# -------------------------------
# Evaluation
# -------------------------------
def evaluate_model_state(state, model_fn, dataset, batch_size=256, device=DEVICE, is_binary=False):
    model = model_fn().to(device); model.load_state_dict(state); model.eval()
    loss_fn = nn.CrossEntropyLoss()
    ys, preds, probpos = [], [], []
    loss=0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb, yb in loader:
            # yb = yb.squeeze().long()
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(1)
            loss += loss_fn(logits, yb)
            ys.append(yb.cpu().numpy()); preds.append(pred.cpu().numpy())
            if is_binary:
                p = torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
                probpos.append(p)
    y = np.concatenate(ys); p = np.concatenate(preds)
    acc = accuracy_score(y, p)
    if is_binary and len(np.unique(y))>1:
        try:
            auc = roc_auc_score(y, np.concatenate(probpos))
        except Exception:
            auc = 0.0
        prec, rec, f1, _ = precision_recall_fscore_support(y, p, average='binary', zero_division=0)
    else:
        auc = 0.0
        prec, rec, f1, _ = precision_recall_fscore_support(y, p, average='macro', zero_division=0)
    return {'acc':float(acc), 'auc':float(auc), 'precision':float(prec), 'recall':float(rec), 'f1':float(f1),'loss':float(loss/len(loader))}

# -------------------------------
# Client / Validator / PBFT / DP
# -------------------------------
def state_to_vector(state:Dict[str,torch.Tensor])->torch.Tensor:
    return torch.cat([v.view(-1) for v in state.values()])

class ClientConfig:
    def __init__(self, local_epochs=1, batch_size=32, lr=0.01, mu=0.0, clip_norm=1.0, noise_multiplier=0.0):
        self.local_epochs=local_epochs; self.batch_size=batch_size; self.lr=lr
        self.mu=mu; self.clip_norm=clip_norm; self.noise_multiplier=noise_multiplier

class ClientNode:
    def __init__(self, cid:int, dataset, model_fn, cfg:ClientConfig, device='cpu'):
        self.cid=cid; self.dataset=dataset; self.model_fn=model_fn; self.cfg=cfg; self.device=device

    def local_update(
        self,
        global_state:Dict[str,torch.Tensor],
        method:str,
        *,
        is_malicious: bool = False,
        attack_type: str = "none",
        num_classes: Optional[int] = None,
        label_flip_shift: int = 1,
        attack_scale: float = 5.0,
        attack_noise_std: float = 5.0,
        noise_multiplier: Optional[float] = None,
    ) -> Dict[str,torch.Tensor]:
        """Run a local client update and optionally submit a Byzantine-corrupted update.

        The returned object is a submitted model state, not a raw update.  Genuine
        Byzantine client attacks are applied to the model delta after local training
        except label-flipping, which corrupts the malicious client's local labels
        during training.
        """
        model = self.model_fn().to(self.device); model.load_state_dict(global_state); model.train()
        loss_fn = nn.CrossEntropyLoss()
        opt = optim.SGD(model.parameters(), lr=self.cfg.lr, momentum=0.9)
        loader = self.dataset
        attack_type = (attack_type or "none").lower()
        effective_noise = self.cfg.noise_multiplier if noise_multiplier is None else float(noise_multiplier)
        if method=='fedprox' and self.cfg.mu>0.0:
            global_vec = state_to_vector(global_state).to(self.device)

        for ep in range(self.cfg.local_epochs):
            if getattr(self.cfg, 'verbose_local', False):
                print('starting epoch ',ep)
                print(f'loader len = {len(loader)}')
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device).long()
                if is_malicious and attack_type == 'label_flip' and num_classes is not None and num_classes > 1:
                    yb = (yb + int(label_flip_shift)) % int(num_classes)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                if method=='fedprox' and self.cfg.mu>0.0:
                    local_vec = state_to_vector({k:v for k,v in model.state_dict().items()})
                    loss = loss + (self.cfg.mu/2.0) * torch.norm(local_vec - global_vec)**2
                loss.backward()
                opt.step()
        new_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        # Build clipped + noisy update. This is also where heterogeneous DP budgets enter.
        update = {k:(new_state[k]-global_state[k]).detach().cpu() for k in global_state}
        flat = torch.cat([v.view(-1) for v in update.values()])
        l2 = float(torch.norm(flat).item()) + 1e-12
        scale = min(1.0, self.cfg.clip_norm / l2)
        for k in update:
            update[k] = update[k] * scale
            if effective_noise > 0.0:
                sigma = effective_noise * self.cfg.clip_norm
                update[k] = update[k] + torch.normal(0.0, sigma, size=update[k].shape)

        # Byzantine update-space attacks. Label flipping has already been applied above.
        if is_malicious and attack_type not in ('none', 'label_flip'):
            if attack_type == 'sign_flip':
                update = {k: -float(attack_scale) * v for k, v in update.items()}
            elif attack_type == 'scaling':
                update = {k: float(attack_scale) * v for k, v in update.items()}
            elif attack_type == 'random_update':
                update = {k: float(attack_scale) * torch.randn_like(v) for k, v in update.items()}
            elif attack_type == 'gaussian_model_poisoning':
                update = {k: v + float(attack_noise_std) * torch.randn_like(v) for k, v in update.items()}
            else:
                raise ValueError(f"Unknown attack_type: {attack_type}")

        return {k:(global_state[k]+update[k]).detach().cpu() for k in update}

    
class ValidatorNode:
    def __init__(self, vid:int, dataset, model_fn, is_binary:bool, device='cpu'):
        self.vid=vid; self.dataset=dataset; self.model_fn=model_fn; self.is_binary=is_binary; self.device=device
    def eval_state(self, state:Dict[str,torch.Tensor]) -> Dict[str,float]:
        return evaluate_model_state(state, self.model_fn, self.dataset, device=self.device, is_binary=self.is_binary)

class Ledger:
    def __init__(self): self.chain=[]
    def append(self, block:dict): self.chain.append(block)
    def to_df(self): return pd.DataFrame(self.chain)

def pbft_decide(votes:dict)->bool:
    n=len(votes); f=(n-1)//3; agrees=sum(1 for v in votes.values() if v)
    return agrees >= (2*f + 1)

def gaussian_rdp(sigma:float, alpha:float)->float:
    if sigma<=0: return float('inf')
    return alpha/(2.0*sigma**2)

def compute_rdp_accountant(noise_multiplier:float, steps:int, delta:float=1e-5)->Tuple[float,float]:
    orders=list(range(2,65))
    per=[gaussian_rdp(noise_multiplier,a) for a in orders]
    total=[p*steps for p in per]
    eps=min([r + math.log(1/delta)/(a-1) for r,a in zip(total,orders) if a>1], default=float('inf'))
    return eps, delta

def temperature_at_round(r:int, total:int, t_start:float, t_end:float)->float:
    """Monotone schedule from t_start (round 1) to t_end (round total)."""
    if total <= 1: return max(t_end, 1e-6)
    frac = (r-1) / (total-1)
    return max(t_end + (t_start - t_end) * (1.0 - frac), 1e-6)

def temperature_schedule(r:int, R:int, t_start:float, t_mid:float, t_end:float) -> float:
    """Three-point schedule: start -> mid (~50%) -> end. S2 multiplies by 0.8 later."""
    if R <= 1: return max(t_end, 1e-6)
    frac = (r-1)/(R-1)
    if frac <= 0.5:
        a = frac/0.5
        return max(t_start*(1-a) + t_mid*a, 1e-6)
    else:
        a = (frac-0.5)/0.5
        return max(t_mid*(1-a) + t_end*a, 1e-6)

def l2_flat(state_dict:Dict[str,torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.detach().float().view(-1) for v in state_dict.values()])

def clip_update(delta:Dict[str,torch.Tensor], max_norm:float) -> Dict[str,torch.Tensor]:
    flat = l2_flat(delta); n = float(torch.norm(flat) + 1e-12)
    if n <= max_norm: return delta
    scale = max_norm / n
    return {k:(v*scale) for k,v in delta.items()}


# -------------------------------
# Reviewer-response extensions:
# Byzantine clients, heterogeneous DP and fairness metrics
# -------------------------------
ATTACK_TYPES = (
    "none", "label_flip", "sign_flip", "scaling",
    "random_update", "gaussian_model_poisoning"
)

ROBUST_METHODS = ("krum", "multikrum", "median", "trimmed_mean", "bulyan")

ADAPTIVE_METHODS = {
    "adaptive": "full",
    "adaptive_no_consensus": "no_consensus",
    "adaptive_consensus_only": "consensus_only",
    "adaptive_quality_only": "quality_only",
    "adaptive_shapley_only": "shapley_only",
    "adaptive_no_quality": "no_quality",
}

def parse_float_list(text: str) -> List[float]:
    if text is None or str(text).strip() == "":
        return []
    return [float(x.strip()) for x in str(text).split(',') if x.strip() != ""]

def select_malicious_clients(num_clients: int, malicious_ratio: float, seed: int) -> Set[int]:
    malicious_ratio = max(0.0, min(1.0, float(malicious_ratio)))
    n_bad = int(round(malicious_ratio * num_clients))
    if n_bad <= 0:
        return set()
    rng = np.random.default_rng(seed + 137)
    return set(rng.choice(np.arange(num_clients), size=n_bad, replace=False).tolist())

def build_client_noise_multipliers(args, num_clients: int, malicious_clients: Optional[Set[int]] = None) -> Dict[int, float]:
    """Return per-client DP noise multipliers.

    When --heterogeneous_dp is enabled, clients are divided across the values in
    --dp_noise_multipliers. This simulates hospitals with different privacy
    requirements. The default values correspond to relaxed/moderate/strict DP.
    """
    malicious_clients = malicious_clients or set()
    if not getattr(args, 'heterogeneous_dp', False):
        return {cid: float(args.base_noise) for cid in range(num_clients)}

    levels = parse_float_list(getattr(args, 'dp_noise_multipliers', ''))
    if not levels:
        levels = [float(args.base_noise), 5.0 * float(args.base_noise), 10.0 * float(args.base_noise)]
    levels = [max(0.0, float(v)) for v in levels]
    out = {}
    for cid in range(num_clients):
        out[cid] = levels[cid % len(levels)]
    return out

def state_delta(submitted_state: Dict[str, torch.Tensor], base_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: (submitted_state[k] - base_state[k]).detach().cpu() for k in base_state}

def l2_state_distance(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    return float(torch.norm(l2_flat({k: a[k] - b[k] for k in a.keys()})).item())

def normalize_nonnegative(scores: Dict[int, float], cids: Optional[List[int]] = None) -> Dict[int, float]:
    cids = sorted(scores.keys()) if cids is None else list(cids)
    vals = np.array([max(0.0, float(scores.get(cid, 0.0))) for cid in cids], dtype=np.float64)
    if vals.sum() <= 1e-12 and len(vals) > 0:
        vals = np.ones_like(vals) / len(vals)
    elif len(vals) > 0:
        vals = vals / vals.sum()
    return {cid: float(vals[i]) for i, cid in enumerate(cids)}

def safe_corr(x, y, method='pearson') -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 1.0 if np.allclose(x, y) else 0.0
    if method == 'spearman':
        xr = pd.Series(x).rank(method='average').to_numpy()
        yr = pd.Series(y).rank(method='average').to_numpy()
        return float(np.corrcoef(xr, yr)[0, 1])
    return float(np.corrcoef(x, y)[0, 1])

def reward_fairness_metrics(contrib: Dict[int, float], rewards: Dict[int, float]) -> Dict[str, float]:
    cids = sorted(set(contrib.keys()) | set(rewards.keys()))
    c = np.array([max(0.0, float(contrib.get(cid, 0.0))) for cid in cids], dtype=np.float64)
    r = np.array([max(0.0, float(rewards.get(cid, 0.0))) for cid in cids], dtype=np.float64)
    if c.sum() > 0: c = c / c.sum()
    if r.sum() > 0: r = r / r.sum()
    return {
        'reward_contrib_l1': float(np.abs(c-r).sum()),
        'reward_contrib_l2': float(np.sqrt(((c-r)**2).sum())),
        'reward_contrib_pearson': safe_corr(c, r, 'pearson'),
        'reward_contrib_spearman': safe_corr(c, r, 'spearman'),
        'reward_gini': gini_coefficient(r) if 'gini_coefficient' in globals() else 0.0,
        'contrib_gini': gini_coefficient(c) if 'gini_coefficient' in globals() else 0.0,
    }

def group_reward_gap(values_by_client: Dict[int, float], groups_by_client: Dict[int, float]) -> float:
    """Max-min group mean; useful for heterogeneous-DP fairness."""
    groups = sorted(set(groups_by_client.values()))
    means = []
    for g in groups:
        vals = [values_by_client[cid] for cid in values_by_client if groups_by_client.get(cid) == g]
        if vals:
            means.append(float(np.mean(vals)))
    return float(max(means) - min(means)) if len(means) >= 2 else 0.0


# -------------------------------
# kNN-Shapley proxy (client-model centric)
# -------------------------------

def knn_shapley_clients(submissions: Dict[int, Dict[str, torch.Tensor]],
                        val_ds,
                        model_fn,
                        k: int = 5,
                        n_jobs: int = 4,
                        sample: int = 512,
                        device: str = 'cpu',
                        is_binary: bool = False) -> Dict[int, float]:
    # subsample validation
    n = len(val_ds)
    idx = np.random.choice(n, size=min(sample, n), replace=False)
    sub = Subset(val_ds, idx.tolist())
    client_ids = sorted(list(submissions.keys()))

    # probe output dimension
    xb0, _ = next(iter(DataLoader(sub, batch_size=1, shuffle=False)))
    with torch.no_grad():
        out0 = model_fn().to(device)(xb0.to(device))
    num_classes = out0.shape[1]

    # ---- Parallelize predictions per client ----
    def eval_client(cid):
        model = model_fn().to(device)
        model.load_state_dict(submissions[cid])
        model.eval()
        probs_all, ys_local = [], []
        with torch.no_grad():
            for xb, yb in DataLoader(sub, batch_size=256, shuffle=False):
                logits = model(xb.to(device)).detach().cpu()
                probs = torch.softmax(logits, dim=1).numpy()
                probs_all.append(probs)
                if cid == client_ids[0]:  # only first client collects y
                    ys_local.append(yb.numpy())
        return np.vstack(probs_all), (np.concatenate(ys_local) if ys_local else None)

    workers = len(client_ids) if n_jobs == -1 else min(n_jobs, len(client_ids))
    with parallel_backend('threading', n_jobs=n_jobs):
        results = Parallel()(
            delayed(eval_client)(cid) for cid in client_ids
            )
        # n_jobs=workers, prefer="threads"
    # results = Parallel(n_jobs=len(client_ids), prefer="threads")(
    #     delayed(eval_client)(cid) for cid in client_ids
    # )

    P = np.zeros((len(client_ids), len(sub), num_classes), dtype=np.float32)
    ys = None
    for j, (probs, y) in enumerate(results):
        P[j] = probs
        if ys is None and y is not None:
            ys = y
    y = ys

    # ---- Vectorized kNN evaluation ----
    # P shape: [n_clients, n_samples, num_classes]
    # For each sample t: we want neighbors among client predictions [n_clients, num_classes]
    vals = np.zeros(len(client_ids), dtype=np.float64)

    for t in range(len(sub)):
        feats = P[:, t, :]                 # [n_clients, num_classes]
        corr = (feats.argmax(axis=1) == y[t]).astype(int)

        # --- sanitize feats ---
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.maximum(norms, 1e-12)


        # compute pairwise distances (clients × clients)
        D = pairwise_distances(feats, feats, metric="euclidean")

        # sort neighbors (excluding self)
        neigh_idx = np.argsort(D, axis=1)[:, 1:k+1]   # top-k neighbors for each client
        local_vals = corr[neigh_idx].mean(axis=1)
        vals += local_vals

    vals /= float(len(sub))
    if vals.max() > 0:
        vals = vals / vals.max()
    return {cid: float(vals[j]) for j, cid in enumerate(client_ids)}

# -------------------------------
# Aggregation
# -------------------------------
def fedavg_aggregate(submissions:Dict[int,Dict[str,torch.Tensor]]) -> Dict[str,torch.Tensor]:
    out = copy.deepcopy(next(iter(submissions.values())))
    for k in out: out[k]=out[k].zero_()
    for st in submissions.values():
        for k in out: out[k] += st[k]
    for k in out: out[k] /= float(len(submissions))
    return out

def coordinate_median_aggregate(submissions:Dict[int,Dict[str,torch.Tensor]]) -> Dict[str,torch.Tensor]:
    out = copy.deepcopy(next(iter(submissions.values())))
    for k in out:
        stack = torch.stack([st[k].float() for st in submissions.values()], dim=0)
        out[k] = torch.median(stack, dim=0).values.type_as(out[k])
    return out

def trimmed_mean_aggregate(submissions:Dict[int,Dict[str,torch.Tensor]], trim_ratio:float=0.2) -> Dict[str,torch.Tensor]:
    n = len(submissions)
    f = int(math.floor(max(0.0, min(0.49, float(trim_ratio))) * n))
    if n - 2*f <= 0:
        f = 0
    out = copy.deepcopy(next(iter(submissions.values())))
    for k in out:
        stack = torch.stack([st[k].float() for st in submissions.values()], dim=0)
        if f > 0:
            stack, _ = torch.sort(stack, dim=0)
            stack = stack[f:n-f]
        out[k] = stack.mean(dim=0).type_as(out[k])
    return out

def krum_scores(submissions:Dict[int,Dict[str,torch.Tensor]], f:int=1) -> Dict[int, float]:
    cids = sorted(submissions.keys())
    n = len(cids)
    if n == 0:
        return {}
    f = int(max(0, min(f, max(0, (n-3)//2))))
    vecs = [l2_flat(submissions[cid]).float() for cid in cids]
    D = torch.zeros((n,n), dtype=torch.float64)
    for i in range(n):
        for j in range(i+1, n):
            d = torch.sum((vecs[i]-vecs[j])**2).item()
            D[i,j] = D[j,i] = d
    nb = max(1, n - f - 2)
    scores = {}
    for i, cid in enumerate(cids):
        dists = torch.sort(D[i][D[i] > 0]).values
        if len(dists) == 0:
            score = 0.0
        else:
            score = float(dists[:min(nb, len(dists))].sum().item())
        scores[cid] = score
    return scores

def krum_aggregate(submissions:Dict[int,Dict[str,torch.Tensor]], f:int=1) -> Dict[str,torch.Tensor]:
    if len(submissions) <= 2*f + 2:
        return coordinate_median_aggregate(submissions)
    scores = krum_scores(submissions, f=f)
    chosen = min(scores, key=scores.get)
    return copy.deepcopy(submissions[chosen])

def multikrum_aggregate(submissions:Dict[int,Dict[str,torch.Tensor]], f:int=1, m:int=0) -> Dict[str,torch.Tensor]:
    n = len(submissions)
    if n <= 2*f + 2:
        return coordinate_median_aggregate(submissions)
    scores = krum_scores(submissions, f=f)
    if m <= 0:
        m = max(1, n - f - 2)
    selected = sorted(scores, key=scores.get)[:min(m, n)]
    return fedavg_aggregate({cid: submissions[cid] for cid in selected})

def bulyan_aggregate(submissions:Dict[int,Dict[str,torch.Tensor]], f:int=1, trim_ratio:float=0.2) -> Dict[str,torch.Tensor]:
    n = len(submissions)
    # Approximate Bulyan: Multi-Krum preselection followed by coordinate-wise trimmed mean.
    # Falls back gracefully when n is too small for the formal n >= 4f+3 condition.
    if n < max(3, 4*f + 3):
        return trimmed_mean_aggregate(submissions, trim_ratio=trim_ratio)
    scores = krum_scores(submissions, f=f)
    m = max(1, n - 2*f)
    selected = sorted(scores, key=scores.get)[:m]
    return trimmed_mean_aggregate({cid: submissions[cid] for cid in selected}, trim_ratio=trim_ratio)

def robust_aggregate(method:str, submissions:Dict[int,Dict[str,torch.Tensor]], args) -> Dict[str,torch.Tensor]:
    n = len(submissions)
    f = int(getattr(args, 'robust_f', -1))
    if f < 0:
        f = int(math.floor(float(getattr(args, 'malicious_ratio', 0.0)) * n))
    f = max(0, min(f, max(0, n-1)))
    method = method.lower()
    if method == 'median':
        return coordinate_median_aggregate(submissions)
    if method == 'trimmed_mean':
        return trimmed_mean_aggregate(submissions, trim_ratio=getattr(args, 'trim_ratio', 0.2))
    if method == 'krum':
        return krum_aggregate(submissions, f=f)
    if method == 'multikrum':
        return multikrum_aggregate(submissions, f=f, m=getattr(args, 'multikrum_m', 0))
    if method == 'bulyan':
        return bulyan_aggregate(submissions, f=f, trim_ratio=getattr(args, 'trim_ratio', 0.2))
    raise ValueError(f"Unknown robust aggregation method: {method}")

def intent_aware_aggregate(
    submissions, shapley, relevance, accepted, 
    alpha, beta, temperature,
    model_gain=None, trust_scores=None, max_selected=0,
):
    """Build logits over accepted clients and softmax with temperature."""
    model_gain = model_gain or {}
    trust_scores = trust_scores or {}
    cids_all = sorted(list(submissions.keys()))
    cids = [cid for cid in cids_all if accepted.get(cid, False)]
    if len(cids)==0:  # fallback to all
        cids = cids_all[:]

    raw = []
    for cid in cids:
        s = float(max(0.0, shapley.get(cid, 0.0)))
        r = float(relevance.get(cid, 0.0))
        g = float(model_gain.get(cid, 0.0))
        t = float(trust_scores.get(cid, 0.0))
        raw.append(alpha*s + beta*r + g + t)
    raw = np.array(raw, dtype=np.float64)

    if max_selected and len(raw) > max_selected:
        top_idx = np.argsort(-raw)[:max_selected]
        cids = [cids[i] for i in top_idx]
        raw  = raw[top_idx]

    tau = max(1e-6, float(temperature))
    logits = raw / tau
    logits = logits - logits.max()
    exps = np.exp(logits)
    weights = exps / np.clip(exps.sum(), 1e-12, None)

    out = copy.deepcopy(next(iter(submissions.values())))
    for k in out: out[k] = out[k].zero_()
    for w, cid in zip(weights, cids):
        for k in out: out[k] += float(w) * submissions[cid][k]
    return out, weights, cids, raw


def clinical_relevance(metrics, domain):
    d = domain.lower()
    # if d.startswith('heart'): return 0.1*metrics['recall'] + 0.9*metrics['auc']
    # # if d.startswith('heart'): return 0.9*metrics['acc'] + 0.1*metrics['recall']
    # if d.startswith('diab'):  return 0.5*metrics['auc'] + 0.5*metrics['f1']
    # # if d.startswith('diab'):  return 1.0*metrics['acc'] + 0.0*metrics['auc']
    # if d.startswith('lung'):
    #     if metrics['auc'] > 0:  # binary case
    #         return 0.8*metrics['recall'] + 0.2*metrics['acc']
    #     else:  # multiclass Stage
    #         return 0.8*metrics['acc'] + 0.2*metrics['f1']
    if d.startswith('pathmnist'):
        if metrics['auc'] > 0:  # binary case
            return 0.8*metrics['recall'] + 0.2*metrics['acc']
        else:  # multiclass Stage
            return 1.0*metrics['acc'] + 0.0*metrics['f1']
    if d.startswith('tissuemnist'):
        if metrics['auc'] > 0:  # binary case
            return 0.8*metrics['recall'] + 0.2*metrics['acc']
        else:  # multiclass Stage
            return 0.9*metrics['acc'] + 0.1*metrics['f1']
    if d.startswith('organamnist'):
        if metrics['auc'] > 0:  # binary case
            return 0.8*metrics['recall'] + 0.2*metrics['acc']
        else:  # multiclass Stage
            return 0.9*metrics['acc'] + 0.1*metrics['f1']
    if d.startswith('organsmnist'):
        if metrics['auc'] > 0:  # binary case
            return 0.8*metrics['recall'] + 0.2*metrics['acc']
        else:  # multiclass Stage
            return 0.9*metrics['acc'] + 0.1*metrics['f1']
    # Generic multiclass
    return 0.9*metrics['acc'] + 0.1*metrics['f1']

# -------------------------------
# Single-trial runner (all methods)
# -------------------------------
def run_single_trial(args, seed:int, run_methods:List[str]):
    set_seed(seed)

    # ----- Load chosen dataset -----
    is_image = args.dataset in ['mnist','cifar10']
    # if args.dataset=="lung":
    #     X_train, X_test, y_train, y_test,_ =load_csv_generic(args.lung_csv,args.lung_target)
    #     scaler=StandardScaler().fit(X_train); X_train=scaler.transform(X_train)
    #     client_idx = dirichlet_partition(y_train,args.clients,args.dirichlet_alpha)
    #     clients=[DataLoader(ClinicalDataset(X_train[idx],y_train[idx]), batch_size=args.batch_size, shuffle=True) for idx in client_idx]
    #     model_fn=lambda: TabularMLP(X_train.shape[1],hidden=256,num_classes=len(np.unique(y_train)))
    #     scaler=StandardScaler().fit(X_test); X_test=scaler.transform(X_test)
    #     testset = ClinicalDataset(X_test,y_test)
    #     is_binary = (len(np.unique(y_train))==2)
    #     domain_for_rel = 'lung'
    # elif args.dataset=="heart":
    #     X_train, X_test, y_train, y_test,_=load_csv_generic(args.heart_csv,args.heart_target)
    #     scaler=StandardScaler().fit(X_train); X_train=scaler.transform(X_train)
    #     client_idx = dirichlet_partition(y_train,args.clients,args.dirichlet_alpha)
    #     clients=[DataLoader(ClinicalDataset(X_train[idx],y_train[idx]), batch_size=args.batch_size, shuffle=True) for idx in client_idx]
    #     model_fn=lambda: TabularMLP(X_train.shape[1],hidden=128,num_classes=len(np.unique(y_train)))
    #     scaler=StandardScaler().fit(X_test); X_test=scaler.transform(X_test)
    #     testset = ClinicalDataset(X_test,y_test)
    #     is_binary = True
    #     domain_for_rel = 'heart'
    # elif args.dataset=="diabetes":
    #     X_train, X_test, y_train, y_test,_ =load_csv_generic(args.diabetes_csv,args.diabetes_target, test_size=0.2)
    #     scaler=StandardScaler().fit(X_train); X_train=scaler.transform(X_train)
    #     client_idx = dirichlet_partition(y_train,args.clients,args.dirichlet_alpha)
    #     clients=[DataLoader(ClinicalDataset(X_train[idx],y_train[idx]), batch_size=args.batch_size, shuffle=True) for idx in client_idx]
    #     model_fn=lambda: TabularMLP(X_train.shape[1],hidden=128,num_classes=len(np.unique(y_train)))
    #     scaler=StandardScaler().fit(X_test); X_test=scaler.transform(X_test)
    #     testset = ClinicalDataset(X_test,y_test)
    #     is_binary = True
    #     domain_for_rel = 'diabetes'
    if args.dataset=="mnist":
        train,test,input_dim,num_classes = load_mnist(binary=args.binary)
        y = train.targets.numpy()
        client_idx = dirichlet_partition(y,args.clients,args.dirichlet_alpha)
        # clients=[Subset(train,idx) for idx in client_idx]
        clients=[DataLoader(Subset(train,idx), batch_size=args.batch_size, shuffle=True) for idx in client_idx]
        model_fn=lambda: MNISTCNN(num_classes=num_classes)
        testset=test
        is_binary = (num_classes==2)
        domain_for_rel = 'generic'
    elif args.dataset=="cifar10":
        train,test,input_dim,num_classes = load_cifar10(binary=args.binary)
        y = np.array(train.targets)
        client_idx = dirichlet_partition(y,args.clients,args.dirichlet_alpha)
        clients=[DataLoader(Subset(train,idx), batch_size=args.batch_size, shuffle=True) for idx in client_idx]
        model_fn=lambda: SimpleCNN(num_classes=num_classes)
        testset=test
        is_binary = (num_classes==2)
        domain_for_rel = 'generic'
    elif args.dataset=="pathmnist":
        # train, test, img_size, num_classes, in_channels
        train, test, img_size, num_classes,in_channels = load_pathmnist()
        print(f"image size= {img_size}, num_classes= {num_classes}, in_channels= {in_channels}")
        y = np.array([t[1] for t in train])
        client_idx = dirichlet_partition(y, args.clients, args.dirichlet_alpha)
        clients = [DataLoader(Subset(train, idx), batch_size=args.batch_size, shuffle=True) for idx in client_idx]
        model_fn = lambda: SimpleCNN_MedMNIST(in_channels=in_channels ,num_classes=num_classes)
        testset = test
        is_binary = (num_classes == 2)
        domain_for_rel = 'pathmnist'

    elif args.dataset=="tissuemnist":
        train, test, img_size, num_classes,in_channels = load_tissuemnist()
        print(f"image size= {img_size}, num_classes= {num_classes}, in_channels= {in_channels}")
        y = np.array([t[1] for t in train])
        client_idx = dirichlet_partition(y, args.clients, args.dirichlet_alpha)
        clients = [DataLoader(Subset(train, idx), batch_size=args.batch_size, shuffle=True) for idx in client_idx]
        model_fn = lambda: SimpleCNN_MedMNIST(in_channels=in_channels ,num_classes=num_classes)
        testset = test
        is_binary = (num_classes == 2)
        domain_for_rel = 'tissuemnist'

    elif args.dataset=="organamnist":
        train, test, img_size, num_classes,in_channels = load_organamnist()
        print(f"image size= {img_size}, num_classes= {num_classes}, in_channels= {in_channels}")
        y = np.array([t[1] for t in train])
        client_idx = dirichlet_partition(y, args.clients, args.dirichlet_alpha)
        clients = [DataLoader(Subset(train, idx), batch_size=args.batch_size, shuffle=True) for idx in client_idx]
        model_fn = lambda: SimpleCNN_MedMNIST(in_channels=in_channels ,num_classes=num_classes)
        testset = test
        is_binary = (num_classes == 2)
        domain_for_rel = 'organamnist'
    elif args.dataset=="organsmnist":
        train, test, img_size, num_classes,in_channels = load_organsmnist()
        print(f"image size= {img_size}, num_classes= {num_classes}, in_channels= {in_channels}")
        y = np.array([t[1] for t in train])
        client_idx = dirichlet_partition(y, args.clients, args.dirichlet_alpha)
        clients = [DataLoader(Subset(train, idx), batch_size=args.batch_size, shuffle=True) for idx in client_idx]
        model_fn = lambda: SimpleCNN_MedMNIST(in_channels=in_channels ,num_classes=num_classes)
        testset = test
        is_binary = (num_classes == 2)
        domain_for_rel = 'organsmnist'

    else:
        raise ValueError("Unknown dataset")
    

    val_indices = np.random.permutation(len(testset))
    validators=[]
    start=0
    for v in range(args.validators):
        take = min(args.validator_val_size, len(testset)-start)
        idx = val_indices[start:start+take]; start += take
        if len(idx)==0: idx = np.random.choice(len(testset), size=args.validator_val_size, replace=True)
        validators.append(ValidatorNode(v, Subset(testset, idx.tolist()), model_fn, is_binary=is_binary, device=str(DEVICE)))
    if len(validators)==0:
        validators.append(ValidatorNode(0, testset, model_fn, is_binary=is_binary, device=str(DEVICE)))

    # ----- Init server and clients -----
    server = model_fn().to(DEVICE)
    init_state = {k:v.detach().cpu().clone() for k,v in server.state_dict().items()}
    server_states = {m: copy.deepcopy(init_state) for m in run_methods}
    cfg = ClientConfig(local_epochs=args.local_epochs, batch_size=args.batch_size, lr=args.lr,
                       mu=args.fedprox_mu, clip_norm=args.clip_norm, noise_multiplier=args.base_noise)
    client_nodes = {cid: ClientNode(cid, clients[cid], model_fn, cfg, device=str(DEVICE)) for cid in range(args.clients)}

    # Genuine Byzantine-client setup. These are malicious FL clients, not merely non-IID clients.
    malicious_clients = select_malicious_clients(args.clients, args.malicious_ratio, seed)
    client_noise = build_client_noise_multipliers(args, args.clients, malicious_clients)
    if args.attack_type != 'none':
        print(f"Byzantine setup: attack={args.attack_type}, malicious_ratio={args.malicious_ratio}, malicious_clients={sorted(malicious_clients)}")
    if args.heterogeneous_dp:
        print(f"Heterogeneous DP noise multipliers: {client_noise}")

    # DP accounting
    best_eval_so_far = None  # optional tracking
    # Persistent trust memory & rolling baseline for PBFT for full/adaptive ablation variants.
    adaptive_like_methods = [m for m in run_methods if m in ADAPTIVE_METHODS]
    trust_scores = {m: defaultdict(float) for m in adaptive_like_methods}
    server_hist_for_pbft = {m: deque(maxlen=max(1, args.rolling_base_k)) for m in adaptive_like_methods}
    for m in adaptive_like_methods:
        server_hist_for_pbft[m].append(copy.deepcopy(server_states[m]))
    eps, delta = (compute_rdp_accountant(args.base_noise, args.rounds) if args.base_noise>0 else (float('inf'), 1.0))

    # ----- Histories -----
    histories = {m: [] for m in run_methods}
    ledger_all = Ledger()
    incentives_all_rounds = []

    # ----- Training rounds -----
    for r in range(1, args.rounds+1):
        subs_cache = {}
        latency_this_round, compute_time_this_round, comm_lat_this_round = {}, {}, {}

        # --- Each client trains once ---
        client_updates = {}

        # --- FedAvg aggregate ---
        if 'fedavg' in run_methods:
            client_updates = {cid: node.local_update(
                                    server_states['fedavg'], method='fedavg',
                                    is_malicious=(cid in malicious_clients), attack_type=args.attack_type,
                                    num_classes=num_classes, label_flip_shift=args.label_flip_shift,
                                    attack_scale=args.attack_scale, attack_noise_std=args.attack_noise_std,
                                    noise_multiplier=client_noise.get(cid, args.base_noise))
                              for cid,node in client_nodes.items()}
            new_state_fedavg = fedavg_aggregate(client_updates)
            h = evaluate_model_state(new_state_fedavg, model_fn, testset, device=DEVICE, is_binary=is_binary)
            h.update({'attack_type': args.attack_type, 'malicious_ratio': float(args.malicious_ratio),
                      'malicious_clients': len(malicious_clients), 'ablation': 'none'})
            histories['fedavg'].append(h)
            server_states['fedavg'] = new_state_fedavg

        # --- FedProx aggregate ---
        if 'fedprox' in run_methods:
            client_updates_prox = {cid: node.local_update(
                                        server_states['fedprox'], method='fedprox',
                                        is_malicious=(cid in malicious_clients), attack_type=args.attack_type,
                                        num_classes=num_classes, label_flip_shift=args.label_flip_shift,
                                        attack_scale=args.attack_scale, attack_noise_std=args.attack_noise_std,
                                        noise_multiplier=client_noise.get(cid, args.base_noise))
                                   for cid,node in client_nodes.items()}
            new_state_fedprox = fedavg_aggregate(client_updates_prox)
            h = evaluate_model_state(new_state_fedprox, model_fn, testset, device=DEVICE, is_binary=is_binary)
            h.update({'attack_type': args.attack_type, 'malicious_ratio': float(args.malicious_ratio),
                      'malicious_clients': len(malicious_clients), 'ablation': 'none'})
            histories['fedprox'].append(h)
            server_states['fedprox'] = new_state_fedprox
        # print('finished fedprox')

        # --- Byzantine-resilient robust aggregation baselines ---
        for robust_method in ROBUST_METHODS:
            if robust_method in run_methods:
                client_updates_robust = {cid: node.local_update(
                                            server_states[robust_method], method='fedavg',
                                            is_malicious=(cid in malicious_clients), attack_type=args.attack_type,
                                            num_classes=num_classes, label_flip_shift=args.label_flip_shift,
                                            attack_scale=args.attack_scale, attack_noise_std=args.attack_noise_std,
                                            noise_multiplier=client_noise.get(cid, args.base_noise))
                                         for cid, node in client_nodes.items()}
                new_state_robust = robust_aggregate(robust_method, client_updates_robust, args)
                h = evaluate_model_state(new_state_robust, model_fn, testset, device=DEVICE, is_binary=is_binary)
                h.update({'attack_type': args.attack_type, 'malicious_ratio': float(args.malicious_ratio),
                          'malicious_clients': len(malicious_clients), 'ablation': 'none'})
                histories[robust_method].append(h)
                server_states[robust_method] = new_state_robust

# --- FedSGD one-step ---
        if 'fedsgd' in run_methods:
            loss_fn = nn.CrossEntropyLoss()

            # Dict of aggregated gradients
            grads_sum = {name: torch.zeros_like(param, device='cpu')
                        for name, param in server_states['fedsgd'].items()}
            n_clients = 0

            for cid, node in client_nodes.items():
                loader = node.dataset  # Already a DataLoader
                try:
                    xb, yb = next(iter(loader))
                except StopIteration:
                    continue

                # Fix MedMNIST labels
                # yb = yb.squeeze().long()

                xb, yb = xb.to(DEVICE), yb.to(DEVICE).long()
                if cid in malicious_clients and args.attack_type == 'label_flip' and num_classes > 1:
                    yb = (yb + int(args.label_flip_shift)) % int(num_classes)
                local_model = model_fn().to(DEVICE)
                local_model.load_state_dict(server_states['fedsgd'])

                local_model.zero_grad(set_to_none=True)
                loss = loss_fn(local_model(xb), yb)
                loss.backward()

                # Accumulate gradients by key safely
                for (name, param) in local_model.named_parameters():
                    if param.grad is None:
                        continue
                    g = param.grad.detach().cpu()
                    if cid in malicious_clients and args.attack_type in ('sign_flip', 'scaling', 'random_update', 'gaussian_model_poisoning'):
                        if args.attack_type == 'sign_flip':
                            g = -float(args.attack_scale) * g
                        elif args.attack_type == 'scaling':
                            g = float(args.attack_scale) * g
                        elif args.attack_type == 'random_update':
                            g = float(args.attack_scale) * torch.randn_like(g)
                        elif args.attack_type == 'gaussian_model_poisoning':
                            g = g + float(args.attack_noise_std) * torch.randn_like(g)
                    if args.heterogeneous_dp and client_noise.get(cid, 0.0) > 0:
                        g = g + torch.normal(0.0, client_noise[cid] * args.clip_norm, size=g.shape)
                    if g.shape == grads_sum[name].shape:  # safety check
                        grads_sum[name] += g
                    else:
                        print(f"[WARNING] Skipping mismatched gradient for {name}: grad{tuple(g.shape)} vs param{tuple(grads_sum[name].shape)}")

                n_clients += 1

            # Build new FedSGD state
            new_state_fedsgd = copy.deepcopy(server_states['fedsgd'])
            if n_clients > 0:
                for name, param in new_state_fedsgd.items():
                    grad = grads_sum[name] / n_clients

                    # Clip
                    max_norm = getattr(args, 'clip_norm', 2.0)
                    grad_norm = grad.norm()
                    if grad_norm > max_norm:
                        grad = grad * (max_norm / (grad_norm + 1e-6))

                    # Gaussian Noise for DP
                    if getattr(args, 'base_noise', 0) > 0:
                        grad += torch.normal(
                            mean=0.0,
                            std=args.base_noise * max_norm,
                            size=grad.shape
                        )

                    # SGD update
                    new_state_fedsgd[name] = param - args.lr * grad

            # Evaluate
            h = evaluate_model_state(new_state_fedsgd, model_fn, testset, device=DEVICE, is_binary=is_binary)
            h.update({'attack_type': args.attack_type, 'malicious_ratio': float(args.malicious_ratio),
                      'malicious_clients': len(malicious_clients), 'ablation': 'none'})
            histories['fedsgd'].append(h)
            server_states['fedsgd'] = new_state_fedsgd


        # --- Adaptive full model and ablation variants --------------------------------
        for adaptive_method in adaptive_like_methods:
            ablation_mode = ADAPTIVE_METHODS[adaptive_method]
            prev_state = copy.deepcopy(server_states[adaptive_method])
            cid_order = []
            weights_adaptive = []
            accepted = {}

            client_updates = {}
            model_bytes = sum(v.numel() for v in server_states[adaptive_method].values()) * 4.0

            # (0) Rolling PBFT baseline (average of last-K server states)
            hist_deque = server_hist_for_pbft[adaptive_method]
            base_state_pbft = copy.deepcopy(hist_deque[0])
            if len(hist_deque) > 1:
                for k in base_state_pbft:
                    acc_param = torch.zeros_like(base_state_pbft[k])
                    for st in hist_deque:
                        acc_param += st[k]
                    base_state_pbft[k] = acc_param / float(len(hist_deque))

            # (1) Local updates + simulated communication latency.
            latency_this_round, compute_time_this_round, comm_lat_this_round = {}, {}, {}
            for cid, node in client_nodes.items():
                t0 = time.time()
                update = node.local_update(
                    server_states[adaptive_method], method='fedavg',
                    is_malicious=(cid in malicious_clients), attack_type=args.attack_type,
                    num_classes=num_classes, label_flip_shift=args.label_flip_shift,
                    attack_scale=args.attack_scale, attack_noise_std=args.attack_noise_std,
                    noise_multiplier=client_noise.get(cid, args.base_noise)
                )
                client_updates[cid] = update
                local_compute = time.time() - t0

                if args.latency_mode == 'simulate':
                    bandwidth_Bps = max(1e-6, args.bandwidth_mbps) * 1e6 / 8.0
                    tx_time = model_bytes / bandwidth_Bps
                    jitter = np.random.lognormal(mean=np.log(max(1e-3, args.latency_jitter_ms)), sigma=0.25) / 1000.0
                    comm_latency = tx_time + jitter
                else:
                    comm_latency = 0.0
                total_lat = local_compute + comm_latency
                compute_time_this_round[cid] = local_compute
                comm_lat_this_round[cid] = comm_latency
                latency_this_round[cid] = total_lat

            # (2) Base metrics once + relevance per-client.
            with parallel_backend('threading', n_jobs=args.n_jobs):
                base_metrics_all = Parallel(n_jobs=args.n_jobs, prefer="threads")(
                    delayed(v.eval_state)(server_states[adaptive_method]) for v in validators
                )
            base_key = 'auc' if is_binary else 'acc'
            base_val = float(np.mean([m[base_key] for m in base_metrics_all]))

            relevance = {}
            for cid, st in client_updates.items():
                with parallel_backend('threading', n_jobs=args.n_jobs):
                    metrics = Parallel(n_jobs=args.n_jobs, prefer="threads")(
                        delayed(v.eval_state)(st) for v in validators
                    )
                rel_vals = [clinical_relevance(m, domain_for_rel) for m in metrics]
                relevance[cid] = float(np.mean(rel_vals))

            # (3) PBFT model-quality validation.
            def eval_client_candidate(cid, st):
                # Evaluate the submitted candidate model directly against the current baseline.
                cand_metrics = [v.eval_state(st) for v in validators]
                gains = [cm[base_key] - bm[base_key] for cm, bm in zip(cand_metrics, base_metrics_all)]
                abs_gain = float(np.mean(gains))
                votes = {}
                for vid, (cm, bm) in enumerate(zip(cand_metrics, base_metrics_all)):
                    diff = cm[base_key] - bm[base_key]
                    vote = (diff >= args.pbft_acceptance_delta)
                    # This parameter simulates Byzantine validators, distinct from malicious FL clients.
                    if random.random() < args.pbft_byzantine_rate:
                        vote = random.choice([True, False])
                    votes[vid] = vote
                return cid, votes, abs_gain

            with parallel_backend('threading', n_jobs=args.n_jobs):
                results = Parallel(n_jobs=args.n_jobs, prefer="threads")(
                    delayed(eval_client_candidate)(cid, st) for cid, st in client_updates.items()
                )

            model_gain = {}
            for cid, votes, abs_gain in results:
                pbft_ok = pbft_decide(votes)
                if ablation_mode in ('no_consensus', 'quality_only', 'shapley_only'):
                    accepted[cid] = True
                else:
                    accepted[cid] = pbft_ok
                model_gain[cid] = abs_gain
                ledger_all.append({
                    'round': r, 'method': adaptive_method, 'ablation': ablation_mode,
                    'client': cid, 'votes': votes, 'accepted': accepted[cid],
                    'pbft_accepted': pbft_ok, 'abs_gain': abs_gain,
                    'is_malicious': int(cid in malicious_clients),
                    'attack_type': args.attack_type,
                    'malicious_ratio': float(args.malicious_ratio),
                    'dp_noise_multiplier': float(client_noise.get(cid, args.base_noise)),
                })

            # (4) Fallback selection. For adversarial experiments, this fallback is disabled by default
            # so that malicious rejection rates are not artificially weakened by min_selected.
            selected = [cid for cid, ok in accepted.items() if ok]
            disable_fallback = (args.attack_type != 'none' and args.disable_fallback_under_attack)
            need = max(args.min_selected - len(selected), 0)
            if need > 0 and not disable_fallback and ablation_mode not in ('no_consensus', 'quality_only', 'shapley_only'):
                not_acc = [cid for cid in client_updates.keys() if cid not in selected]
                comp_scores = {
                    cid: (args.beta  * float(relevance.get(cid, 0.0)) +
                          args.gamma_model_gain * float(model_gain.get(cid, 0.0)) +
                          args.lambda_trust     * float(trust_scores[adaptive_method].get(cid, 0.0)))
                    for cid in not_acc
                }
                filtered = [cid for cid in not_acc if model_gain.get(cid, 0.0) >= args.gain_min_threshold]
                ranked = sorted(filtered, key=lambda c: comp_scores[c], reverse=True)
                for cid in ranked:
                    if need <= 0: break
                    selected.append(cid); accepted[cid] = True; need -= 1
                if need > 0:
                    residual = [cid for cid in not_acc if cid not in selected]
                    ranked2 = sorted(residual, key=lambda c: comp_scores[c], reverse=True)
                    for cid in ranked2:
                        if need <= 0: break
                        selected.append(cid); accepted[cid] = True; need -= 1

            # (5) kNN-Shapley contribution proxy.
            shapley = knn_shapley_clients(
                client_updates, validators[0].dataset, model_fn,
                k=args.knn_k, n_jobs=args.n_jobs, sample=args.knn_sample,
                device=str(DEVICE), is_binary=is_binary
            )

            # (6) Variance penalty.
            deltas = {}
            norms = []
            for cid, st in client_updates.items():
                delta_state = state_delta(st, prev_state)
                flat = l2_flat(delta_state)
                deltas[cid] = flat
                norms.append(float(torch.norm(flat) + 1e-12))
            norms = np.array(norms, dtype=np.float64)
            if len(norms) > 0:
                med = float(np.median(norms))
                if med > 0:
                    var_penalty = {cid: np.exp(-args.var_reg_eta * max(0.0, (float(torch.norm(deltas[cid])) - med)/med) * 1.5)
                                   for cid in client_updates.keys()}
                else:
                    var_penalty = {cid: 1.0 for cid in client_updates.keys()}
            else:
                var_penalty = {cid: 1.0 for cid in client_updates.keys()}

            # (7) Aggregation according to the selected ablation mode.
            tau_r = temperature_schedule(r, args.rounds, args.temp_start, args.temp_mid, args.temp_end)
            if args.stability_mode == "S2":
                tau_r *= 0.8

            if ablation_mode == 'consensus_only':
                # PBFT filtering only, then uniform aggregation over accepted clients.
                accepted_cids = [cid for cid, ok in accepted.items() if ok] or list(client_updates.keys())
                new_state_adaptive = fedavg_aggregate({cid: client_updates[cid] for cid in accepted_cids})
                cid_order = accepted_cids
                weights_adaptive = [1.0/len(cid_order)] * len(cid_order)
                raw_logits = np.ones(len(cid_order), dtype=np.float64)
            else:
                if ablation_mode == 'quality_only':
                    alpha_eff, beta_eff = 0.0, 1.0
                    mg_scaled = {cid: 0.0 for cid in client_updates.keys()}
                    trust_scaled = {cid: 0.0 for cid in client_updates.keys()}
                elif ablation_mode in ('shapley_only', 'no_quality'):
                    alpha_eff, beta_eff = 1.0, 0.0
                    mg_scaled = {cid: 0.0 for cid in client_updates.keys()}
                    trust_scaled = {cid: 0.0 for cid in client_updates.keys()}
                else:
                    alpha_eff, beta_eff = args.alpha, args.beta
                    mg_scaled = {cid: args.gamma_model_gain * float(model_gain.get(cid, 0.0)) for cid in client_updates.keys()}
                    trust_scaled = {cid: args.lambda_trust * float(trust_scores[adaptive_method].get(cid, 0.0)) for cid in client_updates.keys()}

                new_state_adaptive, weights_adaptive, cid_order, raw_logits = intent_aware_aggregate(
                    client_updates, shapley, relevance, accepted,
                    alpha=alpha_eff, beta=beta_eff, temperature=tau_r,
                    model_gain=mg_scaled, trust_scores=trust_scaled, max_selected=args.max_selected
                )

                # Apply variance penalty + weight floor + renormalization for weighted variants.
                v = np.array([var_penalty.get(cid, 1.0) for cid in cid_order], dtype=np.float64)
                if len(weights_adaptive) > 0 and ablation_mode not in ('quality_only', 'shapley_only'):
                    w = np.array(weights_adaptive, dtype=np.float64) * v
                    floor = min(args.weight_floor, 0.25/len(w))
                    w = np.maximum(w, floor)
                    w = w / np.clip(w.sum(), 1e-12, None)
                    weights_adaptive = w.tolist()
                    out = copy.deepcopy(next(iter(client_updates.values())))
                    for k in out: out[k] = out[k].zero_()
                    for wi, cid in zip(weights_adaptive, cid_order):
                        for k in out: out[k] += float(wi) * client_updates[cid][k]
                    new_state_adaptive = out

            # (8) Anti-drift, server proximal pull, delta clipping and EMA.
            if args.stability_mode == "S2":
                for k in new_state_adaptive.keys():
                    drift = new_state_adaptive[k] - prev_state[k]
                    new_state_adaptive[k] -= args.anti_drift_weight * drift

            if args.server_prox_mu > 0.0:
                for k in new_state_adaptive.keys():
                    new_state_adaptive[k] = (1.0 - args.server_prox_mu) * new_state_adaptive[k] + args.server_prox_mu * prev_state[k]

            delta_state = {k: (new_state_adaptive[k] - prev_state[k]) for k in prev_state.keys()}
            delta_state = clip_update(delta_state, args.server_clip_norm)
            clipped = {k: prev_state[k] + delta_state[k] for k in prev_state.keys()}

            ema = args.server_momentum
            if r < 5:
                ema = 0.01
            if args.final_stabilize and (r-10 > args.rounds // 2):
                denom = max(1, args.rounds // 2)
                ema = args.server_momentum + ((r-10-args.rounds // 2)/denom)*(args.server_momentum-args.server_momentum_end)
                ema = min(ema, args.server_momentum_end)
            merged = copy.deepcopy(prev_state)
            for k in merged.keys():
                merged[k] = ema * merged[k] + (1.0 - ema) * clipped[k]
            server_states[adaptive_method] = merged

            # (9) Evaluate and attach attack/fairness diagnostics.
            hist_metrics = evaluate_model_state(server_states[adaptive_method], model_fn, testset, device=DEVICE, is_binary=is_binary)
            bad = set(malicious_clients)
            honest = set(client_updates.keys()) - bad
            accepted_bad = [cid for cid in bad if accepted.get(cid, False)]
            rejected_bad = [cid for cid in bad if not accepted.get(cid, False)]
            accepted_honest = [cid for cid in honest if accepted.get(cid, False)]
            reward_by_client = {cid: float(w) for cid, w in zip(cid_order, weights_adaptive)}
            contrib_norm = normalize_nonnegative(shapley, cids=sorted(client_updates.keys()))
            fairness = reward_fairness_metrics(contrib_norm, reward_by_client)
            malicious_reward_share = float(sum(reward_by_client.get(cid, 0.0) for cid in bad))
            hist_metrics.update({
                'attack_type': args.attack_type,
                'malicious_ratio': float(args.malicious_ratio),
                'malicious_clients': len(bad),
                'accepted_malicious_rate': float(len(accepted_bad) / max(1, len(bad))),
                'rejected_malicious_rate': float(len(rejected_bad) / max(1, len(bad))),
                'accepted_honest_rate': float(len(accepted_honest) / max(1, len(honest))),
                'malicious_reward_share': malicious_reward_share,
                'honest_reward_share': float(1.0 - malicious_reward_share),
                'ablation': ablation_mode,
                **fairness,
            })
            histories[adaptive_method].append(hist_metrics)

            # (10) Trust update.
            rho = np.clip(args.trust_ema, 0.0, 1.0)
            for cid in client_updates.keys():
                g = float(model_gain.get(cid, 0.0))
                g_norm = np.tanh(10.0 * g)
                prior = float(trust_scores[adaptive_method].get(cid, 0.0))
                new_t = (1.0 - rho) * prior + rho * max(0.0, g_norm)
                new_t = (1.0 - args.trust_decay) * new_t
                trust_scores[adaptive_method][cid] = float(max(args.trust_floor, min(1.0, new_t)))

            # (11) Incentives/rewards ledger for contribution and privacy-fairness analysis.
            if args.incentives == "all" and len(cid_order) > 0:
                shap_vals = np.array([max(0.0, shapley.get(cid, 0.0)) for cid in cid_order], dtype=np.float64)
                shap_sum = shap_vals.sum()
                rew_shap = shap_vals / shap_sum if shap_sum > 0 else np.ones_like(shap_vals) / len(shap_vals)
                rew_equal = np.ones_like(shap_vals) / len(shap_vals)
                eps_t = 1e-6
                times = np.array([latency_this_round.get(cid, 1.0) for cid in cid_order], dtype=np.float64)
                inv = 1.0 / np.clip(times, eps_t, None)
                inv_sum = inv.sum()
                rew_lat = inv / inv_sum if inv_sum > 0 else np.ones_like(inv) / len(inv)
                rew_prop = np.array(weights_adaptive, dtype=np.float64)

                for j, cid in enumerate(cid_order):
                    incentives_all_rounds.append({
                        'round': r,
                        'method': adaptive_method,
                        'ablation': ablation_mode,
                        'client': cid,
                        'accepted': int(accepted.get(cid, False)),
                        'is_malicious': int(cid in malicious_clients),
                        'attack_type': args.attack_type,
                        'malicious_ratio': float(args.malicious_ratio),
                        'dp_noise_multiplier': float(client_noise.get(cid, args.base_noise)),
                        'contribution_score': float(contrib_norm.get(cid, 0.0)),
                        'raw_shapley': float(shapley.get(cid, 0.0)),
                        'relevance': float(relevance.get(cid, 0.0)),
                        'model_gain': float(model_gain.get(cid, 0.0)),
                        'reward_shapley': float(rew_shap[j]),
                        'reward_equal': float(rew_equal[j]),
                        'reward_latency': float(rew_lat[j]),
                        'reward_proposed': float(rew_prop[j]),
                        'compute_time_sec': float(compute_time_this_round.get(cid, 0.0)),
                        'comm_latency_sec': float(comm_lat_this_round.get(cid, 0.0)),
                        'total_latency_sec': float(latency_this_round.get(cid, 0.0))
                    })

            print(f"[{adaptive_method} | Round {r}] acc={hist_metrics['acc']:.4f} base={base_val:.4f} "
                  f"selected={len([c for c in accepted if accepted[c]])} bad_reward={malicious_reward_share:.3f} "
                  f"tau={tau_r:.2f} ema={ema:.2f}")

            # (12) Advance rolling baseline after evaluation.
            server_hist_for_pbft[adaptive_method].append(copy.deepcopy(server_states[adaptive_method]))


        #     log line
        parts=[]
        if args.incentives == "all":
            parts.append(f"Incentives ✓")
        for m in run_methods:
            if m in histories and len(histories[m]) > 0:
                parts.append(f"{m} acc={histories[m][-1]['acc']:.3f}")
        print(f"[Round {r}] " + " | ".join(parts))

    return histories, ledger_all.to_df(), (eps, delta), pd.DataFrame(incentives_all_rounds)


# -------------------------------
# Trials, plotting, CSVs
# -------------------------------
def plot_overlay(mean_histories:Dict[str,Dict[str,List[float]]], rounds:int, dataset:str, out_dir:str):
    metrics=['acc','precision','recall','f1','loss']; rr=range(1, rounds+1)
    fig, axes = plt.subplots(3,2, figsize=(14,10))
    for i,metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        for method, hist in mean_histories.items():
            if metric in hist:
                ax.plot(rr, hist[metric], marker='o', label=method.upper())
        ax.set_title(metric.upper()); ax.set_xlabel('Round'); ax.set_ylabel(metric.upper())
        ax.grid(True); ax.legend()
    plt.suptitle(f"FL Methods Comparison on {dataset}", fontsize=16)
    os.makedirs(out_dir, exist_ok=True)
    path=os.path.join(out_dir, f"comparison_{dataset}.png")
    plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(path); plt.close()
    print(f"Saved overlay figure: {path}")


def gini_coefficient(x: np.ndarray) -> float:
    """Gini on nonnegative vector x (0 = perfect equality, 1 = max inequality)."""
    x = np.sort(np.clip(np.asarray(x, dtype=float), 0, None))
    if x.sum() == 0: return 0.0
    n = x.size
    cumx = np.cumsum(x)
    # relative mean absolute difference formula
    return (n + 1 - 2 * (cumx / cumx[-1]).sum()) / n

def plot_incentive_bars_and_lorenz(incentive_df: pd.DataFrame, out_dir: str, title_suffix: str):
    """
    Expects columns: ['round','client','reward_shapley','reward_equal','reward_latency','reward_proposed'].
    Plots: (1) bar charts of mean reward per client for each scheme
           (2) Lorenz curves overlay + Gini table.
    """
    os.makedirs(out_dir, exist_ok=True)
    if incentive_df.empty:
        print("Incentive DF is empty; skipping incentive plots.")
        return

    # Average rewards per client across rounds (and trials, since we concat across trials later)
    by_client = incentive_df.groupby('client')[['reward_shapley','reward_equal','reward_latency','reward_proposed']].mean()

    # ---- Bar plots (4-up) ----
    # --- Define schemes (add contribution = proposed) ---
    schemes = ['reward_shapley', 'reward_equal', 'reward_latency', 'reward_proposed', 'reward_contrib']
    titles  = ['Shapley-only', 'Equal', 'Latency-based', 'Proposed (Adaptive)', 'Contribution']

    # Copy proposed column into contribution
    by_client['reward_contrib'] = by_client['reward_proposed']

    x = np.arange(len(by_client))  # client indices
    width = 0.15  # bar width (smaller since we now have 5 bars)

    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot side-by-side bars
    for i, (col, t) in enumerate(zip(schemes, titles)):
        ax.bar(x + i*width, by_client[col].values, width=width, label=t)

    # Format axes
    ax.set_xticks(x + (len(schemes)-1)*width/2)
    ax.set_xticklabels(by_client.index.astype(str))
    ax.set_xlabel("Client")
    ax.set_ylabel("Contribution-Reward Balance")
    ax.set_title(f"Comparison of Incentive Distributions per Client {title_suffix}")
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    bar_path = os.path.join(out_dir, f"incentive_bars_comparison{title_suffix}.png")
    plt.savefig(bar_path)
    plt.close()
    print(f"Saved combined incentive bar plot: {bar_path}")


    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # schemes = ['reward_shapley','reward_equal','reward_latency','reward_proposed']
    # titles = ['Shapley-only','Equal','Latency-based','Proposed (Adaptive)']
    # for ax, col, t in zip(axes.flatten(), schemes, titles):
    #     ax.bar(by_client.index.astype(str), by_client[col].values)
    #     ax.set_title(t); ax.set_xlabel("Client"); ax.set_ylabel("Mean reward"); ax.grid(True, axis='y', alpha=0.3)
    # plt.suptitle(f"Incentive Distributions per Client {title_suffix}", fontsize=16)
    # plt.tight_layout(rect=[0,0,1,0.96])
    # bar_path = os.path.join(out_dir, f"incentive_bars{title_suffix}.png")
    # plt.savefig(bar_path); plt.close()
    # print(f"Saved incentive bar plots: {bar_path}")

    schemes = ['reward_shapley', 'reward_equal', 'reward_latency', 'reward_proposed']
    titles  = ['Shapley-only', 'Equal', 'Latency-based', 'Proposed (Adaptive)']


    # ---- Lorenz + Gini ----
    plt.figure(figsize=(8,6))
    ginis = {}
    for col, label in zip(schemes, titles):
        v = by_client[col].values/by_client['reward_contrib'].values
        v = np.clip(v, 0, None)
        v_sorted = np.sort(v)
        cum = np.cumsum(v_sorted)
        lorenz = np.insert(cum / (cum[-1] if cum[-1] > 0 else 1.0), 0, 0)
        x = np.linspace(0.0, 1.0, lorenz.size)
        plt.plot(x, lorenz, label=f"{label}")
        ginis[label] = gini_coefficient(v)
    plt.plot([0,1],[0,1],'k--',alpha=0.5,label='Equality line')
    plt.xlabel("Cumulative share of clients"); plt.ylabel("Cumulative share of rewards")
    plt.title(f"Lorenz Curves (Fairness) {title_suffix}")
    plt.grid(True, alpha=0.3); plt.legend()
    lorenz_path = os.path.join(out_dir, f"incentive_lorenz{title_suffix}.png")
    plt.tight_layout(); plt.savefig(lorenz_path); plt.close()
    print(f"Saved Lorenz curves: {lorenz_path}")

    # Save Gini as CSV
    gini_rows = [{'scheme': k, 'gini': float(v)} for k,v in ginis.items()]
    gini_df = pd.DataFrame(gini_rows)
    gini_df.to_csv(os.path.join(out_dir, f"incentive_gini{title_suffix}.csv"), index=False)


def summarize_fairness_and_dp(incentive_df: pd.DataFrame, out_dir: str):
    """Save reward-contribution and heterogeneous-DP fairness summaries."""
    if incentive_df is None or incentive_df.empty:
        return
    rows = []
    group_cols = [c for c in ['trial','method','ablation','attack_type','malicious_ratio','round'] if c in incentive_df.columns]
    for keys, g in incentive_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        contrib = dict(zip(g['client'].astype(int), g['contribution_score'].astype(float)))
        rewards = dict(zip(g['client'].astype(int), g['reward_proposed'].astype(float)))
        row.update(reward_fairness_metrics(contrib, rewards))
        if 'is_malicious' in g.columns:
            row['malicious_reward_share'] = float(g.loc[g['is_malicious'].astype(int)==1, 'reward_proposed'].sum())
        if 'dp_noise_multiplier' in g.columns:
            noise_group = dict(zip(g['client'].astype(int), g['dp_noise_multiplier'].astype(float)))
            row['dp_reward_gap'] = group_reward_gap(rewards, noise_group)
            row['dp_contribution_gap'] = group_reward_gap(contrib, noise_group)
            row['dp_reward_noise_corr'] = safe_corr(g['dp_noise_multiplier'].astype(float), g['reward_proposed'].astype(float), 'spearman')
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'fairness_dp_summary.csv'), index=False)


def run_trials(args):
    # os.makedirs(args.out_dir, exist_ok=True)
    if args.run_all:
        methods_all = ['adaptive','fedavg','fedprox','fedsgd']
        if args.run_robust_baselines:
            methods_all += list(ROBUST_METHODS)
    else:
        methods_all = args.method if isinstance(args.method, list) else [args.method]
    if args.run_ablation:
        for m in ['adaptive_no_consensus','adaptive_consensus_only','adaptive_quality_only','adaptive_shapley_only','adaptive_no_quality']:
            if m not in methods_all:
                methods_all.append(m)
    print(f"Running methods: {methods_all}")
    all_hist=[]
    all_ledgers=[]
    rdp_list=[]
    all_incentives=[]  # NEW
    for t in range(args.trials):
        seed = args.seed + t*11
        print(f"== Trial {t+1}/{args.trials} (seed={seed}) ==")
        hist, ledger_df, (eps,delta), incent_df = run_single_trial(args, seed, methods_all)
        all_hist.append(hist); all_ledgers.append(ledger_df); rdp_list.append((eps,delta))
        if incent_df is not None and not incent_df.empty:
            incent_df['trial'] = t+1
            all_incentives.append(incent_df)

        # Save incentives ledger (concat across trials) + plots
        if len(all_incentives) > 0:
            inc_df = pd.concat(all_incentives, ignore_index=True)
            inc_path = os.path.join(args.out_dir, 'incentive_ledger.csv')
            inc_df.to_csv(inc_path, index=False)
            print(f"Saved incentive ledger: {inc_path}")
            # Plots and fairness tables (aggregate across trials)
            plot_incentive_bars_and_lorenz(inc_df, args.out_dir, title_suffix=f"_{args.dataset}")
            summarize_fairness_and_dp(inc_df, args.out_dir)
        else:
            print("No incentive ledger content (did you run with --incentives all?).")

    # Average histories across trials
    rounds=args.rounds
    methods=set().union(*[set(h.keys()) for h in all_hist])
    mean_hist = {m:{'acc':[], 'precision':[], 'recall':[], 'f1':[],'loss':[]} for m in methods}
    for m in methods:
        for r in range(rounds):
            vals = {k: [] for k in ['acc','precision','recall','f1','loss']}
            for t in range(len(all_hist)):
                if r < len(all_hist[t][m]):
                    for k in vals.keys():
                        vals[k].append(all_hist[t][m][r][k])
            for k in vals.keys():
                mean_hist[m][k].append(float(np.mean(vals[k])) if len(vals[k])>0 else 0.0)

    # Save per-trial detailed CSV
    rows=[]
    for t,hist in enumerate(all_hist):
        for m in hist.keys():
            for r in range(len(hist[m])):
                row={'trial':t+1, 'round':r+1, 'method':m}
                row.update(hist[m][r])
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir,'all_trials_metrics.csv'), index=False)

    # Save ledgers
    for i,ldf in enumerate(all_ledgers):
        ldf.to_csv(os.path.join(args.out_dir, f'ledger_trial_{i+1}.csv'), index=False)

    # Save RDP
    pd.DataFrame(rdp_list, columns=['epsilon','delta']).to_csv(os.path.join(args.out_dir,'rdp_accounting.csv'), index=False)

    # Plot overlay
    plot_overlay(mean_hist, rounds, dataset=args.dataset, out_dir=args.out_dir)
    return df

# -------------------------------
# CLI
# -------------------------------
def parse_args():
    p=argparse.ArgumentParser()
    # Dataset choice
    p.add_argument("--dataset", type=str, choices=["mnist","cifar10","pathmnist","tissuemnist","organamnist","organsmnist"], default="organsmnist")
    # p.add_argument("--dataset", type=str, choices=["lung","heart","diabetes","mnist","cifar10"], default="lung")
    p.add_argument("--binary", action="store_true", help="Binary version of MNIST/CIFAR-10 (MNIST: 0 vs 1; CIFAR-10: cats vs dogs)")
    

    # FL config
    method_choices = ["adaptive","fedavg","fedprox","fedsgd"] + list(ROBUST_METHODS) + list(ADAPTIVE_METHODS.keys())
    method_choices = sorted(set(method_choices))
    p.add_argument("--method", type=str, nargs="+", choices=method_choices, default=["adaptive"],
                   help="One or more methods. Robust baselines and adaptive ablation variants are supported.")
    p.add_argument("--run_all", action="store_true", help="Run primary methods in one call", default=False)
    p.add_argument("--run_robust_baselines", action="store_true", default=False,
                   help="When --run_all is used, also include Krum, Multi-Krum, median, trimmed mean and Bulyan.")
    p.add_argument("--run_ablation", action="store_true", default=False,
                   help="Also run adaptive ablation variants: no_consensus, consensus_only, quality_only, shapley_only, no_quality.")
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--validators", type=int, default=5)
    p.add_argument("--validator_val_size", type=int, default=2000)
    p.add_argument("--local_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--dirichlet_alpha", type=float, default=0.01)
    p.add_argument("--fedprox_mu", type=float, default=0.001)

    # Adaptive weighting
    p.add_argument("--alpha", type=float, default=0.1)  # Shapley weight
    p.add_argument("--beta", type=float, default=0.9)   # Relevance weight
    p.add_argument("--agg_temperature", type=float, default=20.0)

    # kNN-Shapley
    p.add_argument("--knn_k", type=int, default=10)
    p.add_argument("--knn_sample", type=int, default=2000)

    # PBFT
    p.add_argument("--pbft_byzantine_rate", type=float, default=0.0000,
                   help="Probability that a validator casts a random vote. This models Byzantine validators, not malicious clients.")
    p.add_argument("--pbft_acceptance_delta", type=float, default=0.0000,
                   help="Minimum validation metric gain required for a positive PBFT vote.")

    # Genuine Byzantine FL client attacks
    p.add_argument("--attack_type", type=str, choices=list(ATTACK_TYPES), default="none")
    p.add_argument("--malicious_ratio", type=float, default=0.0,
                   help="Fraction of clients that intentionally submit malicious updates.")
    p.add_argument("--attack_scale", type=float, default=5.0,
                   help="Scale used by sign-flip, scaling and random-update attacks.")
    p.add_argument("--attack_noise_std", type=float, default=5.0,
                   help="Noise std multiplier for Gaussian model-poisoning attacks.")
    p.add_argument("--label_flip_shift", type=int, default=1,
                   help="Class shift used by label-flipping malicious clients.")
    p.add_argument("--allow_fallback_under_attack", dest="disable_fallback_under_attack", action="store_false", default=True,
                   help="Allow min_selected fallback even when attack_type != none. By default it is disabled so rejection-rate metrics remain meaningful.")

    # Robust aggregation baselines
    p.add_argument("--robust_f", type=int, default=-1,
                   help="Assumed number of Byzantine clients for Krum/Multi-Krum/Bulyan. -1 derives it from malicious_ratio.")
    p.add_argument("--trim_ratio", type=float, default=0.2,
                   help="Trim ratio for coordinate-wise trimmed mean and approximate Bulyan.")
    p.add_argument("--multikrum_m", type=int, default=0,
                   help="Number of updates selected by Multi-Krum. 0 uses n-f-2.")

    # DP
    p.add_argument("--clip_norm", type=float, default=2.0)
    p.add_argument("--base_noise", type=float, default=0.0005)
    p.add_argument("--heterogeneous_dp", action="store_true", default=False,
                   help="Use client-specific DP noise multipliers to simulate heterogeneous hospital privacy requirements.")
    p.add_argument("--dp_noise_multipliers", type=str, default="",
                   help="Comma-separated per-group DP noise multipliers, e.g., '0.0005,0.005,0.02'.")
    
    # Incentives / latency modeling
    p.add_argument("--incentives", type=str, default="all",
                choices=["none","all"],
                help="If 'all', compute & log Shapley-only, Equal, Latency-based, and Proposed rewards.")
    p.add_argument("--latency_mode", type=str, default="simulate",
                choices=["simulate","off"],
                help="How to obtain communication latency. 'simulate' uses model size / bandwidth + jitter.")
    p.add_argument("--bandwidth_mbps", type=float, default=10.0,
                help="Assumed uplink bandwidth in Mbps for simulate latency.")
    p.add_argument("--latency_jitter_ms", type=float, default=20.0,
                help="Added lognormal jitter (ms) around simulated latency.")
    
     # Parallelization
    p.add_argument("--n_jobs", type=int, default=1,
                   help="Number of CPU workers for parallel validator evaluation (-1 = all cores).")

   
    # ---- Stability Mode / Schedules (S2 defaults) -------------------------------
    p.add_argument("--stability_mode", type=str, default="S2", choices=["S1","S2","S3","S4"])
    p.add_argument("--final_stabilize", action="store_true", default=True,
                help="Enable stronger stabilization after half of the rounds.")

    # Confidence annealing (S2: sharper focus sooner)
    p.add_argument("--temp_start", type=float, default=8.0)
    p.add_argument("--temp_mid",   type=float, default=3.5)
    p.add_argument("--temp_end",   type=float, default=1.5)

    # Variance regularizer on client updates (S2: moderate)
    p.add_argument("--var_reg_eta", type=float, default=0.35)

    # Trust memory (S2: faster decay, lower floor)
    p.add_argument("--lambda_trust", type=float, default=0.6)  # already present earlier; keep
    p.add_argument("--trust_ema",    type=float, default=0.4)
    p.add_argument("--trust_decay",  type=float, default=0.08)
    p.add_argument("--trust_floor",  type=float, default=0.02)

    # Model-gain weight
    p.add_argument("--gamma_model_gain", type=float, default=1.2)

    # Server stabilization (S2: stronger)
    p.add_argument("--server_clip_norm",   type=float, default=1.8)
    p.add_argument("--server_prox_mu",     type=float, default=0.05)
    p.add_argument("--server_momentum",    type=float, default=0.05)
    p.add_argument("--server_momentum_end",type=float, default=0.90)

    # Anti-drift (S2 only)
    p.add_argument("--anti_drift_weight", type=float, default=0.15)

    # Selection / fallback / floors
    p.add_argument("--min_selected",   type=int,   default=8)
    p.add_argument("--max_selected",   type=int,   default=0)     # 0 = no cap
    p.add_argument("--gain_min_threshold", type=float, default=0.001)
    p.add_argument("--weight_floor",   type=float, default=0.008)
    p.add_argument("--rolling_base_k", type=int,   default=4)



    # Misc
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--out_dir", type=str, default="./results1/organsmnist/trial3c")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# The main7 model started in /results1/pathmnist/trial9d
# started a new stabilize S1 model at /results1/pathmnist/trial9g
# started a new stabilize S2 model at /results1/pathmnist/trial9j- momemtum=j-0.65->k-0.6->l-0.5

# -------------------------------
# Entry
# -------------------------------
if __name__=="__main__":
    args=parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.DataFrame([vars(args)])
    df.to_csv(os.path.join(args.out_dir,'config_parameters.csv'), index=False)
    set_seed(args.seed)
    t0=time.time()
    _ = run_trials(args)
    print(f"Done in {time.time()-t0:.1f}s. Outputs in {args.out_dir}")
