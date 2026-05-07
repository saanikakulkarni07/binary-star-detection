"""Model definitions: Random Forest baseline and 1D CNN for binary detection."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from config import MODEL_CONFIG


# ---------------------------------------------------------------------------
# Random Forest baseline
# ---------------------------------------------------------------------------

def build_rf_pipeline():
    """Build a scikit-learn pipeline: StandardScaler -> RandomForestClassifier."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=MODEL_CONFIG["rf_n_estimators"],
            max_depth=MODEL_CONFIG["rf_max_depth"],
            class_weight=MODEL_CONFIG["rf_class_weight"],
            random_state=MODEL_CONFIG["random_state"],
            n_jobs=-1,
        )),
    ])


# ---------------------------------------------------------------------------
# 1D CNN
# ---------------------------------------------------------------------------

class BinaryStarCNN(nn.Module):
    """1D CNN for binary star detection from single-epoch spectra.

    Architecture:
        - 3 convolutional blocks with BatchNorm and MaxPool
        - Large first kernel (16) to capture absorption line profiles
        - Stride-2 in first layer to reduce oversampled APOGEE pixels
        - AdaptiveAvgPool for fixed output size
        - FC classifier with dropout
    """

    def __init__(self, n_pixels=8575, dropout=None):
        super().__init__()
        dropout = dropout or MODEL_CONFIG["cnn_dropout"]

        # Block 1: broad spectral features
        self.conv1 = nn.Conv1d(1, 32, kernel_size=16, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)

        # Block 2: intermediate features
        self.conv2 = nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)

        # Block 3: fine spectral features (line shapes)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=4, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.AdaptiveAvgPool1d(32)

        # Classifier head
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * 32, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        """Forward pass. Input shape: (batch, 1, n_pixels)."""
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ---------------------------------------------------------------------------
# Dataset and DataLoader utilities
# ---------------------------------------------------------------------------

class SpectraDataset(torch.utils.data.Dataset):
    """PyTorch dataset for APOGEE spectra."""

    def __init__(self, spectra, labels, errors=None, augment=False):
        """
        Parameters
        ----------
        spectra : np.ndarray (n_samples, n_pixels)
        labels : np.ndarray (n_samples,)
        errors : np.ndarray (n_samples, n_pixels), optional
            Per-pixel flux errors for noise augmentation.
        augment : bool
            If True, add Gaussian noise at the per-pixel error level.
        """
        self.spectra = torch.FloatTensor(spectra)
        self.labels = torch.FloatTensor(labels)
        self.errors = torch.FloatTensor(errors) if errors is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.spectra[idx].unsqueeze(0)  # (1, n_pixels)
        y = self.labels[idx]

        if self.augment and self.errors is not None:
            noise = torch.randn_like(x) * self.errors[idx].unsqueeze(0)
            x = x + noise

        return x, y


def create_data_loaders(spectra, labels, errors=None):
    """Create stratified train/val/test DataLoaders.

    Returns
    -------
    tuple of (train_loader, val_loader, test_loader)
    """
    from sklearn.model_selection import train_test_split

    rs = MODEL_CONFIG["random_state"]
    test_size = MODEL_CONFIG["test_size"]
    val_size = MODEL_CONFIG["val_size"]

    # First split: train+val vs test
    idx = np.arange(len(labels))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, stratify=labels, random_state=rs
    )
    # Second split: train vs val
    relative_val = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=relative_val, stratify=labels[idx_trainval], random_state=rs
    )

    err_train = errors[idx_train] if errors is not None else None
    err_val = errors[idx_val] if errors is not None else None
    err_test = errors[idx_test] if errors is not None else None

    train_ds = SpectraDataset(spectra[idx_train], labels[idx_train], err_train, augment=True)
    val_ds = SpectraDataset(spectra[idx_val], labels[idx_val], err_val, augment=False)
    test_ds = SpectraDataset(spectra[idx_test], labels[idx_test], err_test, augment=False)

    bs = MODEL_CONFIG["cnn_batch_size"]
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_cnn(model, train_loader, val_loader, device="cpu"):
    """Train the CNN with early stopping on validation AUROC.

    Returns
    -------
    dict with keys: 'train_loss', 'val_loss', 'val_auroc', 'best_epoch'
    """
    from sklearn.metrics import roc_auc_score

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MODEL_CONFIG["cnn_learning_rate"],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    # Compute pos_weight from training data
    all_labels = []
    for _, y in train_loader:
        all_labels.append(y)
    all_labels = torch.cat(all_labels)
    n_neg = (all_labels == 0).sum().float()
    n_pos = (all_labels == 1).sum().float()
    pos_weight = (n_neg / n_pos).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = {"train_loss": [], "val_loss": [], "val_auroc": []}
    best_auroc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(MODEL_CONFIG["cnn_epochs"]):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch).squeeze(-1)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses, val_preds, val_true = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch).squeeze(-1)
                loss = criterion(logits, y_batch)
                val_losses.append(loss.item())
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_true.append(y_batch.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)
        val_auroc = roc_auc_score(val_true, val_preds)

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_auroc"].append(val_auroc)

        scheduler.step(avg_val)

        print(f"Epoch {epoch+1}/{MODEL_CONFIG['cnn_epochs']} — "
              f"train_loss: {avg_train:.4f}, val_loss: {avg_val:.4f}, val_auroc: {val_auroc:.4f}")

        # Early stopping
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= MODEL_CONFIG["cnn_patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_epoch"] = np.argmax(history["val_auroc"]) + 1
    return history
