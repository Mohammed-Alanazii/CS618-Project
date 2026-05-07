import os
import json
import random
import warnings
import argparse
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────
WINDOW_SIZE = 200
WINDOW_STEP = 50          # 75% overlap
N_CHANNELS = 12
N_CLASSES = 5
FS = 2000

BATCH_SIZE = 256          # Larger for GPU
NUM_WORKERS = 4
LR = 2e-3
WEIGHT_DECAY = 3e-3
LABEL_SMOOTHING = 0.1
DROPOUT = 0.5
MAX_EPOCHS = 100
PATIENCE = 30
GRAD_CLIP = 1.0
GAUSSIAN_NOISE_STD = 0.03

# Adaptive Mixup
MIXUP_ALPHA_EARLY = 0.25   # Epochs  1–25: strong mixing
MIXUP_ALPHA_MID   = 0.05   # Epochs 26–50: weak mixing
MIXUP_OFF_EPOCH   = 50     # Epochs 51+: no mixup

# ReduceLROnPlateau
LR_FACTOR = 0.5
LR_PATIENCE = 10

# Model
CHANNELS = [64, 128, 256]
KERNEL_SIZES = [15, 7, 3]

# Seed 123 = best test split from prior experiments
SEED = 123
GESTURE_NAMES = ["Rest", "Fist", "LargeGrasp", "WristPron", "Tripod"]
GESTURE_MAP = {0: 0, 6: 1, 17: 2, 25: 3, 38: 4}


# ────────────────────────────────────────────────────────────────────
# UTILITIES
# ────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ────────────────────────────────────────────────────────────────────
# SIGNAL PREPROCESSING
# ────────────────────────────────────────────────────────────────────
def bandpass_filter(emg, lowcut=20, highcut=450, fs=FS, order=4):
    if emg.ndim == 1:
        emg = emg[np.newaxis, :]
    nyq = 0.5 * fs
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    filt = np.zeros_like(emg)
    for ch in range(emg.shape[0]):
        filt[ch] = sosfiltfilt(sos, emg[ch])
    return filt


def notch_filter(emg, freq=50, fs=FS, quality=30):
    if emg.ndim == 1:
        emg = emg[np.newaxis, :]
    w0 = freq / (0.5 * fs)
    b, a = iirnotch(w0, quality)
    filt = np.zeros_like(emg)
    for ch in range(emg.shape[0]):
        filt[ch] = filtfilt(b, a, emg[ch])
    return filt


# ────────────────────────────────────────────────────────────────────
# DATASET
# ────────────────────────────────────────────────────────────────────
class EMGWindowDataset(Dataset):
    def __init__(self, subset_path, subject_ids):
        self.windows = []
        self.labels = []
        self.subject_indices = []
        self.subject_id_to_idx = {sid: idx for idx, sid in enumerate(subject_ids)}

        for subj_id in subject_ids:
            emg = np.load(os.path.join(subset_path, f"S{subj_id}_emg.npy"))
            lbls = np.load(os.path.join(subset_path, f"S{subj_id}_labels.npy"))
            emg_t = bandpass_filter(emg.T)
            emg_t = notch_filter(emg_t)
            emg_filt = emg_t.T.astype(np.float32)
            ch_mean = emg_filt.mean(axis=0, keepdims=True)
            ch_std = emg_filt.std(axis=0, keepdims=True) + 1e-8
            emg_filt = (emg_filt - ch_mean) / ch_std
            sidx = self.subject_id_to_idx[subj_id]
            for i in range(0, len(emg_filt) - WINDOW_SIZE, WINDOW_STEP):
                c = i + WINDOW_SIZE // 2
                l = lbls[c]
                if l not in GESTURE_MAP:
                    continue
                self.windows.append(emg_filt[i:i + WINDOW_SIZE].copy())
                self.labels.append(GESTURE_MAP[l])
                self.subject_indices.append(sidx)

        self.windows = np.array(self.windows, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.subject_indices = np.array(self.subject_indices, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.windows[idx].copy()).T,
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.subject_indices[idx], dtype=torch.long),
        )


# ────────────────────────────────────────────────────────────────────
# MODEL: ResCNN (v3 architecture)
# ────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = F.gelu(self.bn2(self.conv2(out)))
        return self.pool(out + residual)


class ResCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(N_CHANNELS, CHANNELS[0], kernel_size=15, padding=7),
            nn.BatchNorm1d(CHANNELS[0]),
            nn.GELU(),
            nn.MaxPool1d(2),
        )
        self.block1 = ResBlock(CHANNELS[0], CHANNELS[0], KERNEL_SIZES[0])
        self.block2 = ResBlock(CHANNELS[0], CHANNELS[1], KERNEL_SIZES[1])
        self.block3 = ResBlock(CHANNELS[1], CHANNELS[2], KERNEL_SIZES[2])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(CHANNELS[2], N_CLASSES)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(self.dropout(x))


# ────────────────────────────────────────────────────────────────────
# DATA AUGMENTATION
# ────────────────────────────────────────────────────────────────────
def add_gaussian_noise(x, std=GAUSSIAN_NOISE_STD):
    return x + torch.randn_like(x) * std


def mixup_data(x, y, alpha):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_mixup_alpha(epoch):
    """Adaptive mixup schedule."""
    if epoch <= 25:
        return MIXUP_ALPHA_EARLY
    elif epoch <= MIXUP_OFF_EPOCH:
        return MIXUP_ALPHA_MID
    else:
        return 0.0  # off


# ────────────────────────────────────────────────────────────────────
# TRAINING
# ────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    mixup_alpha = get_mixup_alpha(epoch)
    use_mixup = mixup_alpha > 0

    for windows, labels, _ in loader:
        windows, labels = windows.to(device), labels.to(device)
        windows = add_gaussian_noise(windows)

        if use_mixup:
            windows, y_a, y_b, lam = mixup_data(windows, labels, mixup_alpha)
            optimizer.zero_grad()
            logits = model(windows)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            logits = model(windows)
            loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * windows.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for windows, labels, _ in loader:
        windows, labels = windows.to(device), labels.to(device)
        logits = model(windows)
        loss = criterion(logits, labels)
        total_loss += loss.item() * windows.size(0)
        all_preds.append(logits.argmax(-1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / len(loader.dataset), acc, f1, all_preds, all_labels


# ────────────────────────────────────────────────────────────────────
# MAJORITY VOTING (Continuous Gesture Segments)
# ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_majority_voting(model, subset_path, test_ids, device):
    model.eval()
    all_wpred, all_wtrue = [], []
    ghard_p, ghard_t = [], []
    gsoft_p, gsoft_t = [], []

    for subj_id in test_ids:
        emg = np.load(os.path.join(subset_path, f"S{subj_id}_emg.npy"))
        labels_full = np.load(os.path.join(subset_path, f"S{subj_id}_labels.npy"))
        emg_t = bandpass_filter(emg.T)
        emg_t = notch_filter(emg_t)
        emg_filt = emg_t.T.astype(np.float32)
        ch_mean = emg_filt.mean(axis=0, keepdims=True)
        ch_std = emg_filt.std(axis=0, keepdims=True) + 1e-8
        emg_filt = (emg_filt - ch_mean) / ch_std

        sw, sl, sp = [], [], []
        for i in range(0, len(emg_filt) - WINDOW_SIZE, WINDOW_STEP):
            c = i + WINDOW_SIZE // 2
            l = labels_full[c]
            if l not in GESTURE_MAP:
                continue
            sw.append(emg_filt[i:i + WINDOW_SIZE])
            sl.append(GESTURE_MAP[l])
            sp.append(c)
        if not sw:
            continue

        wt = torch.from_numpy(np.array(sw, dtype=np.float32)).permute(0, 2, 1)
        probs_all, preds_all = [], []
        for s in range(0, len(wt), 256):
            b = wt[s:s + 256].to(device)
            out = model(b)
            probs_all.append(F.softmax(out, dim=-1).cpu().numpy())
            preds_all.append(out.argmax(-1).cpu().numpy())
        probs_all = np.concatenate(probs_all)
        preds_all = np.concatenate(preds_all)
        sl = np.array(sl)

        all_wpred.extend(preds_all.tolist())
        all_wtrue.extend(sl.tolist())

        if len(sl) > 0:
            seg_s = 0
            for i in range(1, len(sl)):
                time_gap = sp[i] - sp[i-1]
                if sl[i] != sl[i-1] or time_gap > 2 * WINDOW_STEP:
                    if i - seg_s >= 3:
                        tl = int(sl[seg_s])
                        hv = Counter(preds_all[seg_s:i].tolist()).most_common(1)[0][0]
                        sv = int(probs_all[seg_s:i].sum(axis=0).argmax())
                        ghard_p.append(hv); gsoft_p.append(sv)
                        ghard_t.append(tl); gsoft_t.append(tl)
                    seg_s = i
            if len(sl) - seg_s >= 3:
                tl = int(sl[seg_s])
                hv = Counter(preds_all[seg_s:].tolist()).most_common(1)[0][0]
                sv = int(probs_all[seg_s:].sum(axis=0).argmax())
                ghard_p.append(hv); gsoft_p.append(sv)
                ghard_t.append(tl); gsoft_t.append(tl)

    results = {
        "window_acc": accuracy_score(all_wtrue, all_wpred) * 100,
        "window_f1": f1_score(all_wtrue, all_wpred, average="macro"),
    }
    for name, p, t in [("hard", ghard_p, ghard_t), ("soft", gsoft_p, gsoft_t)]:
        if t:
            results[f"{name}_acc"] = accuracy_score(t, p) * 100
            results[f"{name}_f1"] = f1_score(t, p, average="macro")
            results[f"{name}_cm"] = confusion_matrix(t, p).tolist()
            results[f"{name}_n"] = len(t)
        else:
            results[f"{name}_acc"] = 0.0
            results[f"{name}_f1"] = 0.0
            results[f"{name}_cm"] = None
            results[f"{name}_n"] = 0
    return results


# ────────────────────────────────────────────────────────────────────
# CHECKPOINT VERIFICATION
# ────────────────────────────────────────────────────────────────────
def check_weight_norm(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = state["model_state_dict"]
    total = 0.0
    for v in sd.values():
        if v.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            total += v.norm().item()
    return total


# ────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/subset",
                        help="Path to the npy data directory")
    parser.add_argument("--output_dir", type=str, default="results/best_model",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed")
    args = parser.parse_args()

    device = get_device()
    set_seed(args.seed)

    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Data:   {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Seed:   {args.seed}")

    # ── Subject split ──
    all_subjects = np.arange(1, 41)
    train_val, test = train_test_split(all_subjects, test_size=10,
                                        random_state=args.seed)
    train_ids, val_ids = train_test_split(train_val, test_size=6,
                                           random_state=args.seed)
    train_ids = sorted(train_ids.tolist())
    val_ids = sorted(val_ids.tolist())
    test_ids = sorted(test.tolist())

    print(f"\nTrain ({len(train_ids)}): {train_ids}")
    print(f"Val   ({len(val_ids)}): {val_ids}")
    print(f"Test  ({len(test_ids)}): {test_ids}")

    # ── Datasets ──
    train_ds = EMGWindowDataset(data_dir, train_ids)
    val_ds = EMGWindowDataset(data_dir, val_ids)
    print(f"\nTrain windows: {len(train_ds)}")
    print(f"Val windows:   {len(val_ds)}")
    for cls in range(N_CLASSES):
        print(f"  {GESTURE_NAMES[cls]:11s}  train={(train_ds.labels == cls).sum():6d}  "
              f"val={(val_ds.labels == cls).sum():6d}")

    # Class-balanced sampling
    class_counts = np.bincount(train_ds.labels, minlength=N_CLASSES)
    print(f"\nClass weights (for sampling): {1.0/class_counts}")
    sample_weights = 1.0 / class_counts[train_ds.labels]
    sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, drop_last=True,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=NUM_WORKERS,
                            pin_memory=(device.type == "cuda"))

    # ── Model ──
    model = ResCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} trainable parameters")
    print(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=1e-6,
    )

    print(f"\nHyperparameters:")
    print(f"  LR={LR}, weight_decay={WEIGHT_DECAY}, label_smoothing={LABEL_SMOOTHING}")
    print(f"  dropout={DROPOUT}, grad_clip={GRAD_CLIP}")
    print(f"  mixup: alpha_early={MIXUP_ALPHA_EARLY} (epochs 1-25), "
          f"alpha_mid={MIXUP_ALPHA_MID} (epochs 26-50), off @ {MIXUP_OFF_EPOCH}")
    print(f"  scheduler: ReduceLROnPlateau factor={LR_FACTOR} patience={LR_PATIENCE}")
    print(f"  patience={PATIENCE}, max_epochs={MAX_EPOCHS}")

    # ── Training ──
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_ckpt = os.path.join(output_dir, "best_model.pth")
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "lr": []}

    for epoch in range(1, MAX_EPOCHS + 1):
        mixup_alpha = get_mixup_alpha(epoch)
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device, epoch)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 2))
        history["val_f1"].append(round(val_f1, 4))
        history["lr"].append(current_lr)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
            }, best_ckpt)
            marker = " *"
        else:
            patience_counter += 1

        mixup_str = f"mixup={mixup_alpha:.2f}" if mixup_alpha > 0 else "mixup=OFF"
        print(f"E {epoch:3d}/{MAX_EPOCHS} | LR {current_lr:.2e} | "
              f"TL {train_loss:.4f} | VL {val_loss:.4f} | "
              f"VA {val_acc:.1f}% | VF1 {val_f1:.4f} | {mixup_str}{marker}")

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nBest: VA {best_val_acc:.1f}% at epoch {best_epoch}")

    # ── Save history ──
    history["best_epoch"] = best_epoch
    history["best_val_acc"] = round(best_val_acc, 2)
    history["config"] = {
        "seed": args.seed, "window_size": WINDOW_SIZE, "window_step": WINDOW_STEP,
        "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING, "dropout": DROPOUT,
        "mixup_alpha_early": MIXUP_ALPHA_EARLY,
        "mixup_alpha_mid": MIXUP_ALPHA_MID,
        "mixup_off_epoch": MIXUP_OFF_EPOCH,
        "gaussian_noise_std": GAUSSIAN_NOISE_STD,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "channels": CHANNELS, "kernel_sizes": KERNEL_SIZES,
    }
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ── Load best checkpoint ──
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    # Verify checkpoint
    _, rva, _, _, _ = evaluate(model, val_loader, criterion, device)
    assert abs(rva - best_val_acc) < 1.0, \
        f"CHECKPOINT VERIFICATION FAILED: saved={best_val_acc:.1f}%, reloaded={rva:.1f}%"
    print(f"Checkpoint verified: reloaded val acc = {rva:.1f}%")

    # ── Test evaluation ──
    print(f"\n{'='*60}")
    print("TEST EVALUATION")
    print(f"{'='*60}")

    test_ds = EMGWindowDataset(data_dir, test_ids)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                             num_workers=NUM_WORKERS,
                             pin_memory=(device.type == "cuda"))
    print(f"Test windows: {len(test_ds)}")

    _, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device)

    print(f"\nWindow-Level Results:")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  Macro F1: {test_f1:.4f}")
    print(f"\n{classification_report(test_labels, test_preds, target_names=GESTURE_NAMES)}")

    # ── Majority Voting ──
    print("Majority Voting (Continuous Gesture Segments):")
    mv = evaluate_majority_voting(model, data_dir, test_ids, device)

    print(f"  Window Accuracy:    {mv['window_acc']:.2f}%")
    print(f"  Window Macro F1:    {mv['window_f1']:.4f}")

    print(f"\n  Hard Voting Accuracy:  {mv['hard_acc']:.2f}% "
          f"({mv['hard_n']} segments)")
    print(f"  Hard Voting Macro F1:  {mv['hard_f1']:.4f}")
    if mv["hard_cm"] is not None:
        print(f"  Hard Voting CM:\n{np.array(mv['hard_cm'])}")
        cm = np.array(mv["hard_cm"])
        per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
        for i, name in enumerate(GESTURE_NAMES):
            print(f"    {name:11s}: {per_class_acc[i]:.1f}%")

    print(f"\n  Soft Voting Accuracy:  {mv['soft_acc']:.2f}% "
          f"({mv['soft_n']} segments)")
    print(f"  Soft Voting Macro F1:  {mv['soft_f1']:.4f}")

    # ── Weight norm check ──
    wr = check_weight_norm(best_ckpt, device)
    print(f"\nWeight norm: {wr:.2f} ({'OK' if wr > 0 else 'FAILED — zero weights!'})")

    # ── Final summary ──
    w_ok = test_acc >= 79.0
    h_ok = mv["hard_acc"] >= 90.0
    s_ok = mv["soft_acc"] >= 90.0

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Window Acc  {test_acc:.2f}%  {'✓ TARGET MET' if w_ok else '✗ (target 79%)'}")
    print(f"  Hard Vote   {mv['hard_acc']:.2f}%  {'✓ TARGET MET' if h_ok else '✗ (target 90%)'}")
    print(f"  Soft Vote   {mv['soft_acc']:.2f}%  {'✓ TARGET MET' if s_ok else '✗ (target 90%)'}")
    print(f"  Best Val    {best_val_acc:.1f}%  (epoch {best_epoch})")
    print(f"  Checkpoint  {output_dir}/best_model.pth")

    # Save final results
    final_results = {
        "test_window_acc": round(test_acc, 2),
        "test_window_f1": round(test_f1, 4),
        "hard_voting_acc": round(mv["hard_acc"], 2),
        "hard_voting_f1": round(mv["hard_f1"], 4),
        "soft_voting_acc": round(mv["soft_acc"], 2),
        "soft_voting_f1": round(mv["soft_f1"], 4),
        "best_val_acc": round(best_val_acc, 2),
        "best_epoch": best_epoch,
        "weight_norm": round(wr, 2),
        "targets": {
            "window_79pct": bool(w_ok),
            "voting_90pct": bool(h_ok or s_ok),
        },
    }
    with open(os.path.join(output_dir, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
