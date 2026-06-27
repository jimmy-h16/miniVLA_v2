import os
import glob
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from data.libero_dataset import LiberoDataset
from models.mini_vla import MiniVLA


# --- Config ---
DATASET_DIR  = os.environ.get("LIBERO_DATASET_DIR", os.path.expanduser("~/.robosuite/datasets"))
TASK_INDICES = list(range(1))

EPOCHS        = 100
BATCH_SIZE    = 16
LR            = 1e-4
CHUNK_SIZE    = 16
MAX_EPISODES  = 50
VAL_SPLIT     = 0.1   # 10% → 5 val episodes, 45 train episodes
WARMUP_EPOCHS = 5


# --- Dataset ---
all_hdf5_files = sorted(glob.glob(os.path.join(DATASET_DIR, "**/*.hdf5"), recursive=True))
assert len(all_hdf5_files) > 0, f"No HDF5 files found under {DATASET_DIR}/"

if TASK_INDICES is not None:
    hdf5_files = [all_hdf5_files[i] for i in TASK_INDICES]
else:
    hdf5_files = all_hdf5_files

print(f"\n[Train] Selected {len(hdf5_files)} tasks:")
for i, p in enumerate(hdf5_files):
    print(f"  [{i}] {os.path.basename(p)}")


# --- Train / Val Episode Split (deterministic, no shuffle) ---
all_episode_indices = list(range(MAX_EPISODES))

n_val   = max(1, int(VAL_SPLIT * MAX_EPISODES))   # = 5
n_train = MAX_EPISODES - n_val                     # = 45

train_indices = set(all_episode_indices[:n_train])   # episodes 0–44
val_indices   = set(all_episode_indices[n_train:])   # episodes 45–49

print(f"\n[Train] Episode split: {n_train} train / {n_val} val")


# --- Build Datasets ---
def build_datasets(hdf5_files, episode_indices, chunk_size):
    datasets = []
    for p in hdf5_files:
        full_ds = LiberoDataset(dataset_path=p, chunk_size=chunk_size, num_episodes=MAX_EPISODES)
        subset_indices = [
            i for i in range(len(full_ds))
            if full_ds.get_episode_index(i) in episode_indices
        ]
        datasets.append(Subset(full_ds, subset_indices))
    return ConcatDataset(datasets)

train_dataset = build_datasets(hdf5_files, train_indices, CHUNK_SIZE)
val_dataset   = build_datasets(hdf5_files, val_indices,   CHUNK_SIZE)

print(f"[Train] {len(train_dataset)} train samples | {len(val_dataset)} val samples")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# --- Model ---
device   = "mps"
model    = MiniVLA().to(device)
optim    = torch.optim.AdamW(model.parameters(), lr=LR)
lossFunc = torch.nn.MSELoss(reduction="none")

print(f"[Train] Parameters: {model.count_parameters():,}")


# --- Scheduler: Linear Warmup → Cosine Decay ---
warmup = LinearLR(
    optim,
    start_factor=0.1,        # start at 10% of LR (1e-5)
    end_factor=1.0,           # reach full LR (1e-4) by end of warmup
    total_iters=WARMUP_EPOCHS,
)
cosine = CosineAnnealingLR(
    optim,
    T_max=EPOCHS - WARMUP_EPOCHS,
    eta_min=1e-6,
)
scheduler = SequentialLR(optim, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])


# --- Helpers ---
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in loader:
            img        = batch["image"].to(device)
            wrist      = batch["wrist_image"].to(device)
            state      = batch["state"].to(device)
            action     = batch["actions"].to(device)
            token      = batch["tokens"].to(device)
            actionMask = batch["action_mask"].to(device)
            textMask   = batch["text_mask"].to(device)

            outputAction = model(img, wrist, token, textMask, state)

            if train:
                optim.zero_grad()

            loss = lossFunc(outputAction, action)                          # [B, 16, 7]
            loss = loss * actionMask.unsqueeze(-1)                         # mask padding steps
            loss = loss.sum() / actionMask.sum().clamp(min=1) / action.shape[-1]

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()

            total_loss += loss.item()

    return total_loss / len(loader)


# --- Training Loop ---
os.makedirs("checkpoints", exist_ok=True)
best_val_loss = float("inf")

print(f"\n[Train] Starting training for {EPOCHS} epochs...")
print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'LR':>10}")
print("-" * 45)

for epoch in range(EPOCHS):
    train_loss = run_epoch(train_loader, train=True)
    val_loss   = run_epoch(val_loader,   train=False)

    scheduler.step()

    current_lr = scheduler.get_last_lr()[0]
    print(f"{epoch+1:>6}/{EPOCHS}  {train_loss:>10.4f}  {val_loss:>10.4f}  {current_lr:>10.2e}")

    # Save best checkpoint based on val loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_ckpt = "checkpoints/mini_vla_v2_best.pt"
        torch.save({
            "epoch":      epoch + 1,
            "model":      model.state_dict(),
            "optim":      optim.state_dict(),
            "val_loss":   val_loss,
            "train_loss": train_loss,
        }, best_ckpt)
        print(f"           ✓ New best val loss {val_loss:.4f} → saved {best_ckpt}")


# --- Save Final Checkpoint ---
final_ckpt = f"checkpoints/mini_vla_v2_final_val{best_val_loss:.4f}.pt"
torch.save(model.state_dict(), final_ckpt)
print(f"\n[Train] Done. Best val loss: {best_val_loss:.4f}")
print(f"[Train] Final checkpoint:    {final_ckpt}")