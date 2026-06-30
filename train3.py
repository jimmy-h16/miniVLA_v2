import os
import glob
import torch
import numpy as np, random, torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from data.libero_dataset import LiberoDataset
from models.mini_vla import MiniVLA


# ============================================================
# Config
# ============================================================
np.random.seed(42); random.seed(42); torch.manual_seed(42)

DATASET_DIR  = os.environ.get("LIBERO_DATASET_DIR", os.path.expanduser("~/.robosuite/datasets"))
TASK_INDICES = list(range(1))   # which HDF5 files to use (None → all)

N_TRAIN_EP   = 50               # explicit episode counts (no ratio needed)
N_VAL_EP     = 0
TOTAL_EP     = N_TRAIN_EP + N_VAL_EP   # = 50

EPOCHS        = 40
BATCH_SIZE    = 16
LR            = 1e-4
CHUNK_SIZE    = 16
WARMUP_EPOCHS = 5
PERIOD        = 4

DEVICE = "mps"    # change to "cuda" or "cpu" as needed


# ============================================================
# Dataset helpers
# ============================================================
def build_split_datasets(hdf5_files: list[str],
                         chunk_size: int,
                         n_train: int,
                         n_val: int) -> tuple:
    """
    Load each HDF5 file twice — once for the first `n_train` episodes
    and once for the last `n_val` episodes — so the split is exact and
    deterministic regardless of episode length.

    Returns (train_dataset, val_dataset).
    """
    train_datasets, val_datasets = [], []

    for p in hdf5_files:
        # Training episodes: 0 … n_train-1
        train_ds = LiberoDataset(
            dataset_path=p,
            chunk_size=chunk_size,
            num_episodes=n_train,
            skip_episodes=0,
        )
        train_datasets.append(train_ds)

        # Validation episodes: n_train … n_train+n_val-1
        val_ds = LiberoDataset(
            dataset_path=p,
            chunk_size=chunk_size,
            num_episodes=n_val,
            skip_episodes=n_train,
        )
        val_datasets.append(val_ds)

    return ConcatDataset(train_datasets), ConcatDataset(val_datasets)


def run_epoch(model, loader, device, loss_fn, optimizer=None):
    """
    One forward pass over `loader`.
    If optimizer is provided → training mode (gradients + update).
    Otherwise             → eval mode (no gradients).

    Returns average loss over all batches.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss  = 0.0
    num_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            img        = batch["image"].to(device)          # [B, 3, H, W]
            wrist      = batch["wrist_image"].to(device)    # [B, 3, H, W]
            state      = batch["state"].to(device)          # [B, 8]
            action     = batch["actions"].to(device)        # [B, 16, 7]
            token      = batch["tokens"].to(device)         # [B, seq_len]
            actionMask = batch["action_mask"].to(device)    # [B, 16]
            textMask   = batch["text_mask"].to(device)      # [B, seq_len]

            pred = model(img, wrist, token, textMask, state)   # [B, 16, 7]

            if is_train:
                optimizer.zero_grad()

            loss = loss_fn(pred, action)                         # [B, 16, 7]
            loss = loss * actionMask.unsqueeze(-1)               # zero out padded steps
            loss = loss.sum() / actionMask.sum().clamp(min=1) / action.shape[-1]

            if is_train:
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss  += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  miniVLA v2 — Training Script (train3.py)")
    print("=" * 60)

    # ---- Discover HDF5 files ----------------------------------------
    all_hdf5 = sorted(glob.glob(os.path.join(DATASET_DIR, "**/*.hdf5"), recursive=True))
    assert len(all_hdf5) > 0, f"No HDF5 files found under {DATASET_DIR}/"

    hdf5_files = [all_hdf5[i] for i in TASK_INDICES] if TASK_INDICES is not None else all_hdf5

    print(f"\n[Dataset] Found {len(hdf5_files)} task(s):")
    for i, p in enumerate(hdf5_files):
        print(f"  [{i}] {os.path.basename(p)}")

    # ---- Episode split ----------------------------------------------
    print(f"\n[Dataset] Episode split:")
    print(f"  Train episodes : {N_TRAIN_EP} (ep 0 – {N_TRAIN_EP - 1})")
    print(f"  Val   episodes : {N_VAL_EP}  (ep {N_TRAIN_EP} – {TOTAL_EP - 1})")

    train_dataset, val_dataset = build_split_datasets(
        hdf5_files, CHUNK_SIZE, N_TRAIN_EP, N_VAL_EP
    )

    print(f"\n[Dataset] Samples after chunking:")
    print(f"  Train : {len(train_dataset):,} samples")
    print(f"  Val   : {len(val_dataset):,} samples")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ---- Model & optimiser -----------------------------------------
    model    = MiniVLA().to(DEVICE)
    optim    = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn  = torch.nn.MSELoss(reduction="none")

    print(f"\n[Model] MiniVLA v2")
    print(f"  Trainable parameters : {model.count_parameters():,}")
    print(f"  Device               : {DEVICE}")

    # ---- Scheduler: warmup → cosine ---------------------------------
    # warmup_sched = LinearLR(optim, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    # cosine_sched = CosineAnnealingLR(optim, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    # scheduler    = SequentialLR(optim, schedulers=[warmup_sched, cosine_sched],
    #                             milestones=[WARMUP_EPOCHS])

    # print(f"\n[Scheduler] Linear warmup ({WARMUP_EPOCHS} ep) → Cosine annealing ({EPOCHS - WARMUP_EPOCHS} ep)")

    # ---- Training loop ---------------------------------------------
    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt_path = None

    header = f"{'Epoch':>7}  {'Train Loss':>11}  {'Val Loss':>10}  {'LR':>10}  {'Best?':>6}"
    sep    = "-" * len(header)

    print(f"\n[Training] {EPOCHS} epochs  |  batch={BATCH_SIZE}  |  lr={LR:.0e}")
    print(sep)
    print(header)
    print(sep)

    for epoch in range(EPOCHS):
        train_loss = run_epoch(model, train_loader, DEVICE, loss_fn, optimizer=optim)
        val_loss   = run_epoch(model, val_loader,   DEVICE, loss_fn, optimizer=None)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        is_best = val_loss < best_val_loss
        marker  = "  ✓" if is_best else ""

        print(f"{epoch+1:>6}/{EPOCHS}  {train_loss:>11.5f}  {val_loss:>10.5f}  {current_lr:>10.2e}{marker}")

        if is_best:
            best_val_loss  = val_loss
            best_ckpt_path = "checkpoints/mini_vla_v2_best.pt"
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"          → Saved best checkpoint: {best_ckpt_path}")

        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % PERIOD == 0:
            periodic = f"checkpoints/mini_vla_v2_ep{epoch+1:04d}.pt"
            torch.save(model.state_dict(), periodic)
            print(f"          → Periodic checkpoint : {periodic}")

    # ---- Final summary ---------------------------------------------
    print(sep)
    print(f"\n[Done] Training complete.")
    print(f"  Best val loss      : {best_val_loss:.5f}")
    print(f"  Best checkpoint    : {best_ckpt_path}")

    final_ckpt = f"checkpoints/mini_vla_v2_final_val{best_val_loss:.5f}.pt"
    torch.save(model.state_dict(), final_ckpt)
    print(f"  Final checkpoint   : {final_ckpt}")
    print("=" * 60)


if __name__ == "__main__":
    main()
