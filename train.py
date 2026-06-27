import os
import glob
import torch
from torch.utils.data import DataLoader, ConcatDataset
from data.libero_dataset import LiberoDataset
from models.mini_vla import MiniVLA

# --- Config ---
DATASET_DIR  = os.environ.get("LIBERO_DATASET_DIR", os.path.expanduser("~/.robosuite/datasets"))
TASK_INDICES = list(range(1))

EPOCHS     = 10
BATCH_SIZE = 16
LR         = 1e-4
CHUNK_SIZE = 16
MAX_EPISODES = 50

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

datasets = [
    LiberoDataset(dataset_path=p, chunk_size=CHUNK_SIZE, num_episodes=MAX_EPISODES)
    for p in hdf5_files
]
dataset = ConcatDataset(datasets)
print(f"\n[Train] {len(hdf5_files)} tasks | {len(dataset)} total samples")

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# --- Model ---
device   = "mps"
model    = MiniVLA().to(device)
optim    = torch.optim.AdamW(model.parameters(), lr=LR)
lossFunc = torch.nn.MSELoss(reduction="none")

print(f"[Train] Parameters: {model.count_parameters():,}")

# --- Training Loop ---
print(f"\n[Train] Starting training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in loader:
        img         = batch["image"].to(device)          # [B, 3, H, W]
        wrist       = batch["wrist_image"].to(device)    # [B, 3, H, W]  NEW
        state       = batch["state"].to(device)
        action      = batch["actions"].to(device)
        token       = batch["tokens"].to(device)
        actionMask  = batch["action_mask"].to(device)
        textMask    = batch["text_mask"].to(device)

        # TODO: model now takes wrist_image as second argument
        outputAction = model(img, wrist, token, textMask, state)
        optim.zero_grad()

        loss = lossFunc(outputAction, action)                          # [B, 16, 7]
        loss = loss * actionMask.unsqueeze(-1)                         # mask padding steps
        loss = loss.sum() / actionMask.sum().clamp(min=1) / action.shape[-1]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}  loss={loss/len(loader):.4f}")

# --- Save Checkpoint ---
os.makedirs("checkpoints", exist_ok=True)
ckpt_name = f"checkpoints/mini_vla_v2_{total_loss:.4f}.pt"
torch.save(model.state_dict(), ckpt_name)
print(f"Saved checkpoint: {ckpt_name}")
