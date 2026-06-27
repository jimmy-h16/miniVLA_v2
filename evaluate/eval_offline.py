# evaluate/eval_offline.py
import os
import sys
import glob
import torch
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.libero_dataset import LiberoDataset
from models.mini_vla import MiniVLA

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_DIR  = os.environ.get("LIBERO_DATASET_DIR", os.path.expanduser("~/.robosuite/datasets"))
TASK_INDICES = list(range(1))
CHECKPOINT   = "checkpoints/mini_vla_v2_1.pt"
BATCH_SIZE   = 32
CHUNK_SIZE   = 16
MAX_EPISODES = 10
DEVICE       = "mps"

# ---------------------------------------------------------------------------
# Dataset (held-out episodes)
# ---------------------------------------------------------------------------
all_hdf5_files = sorted(glob.glob(os.path.join(DATASET_DIR, "**/*.hdf5"), recursive=True))
hdf5_files     = [all_hdf5_files[i] for i in TASK_INDICES]

datasets = [
    LiberoDataset(
        dataset_path=p,
        chunk_size=CHUNK_SIZE,
        num_episodes=MAX_EPISODES,
        skip_episodes=50,           # skip training episodes
    )
    for p in hdf5_files
]
dataset = ConcatDataset(datasets)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = MiniVLA().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()
print(f"[Eval] Loaded checkpoint: {CHECKPOINT}")

lossFunc  = torch.nn.MSELoss(reduction="none")
total_loss = 0.0
total_samples = 0

with torch.no_grad():
    for batch in loader:
        img        = batch["image"].to(DEVICE)
        wrist      = batch["wrist_image"].to(DEVICE)   # NEW
        state      = batch["state"].to(DEVICE)
        action     = batch["actions"].to(DEVICE)
        token      = batch["tokens"].to(DEVICE)
        actionMask = batch["action_mask"].to(DEVICE)
        textMask   = batch["text_mask"].to(DEVICE)

        # TODO: updated model signature
        pred  = model(img, wrist, token, textMask, state)
        loss  = lossFunc(pred, action)
        loss  = loss * actionMask.unsqueeze(-1)
        loss  = loss.sum() / actionMask.sum().clamp(min=1) / action.shape[-1]

        total_loss    += loss.item() * img.size(0)
        total_samples += img.size(0)

print(f"[Eval] Offline MSE loss: {total_loss / total_samples:.4f}")
