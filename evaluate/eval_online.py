# evaluate/eval_online.py
import os
import sys
import torch
import numpy as np
import cv2
import robosuite.utils.transform_utils as T
from transformers import CLIPTokenizer

# os.environ["LIBERO_PATH"] = "/Users/jimmy/Documents/UST/MiniVLA/MiniVLA_v2/LIBERO/libero"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from models.mini_vla import MiniVLA

# ---------------------------------------------------------------------------
# Pytorch Patch
# ---------------------------------------------------------------------------
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TASK_SUITE   = "libero_spatial"
TASK_INDICES = list(range(1))
NUM_EPISODES = 50
MAX_STEPS    = 300
CHUNK_SIZE   = 16
SEQ_LEN      = 77       # CLIP max length
CAMERA_H     = 256
CAMERA_W     = 256
DEVICE       = "mps"
CHECKPOINT   = "checkpoints/mini_vla_v2_best.pt"
ACTION_HORIZON = 16
GENERATE_VIDEO = True

# ---------------------------------------------------------------------------
# Tokenizer  (keep in sync with LiberoDataset._tokenize)
# ---------------------------------------------------------------------------
def _tokenize(text: str):
    """
    TODO: replace with CLIP tokenizer (same change as in libero_dataset.py).

        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        enc = tokenizer(text, max_length=SEQ_LEN, padding="max_length",
                        truncation=True, return_tensors="pt")
        return enc["input_ids"].squeeze(0), enc["attention_mask"].float().squeeze(0)
    """
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    enc = tokenizer(
        text,
        max_length=SEQ_LEN,       # 77
        padding="max_length",
        truncation=True,
        # return_tensors="pt",
    )
    tokens    = torch.tensor(enc["input_ids"],      dtype=torch.long)
    text_mask = torch.tensor(enc["attention_mask"], dtype=torch.float32)

    
    # vocab_size = 49408
    # text      = text[:SEQ_LEN]
    # ids       = [ord(c) % vocab_size for c in text]
    # pad       = SEQ_LEN - len(ids)
    # tokens    = torch.tensor(ids + [0] * pad, dtype=torch.long)
    # text_mask = torch.tensor([1] * len(ids) + [0] * pad, dtype=torch.float32)
    return tokens, text_mask

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = MiniVLA().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()
print(f"[Eval] Loaded checkpoint: {CHECKPOINT}")

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
benchmark_dict = benchmark.get_benchmark_dict()
task_suite_obj = benchmark_dict[TASK_SUITE]()

os.makedirs("videos", exist_ok=True)
all_results = {}

for TASK_IDX in TASK_INDICES:
    task = task_suite_obj.get_task(TASK_IDX)

    tokens, textMask = _tokenize(task.language)
    tokens   = tokens.unsqueeze(0).to(DEVICE)
    textMask = textMask.unsqueeze(0).to(DEVICE)

    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=CAMERA_H,
        camera_widths=CAMERA_W,
    )
    env.seed(0)

    init_states     = task_suite_obj.get_task_init_states(TASK_IDX)
    num_init_states = len(init_states)

    print(f"\n[Task {TASK_IDX:>2}] {task.language}")
    successes = []

    with torch.no_grad():
        for ep_idx in range(NUM_EPISODES):
            obs = env.reset()
            env.set_init_state(init_states[ep_idx % num_init_states])

            ep_success    = False
            step_in_chunk = CHUNK_SIZE
            action_chunk  = None
            frames        = []

            for step in range(MAX_STEPS):
                frames.append(obs["agentview_image"].copy())

                if step_in_chunk >= ACTION_HORIZON:
                    # --- agentview ---
                    img = obs["agentview_image"]
                    img = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
                    img = img.unsqueeze(0).to(DEVICE)

                    # --- wrist (eye_in_hand)  TODO: confirm obs key name ---
                    wrist = obs["robot0_eye_in_hand_image"]
                    wrist = torch.from_numpy(wrist.copy()).permute(2, 0, 1).float() / 255.0
                    wrist = wrist.unsqueeze(0).to(DEVICE)

                    eef_pos = obs["robot0_eef_pos"]
                    eef_ori = T.quat2axisangle(obs["robot0_eef_quat"])
                    gripper = obs["robot0_gripper_qpos"]
                    state   = np.concatenate([eef_pos, eef_ori, gripper])
                    state   = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

                    # TODO: model signature updated — wrist is now 2nd arg
                    output_action = model(img, wrist, tokens, textMask, state)
                    action_chunk  = output_action.squeeze(0).cpu().numpy()
                    step_in_chunk = 0

                obs, reward, done, info = env.step(action_chunk[step_in_chunk].tolist())
                step_in_chunk += 1

                if reward > 0:
                    ep_success = True
                    break
                if done:
                    break

            successes.append(ep_success)
            status    = "SUCCESS" if ep_success else "FAIL"
            if GENERATE_VIDEO:
                video_path = f"videos/task{TASK_IDX:02d}_ep{ep_idx+1:03d}_{status}.mp4"
                h, w      = frames[0].shape[:2]
                writer    = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"avc1"), 20, (w, h))
                for frame in frames:
                    writer.write(cv2.cvtColor(cv2.flip(frame, 0), cv2.COLOR_RGB2BGR))
                writer.release()
            print(f"  Episode {ep_idx+1:>3}/{NUM_EPISODES} | {status} | steps={step+1}")

    env.close()
    success_rate          = sum(successes) / len(successes)
    all_results[task.name] = success_rate
    print(f"  → Task success rate: {success_rate*100:.1f}% ({sum(successes)}/{NUM_EPISODES})")

# Summary
print("\n" + "="*65)
print(f"  Multi-task Eval  |  Suite: {TASK_SUITE}")
print("="*65)
for name, rate in all_results.items():
    print(f"  {name:<55} {rate*100:5.1f}%")
print("-"*65)
avg = sum(all_results.values()) / len(all_results)
print(f"  Average                                                  {avg*100:5.1f}%")
print("="*65)
