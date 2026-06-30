# data/libero_dataset.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import json
from transformers import CLIPTokenizer

class ActionNormalizer:
    """
    Online mean/std normalizer for action chunks.
    Call fit() once on the training dataset, then use
    normalize() in the train loop and denormalize() at inference.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps  = eps
        self.mean = None
        self.std  = None

    def fit(self, dataset) -> "ActionNormalizer":
        all_actions = []
        for sample in dataset:
            actions = sample["actions"]           # [chunk_size, 7]
            mask    = sample["action_mask"].bool() # [chunk_size]
            all_actions.append(actions[mask])      # only real (non-padded) steps
        all_actions = torch.cat(all_actions, dim=0)               # [N, 7]
        self.mean   = all_actions.mean(dim=0)                     # [7]
        self.std    = all_actions.std(dim=0).clamp(min=self.eps)  # [7]
        print(f"[ActionNormalizer] fit on {all_actions.shape[0]:,} steps")
        return self

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.mean.to(actions.device)) / self.std.to(actions.device)

    def denormalize(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * self.std.to(actions.device) + self.mean.to(actions.device)

    def save(self, path: str):
        torch.save({"mean": self.mean, "std": self.std}, path)

    @classmethod
    def load(cls, path: str) -> "ActionNormalizer":
        ckpt = torch.load(path, map_location="cpu")
        n = cls()
        n.mean, n.std = ckpt["mean"], ckpt["std"]
        return n
    

class LiberoDataset(Dataset):
    """
    Loads LIBERO demonstration episodes for Mini-VLA v2 training.

    v2 changes vs v1:
      - Returns 'wrist_image' (eye_in_hand_rgb) in addition to 'image' (agentview_rgb)
      - Tokenizer upgraded from char-level (vocab=1000) to CLIP BPE (vocab=49408)

    Each sample is one (observation, action_chunk) pair from an episode.
    """

    def __init__(
        self,
        dataset_path:   str,
        chunk_size:     int  = 16,
        image_size:     int  = 256,
        seq_len:        int  = 77,      # CLIP max sequence length
        num_episodes:   int  = None,
        skip_episodes:  int  = 0,
    ):
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.seq_len    = seq_len
        self.hdf5_path  = dataset_path

        # --- Load episode keys ---
        with h5py.File(dataset_path, "r") as f:
            demo_keys = list(f["data"].keys())
            demo_keys = demo_keys[skip_episodes:]
            if num_episodes is not None and num_episodes > 0:
                demo_keys = demo_keys[:num_episodes]
            self.num_episodes = len(demo_keys)

        # --- Build flat (episode, timestep) index ---
        self.samples = []
        with h5py.File(dataset_path, "r") as f:
            for ep_key in demo_keys:
                T = f["data"][ep_key]["actions"].shape[0]
                for t in range(T):
                    self.samples.append((ep_key, t))

        # --- Tokenize task instruction (CLIP) ---
        with h5py.File(dataset_path, "r") as f:
            problem_info         = json.loads(f["data"].attrs["problem_info"])
            self.task_instruction = "".join(problem_info["language_instruction"])

        self.tokens, self.text_mask = self._tokenize(self.task_instruction)

    def _tokenize(self, text: str):
        """
        TODO: replace with CLIP tokenizer.

        v1 used a simple char-level tokenizer (vocab=1000).
        v2 should use CLIPTokenizer from HuggingFace:

            from transformers import CLIPTokenizer
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            enc = tokenizer(text, max_length=self.seq_len, padding="max_length",
                            truncation=True, return_tensors="pt")
            tokens    = enc["input_ids"].squeeze(0)       # [seq_len]  long
            text_mask = enc["attention_mask"].squeeze(0)  # [seq_len]  float

        For now a placeholder is kept so the rest of the pipeline runs.
        Swap it out when you add the transformers dependency.
        """

        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        enc = tokenizer(
            text,
            max_length=self.seq_len,       
            padding="max_length",
            truncation=True,
            # return_tensors="pt",
        )
        tokens    = torch.tensor(enc["input_ids"],      dtype=torch.long)
        text_mask = torch.tensor(enc["attention_mask"], dtype=torch.float32)  
        
        # --- PLACEHOLDER: char-level fallback (remove after CLIP integration) ---
        # vocab_size = 49408
        # text  = text[:self.seq_len]
        # ids   = [ord(c) % vocab_size for c in text]
        # pad   = self.seq_len - len(ids)
        # tokens    = torch.tensor(ids + [0] * pad, dtype=torch.long)
        # text_mask = torch.tensor([1] * len(ids) + [0] * pad, dtype=torch.float32)
        return tokens, text_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_key, t = self.samples[idx]

        with h5py.File(self.hdf5_path, "r") as f:
            ep = f["data"][ep_key]
            T  = ep["actions"].shape[0]

            # --- agentview image [3, H, W] ---
            img = ep["obs"]["agentview_rgb"][t]                              # [H, W, 3] uint8
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0   # [3, H, W]

            # --- eye-in-hand (wrist) image [3, H, W]  (NEW in v2) ---
            # TODO: confirm the key name in your HDF5 file.
            # Common keys: "eye_in_hand_rgb", "robot0_eye_in_hand_image"
            wrist = ep["obs"]["eye_in_hand_rgb"][t]                                  # [H, W, 3]
            wrist = torch.from_numpy(wrist).permute(2, 0, 1).float() / 255.0        # [3, H, W]

            # --- State [8] ---
            eef_pos = ep["obs"]["ee_pos"][t]            # [3]
            eef_ori = ep["obs"]["ee_ori"][t]            # [3]
            gripper = ep["obs"]["gripper_states"][t]    # [2]
            state   = np.concatenate([eef_pos, eef_ori, gripper])  # [8]
            state   = torch.from_numpy(state).float()

            # --- Action chunk [chunk_size, 7] with validity mask ---
            end          = min(t + self.chunk_size, T)
            actions_real = ep["actions"][t:end]
            actual_len   = end - t
            pad_len      = self.chunk_size - actual_len

            if pad_len > 0:
                actions_padded = np.concatenate(
                    [actions_real, np.zeros((pad_len, 7))], axis=0
                )
            else:
                actions_padded = actions_real

            actions     = torch.from_numpy(actions_padded).float()          # [16, 7]
            action_mask = torch.tensor(
                [1.0] * actual_len + [0.0] * pad_len
            )                                                                # [16]

        return {
            "image":       img,              # [3, H, W]  agentview
            "wrist_image": wrist,            # [3, H, W]  eye_in_hand  (NEW)
            "tokens":      self.tokens,      # [seq_len]
            "text_mask":   self.text_mask,   # [seq_len]
            "state":       state,            # [8]
            "actions":     actions,          # [chunk_size, 7]
            "action_mask": action_mask,      # [chunk_size]
        }
