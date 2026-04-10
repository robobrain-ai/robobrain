#!/usr/bin/env python3
"""
RoboBrain v6 Training: Gemma 4 E2B (frozen) + 187M action head on LIBERO demos.

This is our own model trained on open data — not distillation, not SmolVLA.
Same approach as SmolVLA (frozen VLM + action head) but with Gemma 4 backbone.

Usage:
    # Single GPU (testing)
    python tools/train_robobrain_v6.py --steps 100

    # 8×H100 DDP
    torchrun --nproc_per_node=8 tools/train_robobrain_v6.py

    # With pre-cached features (60x faster, run cache step first)
    python tools/train_robobrain_v6.py --cache-features
    torchrun --nproc_per_node=8 tools/train_robobrain_v6.py --use-cached
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robobrain.model.action_head import TransformerFlowActionHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "dataset": "HuggingFaceVLA/libero",
    "vlm_model": "google/gemma-4-E2B-it",
    "action_dim": 7,
    "action_horizon": 10,  # chunk size (steps predicted per inference)
    "hidden_dim": 768,
    "num_layers": 12,
    "num_heads": 8,
    "mlp_dim": 3072,
    "num_inference_steps": 10,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "warmup_steps": 1000,
    "total_steps": 200000,
    "batch_size_per_gpu": 4,
    "grad_accum": 4,
    "save_every": 10000,
    "log_every": 50,
    "eval_every": 10000,
    "output_dir": "/tmp/robobrain_v6",
    "cache_dir": "/tmp/robobrain_v6_cache",
}


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_lerobot_dataset(repo_id):
    """Load dataset via LeRobot and extract observation-action pairs."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    logger.info(f"Loading dataset: {repo_id}")
    ds = LeRobotDataset(repo_id)
    logger.info(f"Dataset: {ds.num_episodes} episodes, {len(ds)} frames")

    return ds


def build_action_chunks(ds, horizon):
    """Convert frame-level dataset into (observation, action_chunk) pairs."""
    logger.info(f"Building action chunks with horizon={horizon}...")

    pairs = []
    ep_starts = {}

    # Group frames by episode
    for i in range(len(ds)):
        ep_idx = ds[i]["episode_index"].item()
        if ep_idx not in ep_starts:
            ep_starts[ep_idx] = []
        ep_starts[ep_idx].append(i)

    for ep_idx, frame_indices in ep_starts.items():
        frames = sorted(frame_indices)
        # Create chunks with stride = horizon//2 (50% overlap)
        stride = max(1, horizon // 2)
        for start in range(0, len(frames) - horizon, stride):
            obs_idx = frames[start]
            action_indices = frames[start:start + horizon]

            # Get observation at chunk start
            obs_frame = ds[obs_idx]

            # Stack actions for the chunk
            actions = torch.stack([ds[j]["action"] for j in action_indices])  # (horizon, action_dim)

            # Get task description
            task = obs_frame.get("task", "perform the task")
            if isinstance(task, torch.Tensor):
                task = "perform the task"

            pairs.append({
                "image": obs_frame["observation.images.image"],      # (C, H, W) tensor
                "image2": obs_frame.get("observation.images.image2"), # wrist cam if available
                "state": obs_frame["observation.state"],             # state tensor
                "actions": actions,                                   # (horizon, action_dim)
                "task": str(task),
            })

    logger.info(f"Built {len(pairs)} action chunks from {len(ep_starts)} episodes")
    return pairs


# ─── Feature Caching ─────────────────────────────────────────────────────────

def cache_features(pairs, vlm, processor, device, cache_dir):
    """Pre-extract Gemma 4 features for all observations. 60x faster training."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Caching {len(pairs)} feature vectors to {cache_dir}...")
    vlm.eval()

    for i, pair in enumerate(pairs):
        cache_path = cache_dir / f"features_{i:06d}.pt"
        if cache_path.exists():
            continue

        # Convert tensor image to PIL
        img_tensor = pair["image"]
        if img_tensor.dtype == torch.float32 or img_tensor.dtype == torch.float16:
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            img_np = img_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
        image = Image.fromarray(img_np)

        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": pair["task"]},
        ]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = vlm(**inputs, output_hidden_states=True)
            features = out.hidden_states[-1].cpu()  # (1, seq_len, hidden)
            mask = inputs["attention_mask"].cpu()

        torch.save({"features": features, "mask": mask}, cache_path)

        if (i + 1) % 100 == 0:
            logger.info(f"Cached {i+1}/{len(pairs)}")

    logger.info(f"Feature caching complete: {len(pairs)} vectors")


# ─── Training ────────────────────────────────────────────────────────────────

def train(args):
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(vars(args))

    # DDP setup
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world_size, local_rank = 0, 1, 0

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        logger.info(f"World size: {world_size} GPUs")
        logger.info(f"Config: {json.dumps(cfg, indent=2, default=str)}")

    # Load dataset
    ds = load_lerobot_dataset(cfg["dataset"])
    pairs = build_action_chunks(ds, cfg["action_horizon"])

    # Compute action normalization
    all_actions = torch.stack([p["actions"] for p in pairs])  # (N, horizon, action_dim)
    action_mean = all_actions.mean(dim=(0, 1)).numpy()
    action_std = all_actions.std(dim=(0, 1)).numpy() + 1e-6
    if rank == 0:
        logger.info(f"Action mean: {np.round(action_mean, 3)}")
        logger.info(f"Action std:  {np.round(action_std, 3)}")
        os.makedirs(cfg["output_dir"], exist_ok=True)
        with open(os.path.join(cfg["output_dir"], "norm_stats.json"), "w") as f:
            json.dump({"mean": action_mean.tolist(), "std": action_std.tolist()}, f)

    # Load VLM (frozen, per-GPU)
    if not cfg.get("use_cached"):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        if rank == 0:
            logger.info(f"Loading VLM: {cfg['vlm_model']}")
        vlm = AutoModelForImageTextToText.from_pretrained(
            cfg["vlm_model"], torch_dtype=torch.bfloat16
        ).to(device)
        processor = AutoProcessor.from_pretrained(cfg["vlm_model"], padding_side="left")
        vlm.eval()
        for p in vlm.parameters():
            p.requires_grad = False
        hidden_size = vlm.config.text_config.hidden_size
    else:
        # Using cached features — no VLM needed
        vlm, processor = None, None
        hidden_size = 1536  # Gemma 4 E2B
        if rank == 0:
            logger.info("Using cached features — no VLM loaded")

    # Cache features if requested
    if cfg.get("cache_features") and rank == 0:
        cache_features(pairs, vlm, processor, device, cfg["cache_dir"])
        logger.info("Caching done. Restart with --use-cached to train faster.")
        return

    # Build action head
    head = TransformerFlowActionHead(
        action_dim=cfg["action_dim"],
        action_horizon=cfg["action_horizon"],
        conditioning_dim=hidden_size,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        mlp_dim=cfg["mlp_dim"],
        num_inference_steps=cfg["num_inference_steps"],
    ).to(device).to(torch.bfloat16)

    if distributed:
        head = DDP(head, device_ids=[local_rank])
        head_module = head.module
    else:
        head_module = head

    if rank == 0:
        params = sum(p.numel() for p in head_module.parameters())
        logger.info(f"Action head: {params/1e6:.1f}M params")
        logger.info(f"Training pairs: {len(pairs)}")
        eff_batch = cfg["batch_size_per_gpu"] * cfg["grad_accum"] * world_size
        logger.info(f"Effective batch: {eff_batch}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    # Cosine schedule with warmup
    def lr_schedule(step):
        if step < cfg["warmup_steps"]:
            return step / max(1, cfg["warmup_steps"])
        progress = (step - cfg["warmup_steps"]) / max(1, cfg["total_steps"] - cfg["warmup_steps"])
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Training loop
    rng = np.random.RandomState(42 + rank)
    losses = []
    t0 = time.time()

    os.makedirs(cfg["output_dir"], exist_ok=True)

    for step in range(cfg["total_steps"]):
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(cfg["grad_accum"]):
            idx = rng.randint(len(pairs))
            pair = pairs[idx]

            # Normalize actions
            actions_np = pair["actions"].numpy()
            normalized = (actions_np - action_mean) / action_std
            actions = torch.tensor(normalized, dtype=torch.bfloat16).unsqueeze(0).to(device)

            if cfg.get("use_cached"):
                # Load pre-cached features
                cached = torch.load(
                    os.path.join(cfg["cache_dir"], f"features_{idx:06d}.pt"),
                    map_location=device, weights_only=True
                )
                cond = cached["features"].to(device).to(torch.bfloat16)
                mask = cached["mask"].to(device)
            else:
                # Extract features from VLM
                img_tensor = pair["image"]
                if img_tensor.dtype == torch.float32 or img_tensor.dtype == torch.float16:
                    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                else:
                    img_np = img_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
                image = Image.fromarray(img_np)

                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": pair["task"]},
                ]}]
                text = processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                inputs = processor(
                    text=[text], images=[image], return_tensors="pt", padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    out = vlm(**inputs, output_hidden_states=True)
                    cond = out.hidden_states[-1].to(torch.bfloat16)
                    mask = inputs["attention_mask"]

            loss = head_module.compute_loss(actions, cond, mask) / cfg["grad_accum"]
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(accum_loss)

        # Logging
        if rank == 0 and (step + 1) % cfg["log_every"] == 0:
            avg = np.mean(losses[-cfg["log_every"]:])
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed
            lr = scheduler.get_last_lr()[0]
            eff_batch = cfg["batch_size_per_gpu"] * cfg["grad_accum"] * world_size
            logger.info(
                f"Step {step+1}/{cfg['total_steps']} | Loss: {avg:.4f} | "
                f"LR: {lr:.6f} | Rate: {rate:.2f} steps/s | EffBatch: {eff_batch}"
            )

        # Save checkpoint
        if rank == 0 and (step + 1) % cfg["save_every"] == 0:
            path = os.path.join(cfg["output_dir"], f"checkpoint_{step+1}.pt")
            torch.save(head_module.state_dict(), path)
            logger.info(f"Saved: {path}")

    # Final save
    if rank == 0:
        path = os.path.join(cfg["output_dir"], "final.pt")
        torch.save(head_module.state_dict(), path)
        logger.info(f"Final loss: {np.mean(losses[-100:]):.4f}")
        logger.info(f"Saved: {path}")
        logger.info("TRAINING COMPLETE!")

    if distributed:
        dist.destroy_process_group()


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoboBrain v6 Training")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4, dest="batch_size_per_gpu")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--output-dir", default="/tmp/robobrain_v6")
    parser.add_argument("--cache-dir", default="/tmp/robobrain_v6_cache")
    parser.add_argument("--cache-features", action="store_true",
                        help="Pre-extract VLM features (run once, then use --use-cached)")
    parser.add_argument("--use-cached", action="store_true",
                        help="Use pre-cached features (60x faster)")
    parser.add_argument("--dataset", default="HuggingFaceVLA/libero")
    parser.add_argument("--vlm", default="google/gemma-4-E2B-it", dest="vlm_model")
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    train(args)
