#!/usr/bin/env python3
# Path setup MUST happen before ANY other imports
import sys
sys.path.insert(0, "/home/ubuntu/lerobot/src")

"""RoboBrain v6: Train Gemma 4 E2B + 187M action head on DROID real robot data.
Eval on LIBERO (held out). This is the generalization test.

Usage:
    # Single GPU test
    python tools/train_v6_droid.py --steps 100

    # 8xH100 DDP
    torchrun --nproc_per_node=8 tools/train_v6_droid.py
"""
import os, json, time, logging, argparse

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image

from robobrain.model.action_head import TransformerFlowActionHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

HORIZON = 10
ACTION_DIM = 7


def build_pairs(ds, num_episodes=20000, seed=42):
    """Sample episodes and build (observation, action_chunk) training pairs."""
    rng = np.random.RandomState(seed)
    n_eps = min(num_episodes, ds.num_episodes)
    sampled = set(rng.choice(ds.num_episodes, n_eps, replace=False).tolist())

    # Group frames by episode
    ep_frames = {}
    for i in range(len(ds)):
        ep = ds[i]["episode_index"].item()
        if ep in sampled:
            if ep not in ep_frames:
                ep_frames[ep] = []
            ep_frames[ep].append(i)

    logger.info(f"Grouped {len(ep_frames)} episodes")

    pairs = []
    for ep, frames in ep_frames.items():
        frames = sorted(frames)
        if len(frames) < HORIZON:
            continue
        stride = max(1, HORIZON // 2)
        for start in range(0, len(frames) - HORIZON, stride):
            obs_idx = frames[start]
            chunk_indices = frames[start:start + HORIZON]
            if len(chunk_indices) < HORIZON:
                break

            obs = ds[obs_idx]
            actions = torch.stack([ds[j]["action"][:ACTION_DIM] for j in chunk_indices])

            lang = ""
            for key in ["language_instruction", "task"]:
                val = obs.get(key, "")
                if val and not isinstance(val, torch.Tensor):
                    lang = str(val)
                    break
            if not lang:
                lang = "perform the task"

            # Get first available image
            img = None
            for img_key in ["observation.images.exterior_1_left",
                           "observation.images.exterior_2_left",
                           "observation.images.wrist_left"]:
                if img_key in obs:
                    img = obs[img_key]
                    break

            if img is None:
                continue

            pairs.append({
                "image": img,
                "actions": actions,
                "task": lang,
            })

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-episodes", type=int, default=20000)
    parser.add_argument("--output-dir", default="/tmp/robobrain_v6_droid")
    parser.add_argument("--vlm", default="google/gemma-4-E2B-it")
    args = parser.parse_args()

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

    # Load dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if rank == 0:
        logger.info("Loading DROID dataset...")
    ds = LeRobotDataset("lerobot/droid_1.0.1")
    if rank == 0:
        logger.info(f"DROID: {ds.num_episodes} episodes, {len(ds)} frames")

    # Build training pairs
    pairs = build_pairs(ds, num_episodes=args.num_episodes)
    if rank == 0:
        logger.info(f"Training pairs: {len(pairs)}")

    # Action normalization
    all_actions = torch.stack([p["actions"] for p in pairs])
    action_mean = all_actions.mean(dim=(0, 1)).numpy()
    action_std = all_actions.std(dim=(0, 1)).numpy() + 1e-6
    if rank == 0:
        logger.info(f"Action mean: {np.round(action_mean, 3)}")
        logger.info(f"Action std:  {np.round(action_std, 3)}")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(f"{args.output_dir}/norm_stats.json", "w") as f:
            json.dump({"mean": action_mean.tolist(), "std": action_std.tolist()}, f)

    # Load Gemma 4 (frozen)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if rank == 0:
        logger.info(f"Loading {args.vlm}...")
    vlm = AutoModelForImageTextToText.from_pretrained(
        args.vlm, torch_dtype=torch.bfloat16
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.vlm, padding_side="left")
    vlm.eval()
    for p in vlm.parameters():
        p.requires_grad = False
    hidden_size = vlm.config.text_config.hidden_size

    # Action head
    head = TransformerFlowActionHead(
        action_dim=ACTION_DIM, action_horizon=HORIZON,
        conditioning_dim=hidden_size, hidden_dim=768,
        num_layers=12, num_heads=8, mlp_dim=3072
    ).to(device).to(torch.bfloat16)

    if distributed:
        head = DDP(head, device_ids=[local_rank])
        head_mod = head.module
    else:
        head_mod = head

    if rank == 0:
        params = sum(p.numel() for p in head_mod.parameters())
        logger.info(f"Action head: {params/1e6:.1f}M params")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_fn(step):
        warmup = 1000
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    # Training loop
    rng = np.random.RandomState(42 + rank)
    losses = []
    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    for step in range(args.steps):
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(args.grad_accum):
            idx = rng.randint(len(pairs))
            pair = pairs[idx]

            normalized = (pair["actions"].numpy() - action_mean) / action_std
            actions = torch.tensor(normalized, dtype=torch.bfloat16).unsqueeze(0).to(device)

            img_t = pair["image"]
            if img_t.max() <= 1.0:
                img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            else:
                img_np = img_t.permute(1, 2, 0).numpy().astype(np.uint8)
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

            loss = head_mod.compute_loss(actions, cond, mask) / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(accum_loss)

        if rank == 0 and (step + 1) % 50 == 0:
            avg = np.mean(losses[-50:])
            rate = (step + 1) / (time.time() - t0)
            lr = scheduler.get_last_lr()[0]
            eff = args.grad_accum * world_size
            logger.info(f"Step {step+1}/{args.steps} | Loss: {avg:.4f} | LR: {lr:.6f} | Rate: {rate:.2f}/s | EffBatch: {eff}")

        if rank == 0 and (step + 1) % 10000 == 0:
            path = f"{args.output_dir}/checkpoint_{step+1}.pt"
            torch.save(head_mod.state_dict(), path)
            logger.info(f"Saved: {path}")

    if rank == 0:
        torch.save(head_mod.state_dict(), f"{args.output_dir}/final.pt")
        logger.info(f"Final loss: {np.mean(losses[-100:]):.4f}")
        logger.info("TRAINING COMPLETE!")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
