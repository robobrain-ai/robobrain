#!/usr/bin/env python3
"""Standalone PyTorch inference server for pi0.5 LIBERO policies.

Serves a converted PyTorch checkpoint over WebSocket, compatible with the
openpi eval client (examples/libero/main.py).  Runs on any CUDA GPU (A100,
A10, consumer) or CPU.

Usage:
    python serve_pytorch.py \
        --checkpoint-dir /tmp/pi05_pytorch_5k \
        --port 8000 \
        --device cuda

The checkpoint directory must contain:
    model.safetensors   - converted PyTorch weights
    config.json         - model config (action_dim, action_horizon, etc.)
    assets/<asset_id>/norm_stats.json  - normalization stats

If norm_stats.json is not inside the checkpoint dir, pass --norm-stats explicitly.

Protocol:
    1. Server accepts a WebSocket connection
    2. Server sends metadata dict (msgpack)
    3. Client sends observation dict (msgpack) in a loop
    4. Server replies with action dict (msgpack) per observation
"""

from __future__ import annotations

import argparse
import asyncio
import http
import json
import logging
import pathlib
import time
import traceback
from dataclasses import dataclass

import msgpack
import numpy as np
import sentencepiece
import torch
import torch.nn.functional as F
import websockets
import websockets.asyncio.server as ws_server
import websockets.frames

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (matching openpi)
# ---------------------------------------------------------------------------

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
IMAGE_RESOLUTION = (224, 224)
PALIGEMMA_VOCAB_SIZE = 257_152


# ---------------------------------------------------------------------------
# Norm stats
# ---------------------------------------------------------------------------

@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None


def load_norm_stats(path: str) -> dict[str, NormStats]:
    """Load normalization stats from a JSON file (openpi format)."""
    with open(path) as f:
        raw = json.load(f)
    data = raw.get("norm_stats", raw)
    result = {}
    for key, stats in data.items():
        result[key] = NormStats(
            mean=np.array(stats["mean"], dtype=np.float32),
            std=np.array(stats["std"], dtype=np.float32),
            q01=np.array(stats["q01"], dtype=np.float32) if stats.get("q01") else None,
            q99=np.array(stats["q99"], dtype=np.float32) if stats.get("q99") else None,
        )
    return result


# ---------------------------------------------------------------------------
# Tokenizer (PaliGemma sentencepiece)
# ---------------------------------------------------------------------------

class PaliGemmaTokenizer:
    """Tokenize language prompts for the pi0.5 model."""

    def __init__(self, model_path: str, max_len: int = 200):
        self._max_len = max_len
        with open(model_path, "rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # pi0.5 format: state discretized into language tokens
            discretized = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
            state_str = " ".join(map(str, discretized))
            full_prompt = f"Task: {cleaned}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            tokens = self._tokenizer.encode(cleaned, add_bos=True) + self._tokenizer.encode("\n")

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if tokens_len > self._max_len:
                logger.warning(
                    "Token length (%d) exceeds max (%d), truncating.", tokens_len, self._max_len
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens, dtype=np.int32), np.asarray(mask, dtype=bool)


# ---------------------------------------------------------------------------
# Image tools
# ---------------------------------------------------------------------------

def resize_with_pad(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize image with padding to maintain aspect ratio. Input: (H, W, 3) uint8 or float."""
    h, w = image.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Use torch for interpolation
    img_t = torch.from_numpy(image).float()
    if img_t.ndim == 3:
        img_t = img_t.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    resized = F.interpolate(img_t, size=(new_h, new_w), mode="bilinear", align_corners=False)
    resized = resized.squeeze(0).permute(1, 2, 0).numpy()  # (new_h, new_w, C)

    # Pad - always return same dtype as input
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    if image.dtype == np.uint8:
        resized = resized.clip(0, 255).astype(np.uint8)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    else:
        canvas = np.zeros((target_h, target_w, 3), dtype=np.float32)
    canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
    return canvas


# ---------------------------------------------------------------------------
# Msgpack numpy (matching openpi_client.msgpack_numpy)
# ---------------------------------------------------------------------------

def _pack_array(obj):
    if isinstance(obj, np.ndarray) and obj.dtype.kind not in ("V", "O", "c"):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def msgpack_pack(obj):
    return msgpack.packb(obj, default=_pack_array)


def msgpack_unpack(data):
    return msgpack.unpackb(data, object_hook=_unpack_array)


# ---------------------------------------------------------------------------
# Input transforms (matching openpi pipeline for LIBERO + pi0.5)
# ---------------------------------------------------------------------------

def _parse_image(image: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 (H, W, C)."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).clip(0, 255).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


def libero_inputs(data: dict) -> dict:
    """Convert LIBERO observation dict to model input format (LiberoInputs transform)."""
    base_image = _parse_image(data["observation/image"])
    wrist_image = _parse_image(data["observation/wrist_image"])

    inputs = {
        "state": np.asarray(data["observation/state"], dtype=np.float32),
        "image": {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": wrist_image,
            "right_wrist_0_rgb": np.zeros_like(base_image),
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.False_,  # pi0.5 masks the absent right wrist
        },
    }
    if "prompt" in data:
        inputs["prompt"] = data["prompt"]
    return inputs


def normalize_quantile(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """Quantile normalization (used for pi0.5)."""
    q01 = stats.q01[..., : x.shape[-1]]
    q99 = stats.q99[..., : x.shape[-1]]
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def unnormalize_quantile(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """Inverse quantile normalization."""
    q01, q99 = stats.q01, stats.q99
    dim = q01.shape[-1]
    if dim < x.shape[-1]:
        return np.concatenate(
            [(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]],
            axis=-1,
        )
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def resize_images(data: dict, height: int = 224, width: int = 224) -> dict:
    """Resize all images in the data dict."""
    data["image"] = {k: resize_with_pad(v, height, width) for k, v in data["image"].items()}
    return data


def pad_to_dim(arr: np.ndarray, target_dim: int, axis: int = -1) -> np.ndarray:
    """Zero-pad array along axis to target dimension."""
    current = arr.shape[axis]
    if current >= target_dim:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis if axis >= 0 else arr.ndim + axis] = (0, target_dim - current)
    return np.pad(arr, pad_width, mode="constant", constant_values=0)


def apply_input_transforms(
    data: dict,
    norm_stats: dict[str, NormStats],
    tokenizer: PaliGemmaTokenizer,
    action_dim: int,
    discrete_state_input: bool = True,
) -> dict:
    """Full input transform pipeline: LiberoInputs -> Normalize -> TokenizePrompt -> PadStates."""
    # 1. LiberoInputs
    data = libero_inputs(data)

    # 2. Normalize state (quantile for pi0.5)
    if "state" in norm_stats:
        data["state"] = normalize_quantile(data["state"], norm_stats["state"])

    # 3. Resize images
    data = resize_images(data)

    # 4. Tokenize prompt
    prompt = data.pop("prompt", None)
    if prompt is None:
        raise ValueError("Prompt is required")
    if not isinstance(prompt, str):
        prompt = str(prompt) if not hasattr(prompt, "item") else prompt.item()

    state_for_tokenizer = data["state"] if discrete_state_input else None
    tokens, token_mask = tokenizer.tokenize(prompt, state_for_tokenizer)
    data["tokenized_prompt"] = tokens
    data["tokenized_prompt_mask"] = token_mask

    # 5. Pad state to action_dim
    data["state"] = pad_to_dim(data["state"], action_dim, axis=-1)

    return data


def apply_output_transforms(
    actions: np.ndarray,
    state: np.ndarray,
    norm_stats: dict[str, NormStats],
) -> dict:
    """Output pipeline: Unnormalize -> LiberoOutputs (trim to 7 dims)."""
    output = {"actions": actions, "state": state}

    # Unnormalize actions (quantile for pi0.5)
    if "actions" in norm_stats:
        output["actions"] = unnormalize_quantile(output["actions"], norm_stats["actions"])

    # LiberoOutputs: trim to first 7 action dims (LIBERO has 7-DOF actions)
    output["actions"] = output["actions"][:, :7]

    return output


# ---------------------------------------------------------------------------
# Convert dict to model input tensors
# ---------------------------------------------------------------------------

def dict_to_tensors(data: dict, device: torch.device) -> dict:
    """Convert numpy data dict to batched PyTorch tensors on device."""
    result = {}
    for k, v in data.items():
        if isinstance(v, dict):
            result[k] = {sk: _to_tensor(sv, device) for sk, sv in v.items()}
        else:
            result[k] = _to_tensor(v, device)
    return result


def _to_tensor(v, device: torch.device) -> torch.Tensor:
    arr = np.asarray(v)
    t = torch.from_numpy(arr).to(device)
    return t.unsqueeze(0)  # add batch dim


# ---------------------------------------------------------------------------
# Observation dataclass (mirrors openpi model.Observation)
# ---------------------------------------------------------------------------

class Observation:
    """Lightweight Observation for PyTorch inference."""

    def __init__(
        self,
        images: dict[str, torch.Tensor],
        image_masks: dict[str, torch.Tensor],
        state: torch.Tensor,
        tokenized_prompt: torch.Tensor | None = None,
        tokenized_prompt_mask: torch.Tensor | None = None,
        token_ar_mask: torch.Tensor | None = None,
        token_loss_mask: torch.Tensor | None = None,
    ):
        self.images = images
        self.image_masks = image_masks
        self.state = state
        self.tokenized_prompt = tokenized_prompt
        self.tokenized_prompt_mask = tokenized_prompt_mask
        self.token_ar_mask = token_ar_mask
        self.token_loss_mask = token_loss_mask

    @classmethod
    def from_dict(cls, data: dict[str, torch.Tensor | dict]) -> "Observation":
        """Build from a dict of tensors (matching openpi model.Observation.from_dict)."""
        images = data["image"]
        # Convert uint8 images to [-1, 1] float32
        processed_images = {}
        for key, img in images.items():
            if img.dtype == torch.uint8:
                img = img.to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
            processed_images[key] = img

        return cls(
            images=processed_images,
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_dir: str, device: str = "cuda", compile_mode: str | None = None):
    """Load a PI0Pytorch model from a checkpoint directory.

    Returns (model, config_dict).
    """
    import safetensors.torch

    # Lazy import to allow running on machines without the full openpi install.
    # We import PI0Pytorch and its config from the openpi package.
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    ckpt_dir = pathlib.Path(checkpoint_dir)
    with open(ckpt_dir / "config.json") as f:
        cfg = json.load(f)

    model_config = Pi0Config(
        pi05=True,
        action_dim=cfg.get("action_dim", 32),
        action_horizon=cfg.get("action_horizon", 10),
        paligemma_variant=cfg.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=cfg.get("action_expert_variant", "gemma_300m"),
        dtype=cfg.get("precision", "bfloat16"),
        discrete_state_input=True,
        pytorch_compile_mode=compile_mode,
    )

    logger.info("Creating PI0Pytorch model (pi0.5, action_dim=%d, horizon=%d)...",
                model_config.action_dim, model_config.action_horizon)

    model = PI0Pytorch(config=model_config)

    weight_path = str(ckpt_dir / "model.safetensors")
    logger.info("Loading weights from %s ...", weight_path)
    safetensors.torch.load_model(model, weight_path)

    # Convert selected params to bfloat16 (matching openpi convention)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    model = model.to(device)
    model.eval()
    logger.info("Model loaded on %s", device)

    return model, model_config


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------

class PyTorchPolicy:
    """Wraps PI0Pytorch model with transforms for end-to-end inference."""

    def __init__(
        self,
        model,
        model_config,
        norm_stats: dict[str, NormStats],
        tokenizer: PaliGemmaTokenizer,
        device: str = "cuda",
    ):
        self.model = model
        self.model_config = model_config
        self.norm_stats = norm_stats
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.action_dim = model_config.action_dim
        self.action_horizon = model_config.action_horizon

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        """Run full inference pipeline: transforms -> model -> untransforms.

        Args:
            obs: Raw observation dict from eval client (LIBERO format).

        Returns:
            Dict with 'actions' np.ndarray of shape (action_horizon, 7).
        """
        start = time.monotonic()

        # Apply input transforms (numpy)
        data = apply_input_transforms(
            obs,
            self.norm_stats,
            self.tokenizer,
            self.action_dim,
            discrete_state_input=True,
        )

        # Save state before converting to tensors (for output transforms)
        state_np = data["state"].copy()

        # Convert to batched tensors
        tensors = dict_to_tensors(data, self.device)

        # Build Observation
        observation = Observation.from_dict(tensors)

        # Run model inference
        actions = self.model.sample_actions(
            self.device,
            observation,
            num_steps=10,
        )

        # Convert to numpy, remove batch dim
        actions_np = actions[0].detach().cpu().float().numpy()
        state_out = state_np

        model_time = time.monotonic() - start

        # Apply output transforms
        output = apply_output_transforms(actions_np, state_out, self.norm_stats)

        output["policy_timing"] = {"infer_ms": model_time * 1000}
        return output

    @property
    def metadata(self) -> dict:
        return {
            "model": "pi0.5",
            "backend": "pytorch",
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
        }


# ---------------------------------------------------------------------------
# WebSocket server (matching openpi protocol exactly)
# ---------------------------------------------------------------------------

class WebSocketPolicyServer:
    """Serves a policy over WebSocket, compatible with openpi_client."""

    def __init__(self, policy: PyTorchPolicy, host: str = "0.0.0.0", port: int = 8000):
        self.policy = policy
        self.host = host
        self.port = port

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        async with ws_server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=self._health_check,
        ) as server:
            logger.info("Serving on ws://%s:%d", self.host, self.port)
            await server.serve_forever()

    async def _handler(self, websocket):
        logger.info("Connection from %s", websocket.remote_address)

        # Send metadata as first message
        await websocket.send(msgpack_pack(self.policy.metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()

                # Receive observation
                raw = await websocket.recv()
                obs = msgpack_unpack(raw)

                # Infer
                infer_start = time.monotonic()
                result = self.policy.infer(obs)
                infer_time = time.monotonic() - infer_start

                result["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    result["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(msgpack_pack(result))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                tb = traceback.format_exc()
                logger.error("Error during inference:\n%s", tb)
                await websocket.send(tb)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback in previous frame.",
                )
                raise

    @staticmethod
    def _health_check(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def find_norm_stats(checkpoint_dir: str) -> str | None:
    """Search for norm_stats.json inside the checkpoint directory tree."""
    ckpt = pathlib.Path(checkpoint_dir)
    for p in ckpt.rglob("norm_stats.json"):
        return str(p)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Standalone PyTorch inference server for pi0.5 LIBERO policies"
    )
    parser.add_argument(
        "--checkpoint-dir", required=True,
        help="Directory containing model.safetensors and config.json",
    )
    parser.add_argument(
        "--norm-stats", default=None,
        help="Path to norm_stats.json. Auto-detected inside checkpoint dir if not provided.",
    )
    parser.add_argument(
        "--tokenizer-model", default=None,
        help="Path to paligemma_tokenizer.model. Defaults to ~/.cache/openpi/big_vision/paligemma_tokenizer.model",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--device", default=None,
        help="PyTorch device (cuda, cuda:0, cpu). Auto-detected if not set.",
    )
    parser.add_argument(
        "--compile", default=None, choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="torch.compile mode for the model. None = no compile.",
    )
    parser.add_argument("--num-steps", type=int, default=10, help="Number of denoising steps")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Auto-detect device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load model
    model, model_config = load_model(args.checkpoint_dir, device=device, compile_mode=args.compile)

    # Load norm stats
    norm_stats_path = args.norm_stats or find_norm_stats(args.checkpoint_dir)
    if norm_stats_path is None:
        logger.error(
            "norm_stats.json not found in checkpoint dir. "
            "Copy it from: checkpoints/<name>/assets/<asset_id>/norm_stats.json "
            "or pass --norm-stats explicitly."
        )
        raise FileNotFoundError("norm_stats.json not found")
    logger.info("Loading norm stats from %s", norm_stats_path)
    norm_stats = load_norm_stats(norm_stats_path)

    # Load tokenizer
    tokenizer_path = args.tokenizer_model
    if tokenizer_path is None:
        default_paths = [
            pathlib.Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model",
            pathlib.Path("/tmp/paligemma_tokenizer.model"),
        ]
        for p in default_paths:
            if p.exists():
                tokenizer_path = str(p)
                break
    if tokenizer_path is None:
        logger.error(
            "paligemma_tokenizer.model not found. Download from "
            "gs://big_vision/paligemma_tokenizer.model or pass --tokenizer-model."
        )
        raise FileNotFoundError("paligemma_tokenizer.model not found")
    logger.info("Loading tokenizer from %s", tokenizer_path)
    tokenizer = PaliGemmaTokenizer(tokenizer_path, max_len=model_config.max_token_len)

    # Build policy
    policy = PyTorchPolicy(model, model_config, norm_stats, tokenizer, device=device)

    # Warm up with a dummy inference
    logger.info("Running warm-up inference...")
    dummy_obs = {
        "observation/image": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.randn(8).astype(np.float32),
        "prompt": "pick up the red cube",
    }
    warmup_start = time.monotonic()
    _ = policy.infer(dummy_obs)
    logger.info("Warm-up done in %.1f ms", (time.monotonic() - warmup_start) * 1000)

    # Start server
    server = WebSocketPolicyServer(policy, host=args.host, port=args.port)
    logger.info("Starting WebSocket server on ws://%s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
