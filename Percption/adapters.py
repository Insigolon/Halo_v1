"""
adapters.py
───────────
One SensorAdapter per modality.

Responsibility:
  1. Preprocess raw sensor data into a normalized tensor
  2. Tokenize that tensor into [N, D] tokens the shared transformer can consume
  3. (RGB only) estimate metric depth via DepthAnything V2

Every adapter implements the same three-method contract so the
UnifiedEncoder never needs to know which sensor produced the data.
Adding support for a future sensor = write one new class + register it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np


# ─────────────────────────────────────────────
# Adapter base class
# ─────────────────────────────────────────────

class SensorAdapter(ABC):
    """
    Contract every modality adapter must satisfy.
    
    The UnifiedEncoder calls:
        tensor = adapter.preprocess(raw_data)   → normalized tensor
        tokens = adapter.tokenize(tensor)        → [N, TOKEN_DIM]
    """

    TOKEN_DIM = 768     # shared across all adapters — must match backbone

    @property
    @abstractmethod
    def modality_id(self) -> str:
        """
        Unique string identifier, e.g. 'rgb', 'lidar_velodyne'.
        Used for logging and adapter registry lookup.
        """

    @abstractmethod
    def preprocess(self, raw_data: Any) -> np.ndarray:
        """
        Raw sensor data  →  normalized, consistently shaped numpy array.
        All sensor-specific normalization logic lives here.
        """

    @abstractmethod
    def tokenize(self, tensor: np.ndarray) -> np.ndarray:
        """
        Normalized tensor  →  shape [N, TOKEN_DIM].
        This is the token sequence the transformer backbone consumes.
        N can differ across modalities; the backbone handles variable length.
        """

    def depth_points(self, raw_data: Any, preprocessed: np.ndarray) -> Optional[np.ndarray]:
        """
        Optional: return metric 3D points as shape [M, 3] (X, Y, Z in metres).
        Override in adapters where depth is available natively or estimated.
        Default: returns None (no depth available for this sensor).
        Used by PerceptionAgent to fill PerceptionOutput.depth_points.
        """
        return None

    def obstacle_voxels(self, depth_pts: Optional[np.ndarray],
                        voxel_size: float = 0.2) -> Optional[np.ndarray]:
        """
        Convert metric 3D points into voxel coordinates [M, 3].
        Default implementation works for any sensor that provides depth_points.
        Specific adapters can override for efficiency.
        """
        if depth_pts is None:
            return None
        voxels = np.floor(depth_pts / voxel_size).astype(np.int32)
        return np.unique(voxels, axis=0)   # deduplicate


# ─────────────────────────────────────────────
# RGB Camera Adapter
# ─────────────────────────────────────────────

class RGBAdapter(SensorAdapter):
    """
    Handles RGB frames.
    Depth estimation via DepthAnything V2 (small, runs on-device).
    Falls back gracefully if DepthAnything is not installed.
    """

    TARGET_SIZE = (224, 224)    # encoder input size

    def __init__(self,
                 depth_model_size: str = "small",     # "small" | "base" | "large"
                 camera_intrinsics: Optional[Dict] = None):
        """
        camera_intrinsics: {"fx": float, "fy": float,
                             "cx": float, "cy": float}
        Used to unproject depth map to 3D point cloud.
        If None, depth_points returns 2.5D (u, v, d) instead of metric XYZ.
        """
        self._intrinsics     = camera_intrinsics
        self._depth_size     = depth_model_size
        self._depth_model    = None                 # lazy-loaded
        self._depth_transform= None

    @property
    def modality_id(self) -> str:
        return "rgb"

    def preprocess(self, raw_data: np.ndarray) -> np.ndarray:
        """
        raw_data: [H, W, 3] uint8 RGB
        output:   [3, 224, 224] float32, ImageNet-normalised
        """
        import cv2
        img = cv2.resize(raw_data, self.TARGET_SIZE,
                         interpolation=cv2.INTER_LINEAR).astype(np.float32)
        # ImageNet normalisation
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img / 255.0 - mean) / std
        return img.transpose(2, 0, 1)   # [3, H, W]

    def tokenize(self, tensor: np.ndarray) -> np.ndarray:
        """
        Patch-embed [3, 224, 224] into [N, 768] tokens.
        Uses 16×16 patches → N = (224/16)^2 = 196 tokens.
        In production the ViT backbone does this; here we simulate it.
        """
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(tensor).unsqueeze(0)   # [1, 3, 224, 224]
        # Unfold into 16×16 patches
        patches = t.unfold(2, 16, 16).unfold(3, 16, 16)   # [1,3,14,14,16,16]
        patches = patches.contiguous().view(1, 3, -1, 16*16)
        patches = patches.permute(0, 2, 1, 3).reshape(-1, 3*16*16)  # [196, 768]
        return patches.numpy()

    # ── Depth estimation ──────────────────────

    def _load_depth_model(self):
        """Lazy-load DepthAnything V2. No-ops if not installed."""
        if self._depth_model is not None:
            return
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
            import torch
            configs = {
                "small": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192, 384]},
                "base":  {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384, 768]},
                "large": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024,1024]},
            }
            cfg   = configs[self._depth_size]
            model = DepthAnythingV2(**cfg)
            # weights path — set via env var DEPTH_ANYTHING_WEIGHTS
            import os
            ckpt = os.getenv("DEPTH_ANYTHING_WEIGHTS",
                             f"checkpoints/depth_anything_v2_{self._depth_size}.pth")
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            model.eval()
            self._depth_model = model
        except ImportError:
            print("[RGBAdapter] DepthAnythingV2 not installed — depth_points unavailable.")
        except FileNotFoundError:
            print("[RGBAdapter] Depth model weights not found — depth_points unavailable.")

    def depth_points(self, raw_data: np.ndarray,
                     preprocessed: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth from RGB, unproject to 3D point cloud.
        Returns [M, 3] float32 (X, Y, Z) if intrinsics provided,
        else [M, 3] (u, v, depth) in image space.
        """
        self._load_depth_model()
        if self._depth_model is None:
            return None

        import torch
        import cv2

        h, w = raw_data.shape[:2]
        with torch.no_grad():
            depth = self._depth_model.infer_image(raw_data)   # [H, W] float32

        # Unproject if intrinsics are known
        if self._intrinsics:
            fx = self._intrinsics["fx"]
            fy = self._intrinsics["fy"]
            cx = self._intrinsics.get("cx", w / 2)
            cy = self._intrinsics.get("cy", h / 2)
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            Z = depth
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        else:
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            pts  = np.stack([u.ravel(), v.ravel(), depth.ravel()], axis=-1)

        # Filter invalid / very far points
        valid = pts[:, 2] > 0.1
        return pts[valid].astype(np.float32)


# ─────────────────────────────────────────────
# LiDAR Adapter
# ─────────────────────────────────────────────

class LiDARAdapter(SensorAdapter):
    """
    Handles [N, 4] float32 point clouds (x, y, z, intensity).
    Uses a PointNet-style mini-encoder for tokenization.
    """

    NUM_POINTS = 2048       # fixed sample size for the encoder
    NUM_TOKENS = 64         # number of output tokens

    @property
    def modality_id(self) -> str:
        return "lidar"

    def preprocess(self, raw_data: np.ndarray) -> np.ndarray:
        """
        raw_data: [N, 4] float32  (x, y, z, intensity)
        output:   [NUM_POINTS, 4] float32, normalised
        """
        pts = raw_data.astype(np.float32)

        # Sample or pad to fixed size
        n = pts.shape[0]
        if n >= self.NUM_POINTS:
            idx = np.random.choice(n, self.NUM_POINTS, replace=False)
        else:
            idx = np.concatenate([
                np.arange(n),
                np.random.choice(n, self.NUM_POINTS - n, replace=True)
            ])
        pts = pts[idx]

        # Centre the point cloud
        pts[:, :3] -= pts[:, :3].mean(axis=0)

        # Normalise scale to unit sphere
        scale = np.max(np.linalg.norm(pts[:, :3], axis=1))
        if scale > 0:
            pts[:, :3] /= scale

        # Normalise intensity to [0, 1]
        i_max = pts[:, 3].max()
        if i_max > 0:
            pts[:, 3] /= i_max

        return pts   # [NUM_POINTS, 4]

    def tokenize(self, tensor: np.ndarray) -> np.ndarray:
        """
        PointNet-style max-pool over local groups to get [NUM_TOKENS, 768].
        """
        import torch

        pts = torch.from_numpy(tensor)   # [NUM_POINTS, 4]

        # Group NUM_POINTS into NUM_TOKENS groups
        group_size = self.NUM_POINTS // self.NUM_TOKENS
        pts = pts[:self.NUM_TOKENS * group_size]
        groups = pts.view(self.NUM_TOKENS, group_size, 4)

        # Project each point to 768-d via a small linear stack (simulated here)
        # In training this is a learned MLP; we initialise with random weights
        # which will be replaced once training runs.
        np.random.seed(42)  # deterministic init for reproducibility
        W = np.random.randn(4, self.TOKEN_DIM).astype(np.float32) * 0.02
        projected = groups.numpy() @ W          # [NUM_TOKENS, group_size, 768]

        # Max pool over group members → one token per group
        tokens = projected.max(axis=1)          # [NUM_TOKENS, 768]
        return tokens

    def depth_points(self, raw_data: np.ndarray,
                     preprocessed: np.ndarray) -> np.ndarray:
        """LiDAR natively provides 3D points — just return XYZ."""
        return raw_data[:, :3].astype(np.float32)


# ─────────────────────────────────────────────
# ToF Adapter
# ─────────────────────────────────────────────

class ToFAdapter(SensorAdapter):
    """
    Handles [H, W] float32 depth maps in metres.
    """

    TARGET_SIZE = (64, 64)   # ToF is typically low-res; keep small

    def __init__(self, camera_intrinsics: Optional[Dict] = None):
        self._intrinsics = camera_intrinsics

    @property
    def modality_id(self) -> str:
        return "tof"

    def preprocess(self, raw_data: np.ndarray) -> np.ndarray:
        """
        raw_data: [H, W] float32 depth in metres
        output:   [1, 64, 64] float32, normalised to [0, 1]
        """
        import cv2
        h, w = self.TARGET_SIZE
        depth = cv2.resize(raw_data, (w, h), interpolation=cv2.INTER_NEAREST)
        depth = depth.astype(np.float32)
        d_max = depth.max()
        if d_max > 0:
            depth /= d_max
        return depth[np.newaxis]   # [1, 64, 64]

    def tokenize(self, tensor: np.ndarray) -> np.ndarray:
        """
        Patch-embed [1, 64, 64] into [N, 768] tokens.
        Uses 8×8 patches → N = (64/8)^2 = 64 tokens.
        """
        import torch
        t = torch.from_numpy(tensor).unsqueeze(0)   # [1, 1, 64, 64]
        patches = t.unfold(2, 8, 8).unfold(3, 8, 8)
        patches = patches.contiguous().view(1, 1, -1, 8*8)
        patches = patches.permute(0, 2, 1, 3).reshape(-1, 1*8*8)   # [64, 64]
        # Pad to TOKEN_DIM=768
        padded  = np.zeros((patches.shape[0], self.TOKEN_DIM), dtype=np.float32)
        padded[:, :patches.shape[1]] = patches.numpy()
        return padded   # [64, 768]

    def depth_points(self, raw_data: np.ndarray,
                     preprocessed: np.ndarray) -> Optional[np.ndarray]:
        """Unproject ToF depth map to 3D points."""
        h, w = raw_data.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        if self._intrinsics:
            fx = self._intrinsics["fx"]
            fy = self._intrinsics["fy"]
            cx = self._intrinsics.get("cx", w / 2)
            cy = self._intrinsics.get("cy", h / 2)
            Z  = raw_data
            X  = (u - cx) * Z / fx
            Y  = (v - cy) * Z / fy
        else:
            X, Y, Z = u.astype(np.float32), v.astype(np.float32), raw_data

        pts   = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        valid = pts[:, 2] > 0.05
        return pts[valid].astype(np.float32)


# ─────────────────────────────────────────────
# Adapter Registry — the plugin mechanism
# ─────────────────────────────────────────────

class AdapterRegistry:
    """
    Maps modality_id strings to SensorAdapter instances.
    To support a future sensor:
        1. Subclass SensorAdapter
        2. Call AdapterRegistry.register("my_sensor", MyAdapter())
        3. Done — the Perception Agent will use it automatically.
    """

    def __init__(self):
        self._adapters: Dict[str, SensorAdapter] = {}

    def register(self, modality_id: str, adapter: SensorAdapter) -> None:
        self._adapters[modality_id] = adapter
        print(f"[AdapterRegistry] Registered adapter: {modality_id}")

    def get(self, modality_id: str) -> Optional[SensorAdapter]:
        return self._adapters.get(modality_id)

    def get_or_raise(self, modality_id: str) -> SensorAdapter:
        adapter = self.get(modality_id)
        if adapter is None:
            raise KeyError(
                f"No adapter for modality '{modality_id}'. "
                f"Registered: {list(self._adapters.keys())}"
            )
        return adapter

    @classmethod
    def default(cls) -> "AdapterRegistry":
        """Returns a registry pre-loaded with the built-in adapters."""
        reg = cls()
        reg.register("rgb",   RGBAdapter())
        reg.register("lidar", LiDARAdapter())
        reg.register("tof",   ToFAdapter())
        return reg