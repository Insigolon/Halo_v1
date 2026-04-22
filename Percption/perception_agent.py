"""
perception_agent.py
───────────────────
The Perception Agent.

Single responsibility: take a SensorFrame, run it through the
pipeline, and emit a PerceptionOutput for the State Agent.

It knows nothing about SLAM, occupancy grids, or flight planning.
It observes and packages. The State Agent reasons.

Pipeline per frame:
    SensorFrame
        │
        ├─ adapter.preprocess()       → normalized tensor
        ├─ adapter.tokenize()         → [N, 768] tokens
        ├─ encoder.encode()           → [512] embedding  ──→ Vector DB
        ├─ adapter.depth_points()     → [M, 3] XYZ metric
        ├─ adapter.obstacle_voxels()  → [K, 3] voxel coords
        └─ raw_frame buffered         → ground station stream

Output: PerceptionOutput → State Agent
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from Halo_v1.Percption.sensor_hal import SensorFrame, SensorHAL, SensorType
from Halo_v1.Percption.adapters import AdapterRegistry, SensorAdapter
from Halo_v1.Percption.encoder import UnifiedEncoder


# ─────────────────────────────────────────────
# PerceptionOutput — the contract with State Agent
# ─────────────────────────────────────────────

@dataclass
class PerceptionOutput:
    """
    One PerceptionOutput is emitted per SensorFrame processed.
    The State Agent is the sole consumer of this object.

    Fields
    ──────
    frame_id        : unique ID, matches the source SensorFrame.frame_id
    timestamp       : unix time the frame was captured
    sensor_type     : which modality produced this frame
    sensor_id       : which physical sensor (e.g. "lidar_front")
    uav_id          : which UAV

    embedding       : [512] float32 — unified vector representation.
                      State Agent stores this in the Vector DB with pose metadata.

    depth_points    : [M, 3] float32 (X, Y, Z in metres) or None.
                      State Agent feeds this into SLAM for localisation.
                      None if sensor cannot produce depth (e.g. uncalibrated RGB
                      without DepthAnything installed).

    obstacle_voxels : [K, 3] int32  voxel grid coordinates or None.
                      State Agent updates the OctoMap occupancy grid with this.
                      Voxel size used: perception_agent.voxel_size (default 0.2m)

    confidence      : float in [0, 1].
                      Quality estimate of this frame's data.
                      State Agent weights occupancy grid updates by this value.
                      Low confidence = tentative occupancy (not fully blocked).

    raw_frame       : original SensorFrame, untouched.
                      State Agent forwards this to the ground-station frame buffer
                      so the Cartography Agent can run 3D reconstruction.

    camera_params   : {"fx","fy","cx","cy"} or None.
                      Only present for RGB/ToF sensors.
                      State Agent passes this to SLAM for reprojection math.
    """

    # ── Identity ──────────────────────────────────────────────────────
    frame_id:    str
    timestamp:   float
    sensor_type: SensorType
    sensor_id:   str
    uav_id:      str

    # ── Semantic embedding → Vector DB ────────────────────────────────
    embedding: np.ndarray                    # shape [512]

    # ── Spatial data → SLAM ───────────────────────────────────────────
    depth_points: Optional[np.ndarray]       # shape [M, 3] metres, or None

    # ── Occupancy → OctoMap ───────────────────────────────────────────
    obstacle_voxels: Optional[np.ndarray]    # shape [K, 3] int32,  or None
    confidence:      float                   # 0.0 – 1.0

    # ── Ground station stream → Cartography Agent ─────────────────────
    raw_frame:     SensorFrame

    # ── SLAM projection ───────────────────────────────────────────────
    camera_params: Optional[Dict[str, float]]  # fx, fy, cx, cy

    def has_depth(self) -> bool:
        return self.depth_points is not None and len(self.depth_points) > 0

    def has_voxels(self) -> bool:
        return self.obstacle_voxels is not None and len(self.obstacle_voxels) > 0

    def summary(self) -> str:
        pts    = len(self.depth_points)    if self.has_depth()   else 0
        voxels = len(self.obstacle_voxels) if self.has_voxels()  else 0
        return (
            f"PerceptionOutput | {self.sensor_type.value} | "
            f"emb={self.embedding.shape} | "
            f"depth_pts={pts} | voxels={voxels} | conf={self.confidence:.2f}"
        )


# ─────────────────────────────────────────────
# Confidence estimator
# ─────────────────────────────────────────────

def estimate_confidence(sensor_type: SensorType,
                        raw_data: Any,
                        depth_pts: Optional[np.ndarray]) -> float:
    """
    Heuristic confidence score [0, 1] for a frame.
    
    RGB  : based on image brightness variance (blurry/dark → low)
    LiDAR: based on number of valid returns
    ToF  : based on fraction of non-zero depth pixels
    """
    try:
        if sensor_type == SensorType.RGB:
            gray = raw_data.mean(axis=2)
            var  = float(gray.var())
            return float(np.clip(var / 1000.0, 0.1, 1.0))

        elif sensor_type == SensorType.LIDAR:
            n      = raw_data.shape[0]
            target = 2048
            return float(np.clip(n / target, 0.1, 1.0))

        elif sensor_type == SensorType.TOF:
            valid_frac = float((raw_data > 0.05).mean())
            return float(np.clip(valid_frac, 0.1, 1.0))

        else:
            return 0.5   # unknown sensor — neutral confidence

    except Exception:
        return 0.5


# ─────────────────────────────────────────────
# Perception Agent
# ─────────────────────────────────────────────

class PerceptionAgent:
    """
    Orchestrates the full perception pipeline for one UAV.

    Construction
    ────────────
    agent = PerceptionAgent(
        uav_id   = "quad_01",
        registry = AdapterRegistry.default(),
        encoder  = UnifiedEncoder(),
    )

    Processing a frame
    ──────────────────
    output = agent.process(sensor_frame)
    # output is a PerceptionOutput ready for the State Agent

    Async loop
    ──────────
    agent.start(hal)           # starts background thread
    agent.subscribe(callback)  # callback receives each PerceptionOutput
    agent.stop()
    """

    def __init__(self,
                 uav_id:      str,
                 registry:    AdapterRegistry,
                 encoder:     UnifiedEncoder,
                 voxel_size:  float = 0.2):
        self.uav_id     = uav_id
        self.registry   = registry
        self.encoder    = encoder
        self.voxel_size = voxel_size

        # Async loop state
        self._running     = False
        self._thread:  Optional[threading.Thread] = None
        self._callbacks: List[Callable[[PerceptionOutput], None]] = []

        # Raw frame buffer for ground station (ring buffer, 300 frames ≈ ~10s at 30fps)
        self._frame_buffer: deque = deque(maxlen=300)

    # ─────────────────────────────────────────
    # Core pipeline
    # ─────────────────────────────────────────

    def process(self, frame: SensorFrame) -> Optional[PerceptionOutput]:
        """
        Synchronous. Process one SensorFrame and return a PerceptionOutput.
        Returns None if the frame is invalid or no adapter is registered.
        """
        if not frame.is_valid():
            print(f"[PerceptionAgent] Invalid frame from {frame.sensor_id}, skipping.")
            return None

        modality = frame.sensor_type.value

        # 1. Get adapter
        adapter = self.registry.get(modality)
        if adapter is None:
            print(
                f"[PerceptionAgent] No adapter for modality '{modality}'. "
                f"Register one via AdapterRegistry to support this sensor."
            )
            return None

        # 2. Preprocess
        tensor = adapter.preprocess(frame.raw_data)           # normalized tensor

        # 3. Tokenize
        tokens = adapter.tokenize(tensor)                     # [N, 768]

        # 4. Encode → embedding
        embedding = self.encoder.encode(modality, tokens)     # [512]

        # 5. Depth points (for SLAM)
        depth_pts = adapter.depth_points(frame.raw_data, tensor)

        # 6. Obstacle voxels (for OctoMap)
        voxels = adapter.obstacle_voxels(depth_pts, self.voxel_size)

        # 7. Confidence
        conf = estimate_confidence(frame.sensor_type, frame.raw_data, depth_pts)

        # 8. Camera params (if available in metadata)
        camera_params = None
        if frame.sensor_type in (SensorType.RGB, SensorType.TOF):
            meta = frame.metadata
            if meta.get("fx") is not None:
                camera_params = {
                    "fx": meta["fx"], "fy": meta["fy"],
                    "cx": meta.get("cx", meta.get("resolution", (0, 0))[0] / 2),
                    "cy": meta.get("cy", meta.get("resolution", (0, 0))[1] / 2),
                }

        # 9. Buffer raw frame for ground station
        self._frame_buffer.append(frame)

        output = PerceptionOutput(
            frame_id        = frame.frame_id,
            timestamp       = frame.timestamp,
            sensor_type     = frame.sensor_type,
            sensor_id       = frame.sensor_id,
            uav_id          = frame.uav_id,
            embedding       = embedding,
            depth_points    = depth_pts,
            obstacle_voxels = voxels,
            confidence      = conf,
            raw_frame       = frame,
            camera_params   = camera_params,
        )

        return output

    # ─────────────────────────────────────────
    # Async loop (for live HAL usage)
    # ─────────────────────────────────────────

    def subscribe(self, callback: Callable[[PerceptionOutput], None]) -> None:
        """
        Register a callback invoked for every PerceptionOutput.
        The State Agent calls this to receive perception data.
        Multiple subscribers allowed (e.g. logger, State Agent).
        """
        self._callbacks.append(callback)

    def start(self, hal: SensorHAL) -> None:
        """Start background thread that reads all sensors and processes frames."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target = self._loop,
            args   = (hal,),
            daemon = True,
            name   = f"PerceptionAgent-{self.uav_id}"
        )
        self._thread.start()
        print(f"[PerceptionAgent] Started for UAV: {self.uav_id}")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print(f"[PerceptionAgent] Stopped.")

    def _loop(self, hal: SensorHAL) -> None:
        while self._running:
            frames = hal.read_all()
            for sensor_id, frame in frames.items():
                output = self.process(frame)
                if output is not None:
                    for cb in self._callbacks:
                        try:
                            cb(output)
                        except Exception as e:
                            print(f"[PerceptionAgent] Callback error: {e}")

    # ─────────────────────────────────────────
    # Frame buffer access (for ground station)
    # ─────────────────────────────────────────

    def drain_frame_buffer(self) -> List[SensorFrame]:
        """
        Called by the ground station / Cartography Agent to collect
        buffered raw frames for 3D reconstruction.
        Drains and returns all buffered frames.
        """
        frames = list(self._frame_buffer)
        self._frame_buffer.clear()
        return frames

    def peek_frame_buffer(self) -> List[SensorFrame]:
        """Non-destructive read of the buffer (for monitoring)."""
        return list(self._frame_buffer)


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def build_perception_agent(uav_id: str,
                            extra_adapters: Optional[Dict] = None,
                            voxel_size: float = 0.2) -> PerceptionAgent:
    """
    Convenience factory.
    extra_adapters: {"my_sensor": MyAdapter()} to register alongside defaults.
    """
    registry = AdapterRegistry.default()
    if extra_adapters:
        for mid, adapter in extra_adapters.items():
            registry.register(mid, adapter)

    encoder = UnifiedEncoder()

    return PerceptionAgent(
        uav_id     = uav_id,
        registry   = registry,
        encoder    = encoder,
        voxel_size = voxel_size,
    )