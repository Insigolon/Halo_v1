"""
example_usage.py
────────────────
Shows how the Sensor HAL, adapters, encoder, and Perception Agent
wire together. Run this to verify the pipeline end-to-end with
simulated sensor data (no real hardware needed).
"""

import numpy as np

from Halo_v1.Percption.sensor_hal import SensorHAL, RGBCameraDriver, LiDARDriver, ToFDriver, SensorFrame, SensorType
from Halo_v1.Percption.adapters import AdapterRegistry, SensorAdapter
from Halo_v1.Percption.encoder import UnifiedEncoder
from Halo_v1.Percption.perception_agent import PerceptionAgent, PerceptionOutput, build_perception_agent


# ─────────────────────────────────────────────
# 1. Build the HAL
# ─────────────────────────────────────────────

hal = SensorHAL(uav_id="quad_01")

# Register sensors (use simulation mode — no real hardware)
lidar = LiDARDriver("lidar_front", "quad_01")
hal.register(lidar)

# ─────────────────────────────────────────────
# 2. Build the Perception Agent
# ─────────────────────────────────────────────

agent = build_perception_agent(uav_id="quad_01")

# ─────────────────────────────────────────────
# 3. State Agent callback stub
#    (replace with real State Agent later)
# ─────────────────────────────────────────────

def state_agent_receive(output: PerceptionOutput):
    print("\n" + "="*60)
    print(f"  State Agent received:")
    print(f"  {output.summary()}")
    print(f"  → Store embedding[512] in Vector DB")
    if output.has_depth():
        print(f"  → Feed {len(output.depth_points)} depth points into SLAM")
    if output.has_voxels():
        print(f"  → Update OctoMap with {len(output.obstacle_voxels)} voxels "
              f"(conf={output.confidence:.2f})")
    print(f"  → Forward raw frame to ground station buffer")
    if output.camera_params:
        print(f"  → Camera params available for SLAM: {output.camera_params}")
    print("="*60)

agent.subscribe(state_agent_receive)

# ─────────────────────────────────────────────
# 4. Simulate sensor data and run the pipeline
# ─────────────────────────────────────────────

print("\n── Simulating RGB frame ──")
rgb_frame = SensorFrame(
    sensor_type = SensorType.RGB,
    sensor_id   = "rgb_gimbal",
    uav_id      = "quad_01",
    raw_data    = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),
    metadata    = {
        "resolution": (1280, 720),
        "fps": 30,
        "fx": 800.0, "fy": 800.0, "cx": 640.0, "cy": 360.0,
    }
)
out = agent.process(rgb_frame)
state_agent_receive(out)

print("\n── Simulating LiDAR frame ──")
# Inject a simulated point cloud
n_pts = 5000
fake_cloud = np.random.randn(n_pts, 4).astype(np.float32)
fake_cloud[:, :3] *= 20     # spread 20m
fake_cloud[:,  3]  = np.abs(fake_cloud[:, 3])   # intensity positive
lidar_frame = SensorFrame(
    sensor_type = SensorType.LIDAR,
    sensor_id   = "lidar_front",
    uav_id      = "quad_01",
    raw_data    = fake_cloud,
    metadata    = {"max_range": 100.0}
)
out = agent.process(lidar_frame)
state_agent_receive(out)

print("\n── Simulating ToF frame ──")
tof_frame = SensorFrame(
    sensor_type = SensorType.TOF,
    sensor_id   = "tof_bottom",
    uav_id      = "quad_01",
    raw_data    = np.random.uniform(0.5, 5.0, (480, 640)).astype(np.float32),
    metadata    = {"resolution": (640, 480), "max_depth": 10.0}
)
out = agent.process(tof_frame)
state_agent_receive(out)

# ─────────────────────────────────────────────
# 5. Adding a future sensor (plugin demo)
# ─────────────────────────────────────────────

print("\n── Adding a future sensor (Thermal IR) ──")

class ThermalIRAdapter(SensorAdapter):
    """
    Example future sensor adapter.
    Thermal IR camera: [H, W] float32 in degrees Celsius.
    """

    @property
    def modality_id(self) -> str:
        return "thermal_ir"

    def preprocess(self, raw_data: np.ndarray) -> np.ndarray:
        # Normalise temp range [0°C, 100°C] → [0, 1]
        tensor = np.clip(raw_data, 0, 100) / 100.0
        import cv2
        tensor = cv2.resize(tensor, (64, 64)).astype(np.float32)
        return tensor[np.newaxis]   # [1, 64, 64]

    def tokenize(self, tensor: np.ndarray) -> np.ndarray:
        # Re-use patch embedding: 8x8 patches on 64x64 → 64 tokens
        import torch
        t       = torch.from_numpy(tensor).unsqueeze(0)
        patches = t.unfold(2, 8, 8).unfold(3, 8, 8)
        patches = patches.contiguous().view(1, 1, -1, 64)
        patches = patches.permute(0, 2, 1, 3).reshape(-1, 64).numpy()
        padded  = np.zeros((patches.shape[0], self.TOKEN_DIM), dtype=np.float32)
        padded[:, :64] = patches
        return padded   # [64, 768]

# Register the new adapter — zero changes to anything else
agent.registry.register("thermal_ir", ThermalIRAdapter())

thermal_frame = SensorFrame(
    sensor_type = SensorType.UNKNOWN,   # new sensor type
    sensor_id   = "thermal_front",
    uav_id      = "quad_01",
    raw_data    = np.random.uniform(20, 80, (480, 640)).astype(np.float32),
)
# Override modality id for lookup
thermal_frame.sensor_type.value  # this is "unknown"
# We process it using the adapter directly
adapter = agent.registry.get("thermal_ir")
tensor  = adapter.preprocess(thermal_frame.raw_data)
tokens  = adapter.tokenize(tensor)
emb     = agent.encoder.encode("thermal_ir", tokens)
print(f"  Thermal IR embedding shape: {emb.shape}, norm: {np.linalg.norm(emb):.4f}")
print("  → Plugin sensor works without modifying any existing code ✓")

# ─────────────────────────────────────────────
# 6. Show ground station frame buffer
# ─────────────────────────────────────────────

buffered = agent.drain_frame_buffer()
print(f"\n── Ground station frame buffer: {len(buffered)} frames queued ──")
print("  → Cartography Agent would consume these for 3D reconstruction")