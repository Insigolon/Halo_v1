"""
sensor_hal.py
─────────────
Hardware Abstraction Layer (HAL) for all UAV sensors.

Responsibility: get raw data OUT of physical/simulated sensors
in a normalized, typed container (SensorFrame). Everything above
this layer is sensor-agnostic.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Type


# ─────────────────────────────────────────────
# Sensor Types
# ─────────────────────────────────────────────

class SensorType(str, Enum):
    RGB        = "rgb"
    LIDAR      = "lidar"
    TOF        = "tof"
    IMU        = "imu"
    UNKNOWN    = "unknown"   # future sensors register here first


# ─────────────────────────────────────────────
# SensorFrame  — the single contract object
# that the HAL produces and everything above
# it consumes.
# ─────────────────────────────────────────────

@dataclass
class SensorFrame:
    """
    Normalized container for one frame/reading from any sensor.
    The HAL fills this; adapters and the Perception Agent read it.
    
    raw_data : the original sensor output, untouched.
               Type varies by sensor:
                 RGB   → np.ndarray  shape [H, W, 3]  uint8
                 LiDAR → np.ndarray  shape [N, 4]     float32  (x,y,z,intensity)
                 ToF   → np.ndarray  shape [H, W]     float32  (metres)
                 IMU   → dict        {accel, gyro, mag} each np.ndarray [3]
                 Future → Any
    metadata : sensor-specific extra info (resolution, FoV, etc.)
    """
    frame_id:    str        = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:   float      = field(default_factory=time.time)
    sensor_type: SensorType = SensorType.UNKNOWN
    sensor_id:   str        = ""          # e.g. "front_lidar", "rgb_gimbal"
    uav_id:      str        = ""
    raw_data:    Any        = None
    metadata:    Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        return self.raw_data is not None


# ─────────────────────────────────────────────
# Abstract HAL driver
# ─────────────────────────────────────────────

class SensorDriver(ABC):
    """
    One SensorDriver per physical sensor model.
    Subclass this to support a new sensor.
    """

    def __init__(self, sensor_id: str, uav_id: str):
        self.sensor_id  = sensor_id
        self.uav_id     = uav_id
        self._connected = False

    # ── Lifecycle ─────────────────────────────

    @abstractmethod
    def connect(self) -> bool:
        """
        Open connection to the sensor (USB, ROS topic, SDK, etc.)
        Return True on success.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Release all resources."""

    # ── Data ──────────────────────────────────

    @abstractmethod
    def read_frame(self) -> SensorFrame:
        """
        Block until one frame is available, then return it.
        Always returns a SensorFrame; check .is_valid() on the result.
        """

    @property
    @abstractmethod
    def sensor_type(self) -> SensorType:
        """Which modality this driver produces."""

    # ── Helpers ───────────────────────────────

    def _make_frame(self, raw_data: Any, metadata: Optional[Dict] = None) -> SensorFrame:
        """Convenience builder used by subclasses."""
        return SensorFrame(
            sensor_type = self.sensor_type,
            sensor_id   = self.sensor_id,
            uav_id      = self.uav_id,
            raw_data    = raw_data,
            metadata    = metadata or {},
        )

    @property
    def is_connected(self) -> bool:
        return self._connected


# ─────────────────────────────────────────────
# Concrete driver implementations
# ─────────────────────────────────────────────

class RGBCameraDriver(SensorDriver):
    """
    Driver for any RGB camera (USB, MIPI, GStreamer pipeline).
    Uses OpenCV VideoCapture under the hood — swap with SDK if needed.
    """

    def __init__(self, sensor_id: str, uav_id: str,
                 source: Any = 0,                # device index OR gstreamer string
                 resolution: tuple = (1280, 720),
                 fps: int = 30):
        super().__init__(sensor_id, uav_id)
        self._source     = source
        self._resolution = resolution
        self._fps        = fps
        self._cap        = None

    @property
    def sensor_type(self) -> SensorType:
        return SensorType.RGB

    def connect(self) -> bool:
        try:
            import cv2
            self._cap = cv2.VideoCapture(self._source)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
            self._cap.set(cv2.CAP_PROP_FPS,          self._fps)
            self._connected = self._cap.isOpened()
            return self._connected
        except Exception as e:
            print(f"[RGBCameraDriver] connect failed: {e}")
            return False

    def disconnect(self) -> None:
        if self._cap:
            self._cap.release()
        self._connected = False

    def read_frame(self) -> SensorFrame:
        if not self._connected:
            return self._make_frame(None)
        import cv2
        ret, frame = self._cap.read()
        if not ret:
            return self._make_frame(None)
        # OpenCV gives BGR — convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._make_frame(
            raw_data = frame_rgb,          # np.ndarray [H,W,3] uint8
            metadata = {
                "resolution": self._resolution,
                "fps":        self._fps,
                # intrinsics can be added after calibration
                "fx": None, "fy": None, "cx": None, "cy": None,
            }
        )


class LiDARDriver(SensorDriver):
    """
    Driver for LiDAR sensors via ROS2 topic or direct SDK.
    Expects point cloud as [N,4] float32 (x, y, z, intensity).
    Replace _read_from_ros2() with vendor SDK call as needed.
    """

    def __init__(self, sensor_id: str, uav_id: str,
                 topic: str = "/lidar/points",
                 max_range: float = 100.0):
        super().__init__(sensor_id, uav_id)
        self._topic     = topic
        self._max_range = max_range
        self._latest_cloud = None

    @property
    def sensor_type(self) -> SensorType:
        return SensorType.LIDAR

    def connect(self) -> bool:
        # In a real system: subscribe to ROS2 topic or open SDK handle.
        # For now we mark connected and expect read_frame to be called
        # once data is injected (simulation / test mode).
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def inject_cloud(self, point_cloud) -> None:
        """
        Test / simulation hook.
        In production this is replaced by a ROS2 subscriber callback.
        """
        self._latest_cloud = point_cloud

    def read_frame(self) -> SensorFrame:
        import numpy as np
        if self._latest_cloud is None:
            return self._make_frame(None)
        cloud = self._latest_cloud
        self._latest_cloud = None       # consume it
        # clip to max range
        dists = np.linalg.norm(cloud[:, :3], axis=1)
        cloud = cloud[dists <= self._max_range]
        return self._make_frame(
            raw_data = cloud,           # np.ndarray [N,4] float32
            metadata = {"max_range": self._max_range, "topic": self._topic}
        )


class ToFDriver(SensorDriver):
    """
    Driver for Time-of-Flight depth sensors (Intel RealSense, OAK-D, etc.)
    Uses pyrealsense2 as reference; swap for another SDK as needed.
    """

    def __init__(self, sensor_id: str, uav_id: str,
                 resolution: tuple = (640, 480),
                 max_depth: float = 10.0):
        super().__init__(sensor_id, uav_id)
        self._resolution = resolution
        self._max_depth  = max_depth
        self._pipeline   = None

    @property
    def sensor_type(self) -> SensorType:
        return SensorType.TOF

    def connect(self) -> bool:
        try:
            import pyrealsense2 as rs
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(
                rs.stream.depth,
                self._resolution[0], self._resolution[1],
                rs.format.z16, 30
            )
            self._pipeline.start(config)
            self._connected = True
            return True
        except Exception as e:
            print(f"[ToFDriver] connect failed: {e}")
            return False

    def disconnect(self) -> None:
        if self._pipeline:
            self._pipeline.stop()
        self._connected = False

    def read_frame(self) -> SensorFrame:
        import numpy as np
        if not self._connected:
            return self._make_frame(None)
        try:
            import pyrealsense2 as rs
            frames     = self._pipeline.wait_for_frames()
            depth      = frames.get_depth_frame()
            depth_arr  = np.asanyarray(depth.get_data()).astype(np.float32)
            depth_m    = depth_arr * depth.get_units()              # convert to metres
            depth_m    = np.clip(depth_m, 0, self._max_depth)
            return self._make_frame(
                raw_data = depth_m,     # np.ndarray [H,W] float32 in metres
                metadata = {
                    "resolution": self._resolution,
                    "max_depth":  self._max_depth,
                }
            )
        except Exception as e:
            print(f"[ToFDriver] read_frame failed: {e}")
            return self._make_frame(None)


# ─────────────────────────────────────────────
# Future-proof driver base for unknown sensors
# ─────────────────────────────────────────────

class GenericDriver(SensorDriver):
    """
    Drop-in base for any sensor that doesn't fit the above categories.
    Subclass, set sensor_type = SensorType.UNKNOWN (or add a new enum value),
    implement connect / disconnect / read_frame, and register it below.
    """

    @property
    def sensor_type(self) -> SensorType:
        return SensorType.UNKNOWN

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def read_frame(self) -> SensorFrame:
        return self._make_frame(None)


# ─────────────────────────────────────────────
# HAL Registry — runtime plugin system
# ─────────────────────────────────────────────

class SensorHAL:
    """
    Central registry that holds all active sensor drivers for one UAV.

    Usage:
        hal = SensorHAL(uav_id="quad_01")
        hal.register(RGBCameraDriver("rgb_front", "quad_01", source=0))
        hal.register(LiDARDriver("lidar_main", "quad_01"))
        hal.connect_all()

        frame = hal.read("rgb_front")
        all_frames = hal.read_all()
    """

    def __init__(self, uav_id: str):
        self.uav_id   = uav_id
        self._drivers: Dict[str, SensorDriver] = {}

    def register(self, driver: SensorDriver) -> None:
        """Add a driver. Call before connect_all()."""
        if driver.sensor_id in self._drivers:
            raise ValueError(f"Sensor '{driver.sensor_id}' already registered.")
        self._drivers[driver.sensor_id] = driver
        print(f"[HAL] Registered sensor: {driver.sensor_id} ({driver.sensor_type})")

    def connect_all(self) -> Dict[str, bool]:
        """Attempt to connect every registered driver."""
        results = {}
        for sid, driver in self._drivers.items():
            ok = driver.connect()
            results[sid] = ok
            status = "✓" if ok else "✗"
            print(f"[HAL] {status} {sid}")
        return results

    def disconnect_all(self) -> None:
        for driver in self._drivers.values():
            driver.disconnect()

    def read(self, sensor_id: str) -> SensorFrame:
        """Read one frame from a specific sensor by ID."""
        if sensor_id not in self._drivers:
            raise KeyError(f"No sensor registered with id '{sensor_id}'")
        return self._drivers[sensor_id].read_frame()

    def read_all(self) -> Dict[str, SensorFrame]:
        """Read one frame from every connected sensor."""
        return {
            sid: driver.read_frame()
            for sid, driver in self._drivers.items()
            if driver.is_connected
        }

    def sensor_manifest(self) -> Dict[str, str]:
        """Returns {sensor_id: sensor_type} — used by Drone Profile."""
        return {sid: d.sensor_type.value for sid, d in self._drivers.items()}

    def __repr__(self) -> str:
        sensors = ", ".join(f"{s}({d.sensor_type})" for s, d in self._drivers.items())
        return f"SensorHAL(uav={self.uav_id}, sensors=[{sensors}])"