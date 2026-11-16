# Copyright (c) 2026, Microduck Robotics Project
# Microduck 480p OpenCV Camera Edge Feed Module

"""Real-time OpenCV camera feed with Canny edge extraction for Microduck biped.

Captures 480p (640x480) video frames from webcam or synthetic stream,
applies Gaussian noise reduction and adaptive Canny edge detection,
and downsamples edges into an 8x8 spatial edge density grid matching the
8x8 ToF depth sensor for multimodal RL perception and live visual display.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import cv2
import numpy as np
import torch


class MicroduckCameraEdgeFeed:
    """Threaded OpenCV 480p camera edge extractor and visualizer."""

    def __init__(
        self,
        camera_id: int | str = 0,
        width: int = 640,
        height: int = 480,
        target_fps: int = 30,
        enable_display: bool = True,
        window_name: str = "Microduck 480p Camera - Real-Time OpenCV Edge Feed",
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.enable_display = enable_display
        self.window_name = window_name

        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Edge detection parameters
        self.blur_ksize = 5
        self.sigma = 0.33
        self.lower_thresh = 50
        self.upper_thresh = 150
        self.auto_threshold = True

        # Latest frames and features (thread-safe)
        self.latest_raw_frame: np.ndarray | None = None
        self.latest_edge_frame: np.ndarray = np.zeros((height, width), dtype=np.uint8)
        self.latest_edge_grid: np.ndarray = np.zeros((8, 8), dtype=np.float32)
        self.fps = 0.0

        # View modes: "edges", "side_by_side", "overlay"
        self.view_mode = "side_by_side"
        self._window_initialized = False

        # Try opening camera
        self.cap = None
        self.is_synthetic = False
        self._init_camera()

        if self.cap is not None or self.is_synthetic:
            self._running = True
            self._thread = threading.Thread(target=self._capture_and_process_loop, daemon=True)
            self._thread.start()

    def _init_camera(self):
        """Attempts to open hardware webcam; falls back to synthetic generator if unavailable."""
        try:
            if isinstance(self.camera_id, int):
                # Try V4L2 on Linux for low latency
                backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
                self.cap = cv2.VideoCapture(self.camera_id, backend)
            else:
                self.cap = cv2.VideoCapture(self.camera_id)

            if self.cap is not None and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.height, self.width = frame.shape[:2]
                    print(f"[INFO][OpenCV Camera]: Connected to {self.camera_id} ({self.width}x{self.height} @ {self.target_fps} FPS)")
                    return
        except Exception as e:
            print(f"[WARNING][OpenCV Camera]: Failed to open hardware camera {self.camera_id}: {e}")

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        print("[INFO][OpenCV Camera]: Hardware camera unavailable. Running synthetic 480p edge generator.")
        self.is_synthetic = True

    def _capture_and_process_loop(self):
        """Asynchronous worker loop continuously capturing frames and extracting edges."""
        frame_count = 0
        t_start = time.perf_counter()

        while self._running:
            raw = None
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    raw = frame
                else:
                    time.sleep(0.01)
                    continue
            elif self.is_synthetic:
                raw = self._generate_synthetic_frame(frame_count)
                time.sleep(1.0 / self.target_fps)

            if raw is None:
                continue

            # Process frame: Grayscale -> Gaussian Blur -> Canny Edges
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 1.2)

            if self.auto_threshold:
                med = float(np.median(blur))
                lower = int(max(0, (1.0 - self.sigma) * med))
                upper = int(min(255, (1.0 + self.sigma) * med))
            else:
                lower = self.lower_thresh
                upper = self.upper_thresh

            edges = cv2.Canny(blur, lower, upper)

            # Downsample to 8x8 spatial edge density grid matching 8x8 ToF sensor
            # Each cell in [0, 1] indicates fraction of edge pixels in that spatial receptive field
            edge_grid = cv2.resize(edges.astype(np.float32) / 255.0, (8, 8), interpolation=cv2.INTER_AREA)

            # Calculate FPS
            frame_count += 1
            elapsed = time.perf_counter() - t_start
            if elapsed >= 1.0:
                self.fps = frame_count / elapsed
                frame_count = 0
                t_start = time.perf_counter()

            # Store processed data safely
            with self._lock:
                self.latest_raw_frame = raw
                self.latest_edge_frame = edges
                self.latest_edge_grid = edge_grid

    def _generate_synthetic_frame(self, frame_count: int) -> np.ndarray:
        """Generates dynamic synthetic 480p hallway scene with obstacles for headless testing."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Floor gradient
        for y in range(self.height // 2, self.height):
            c = int(40 + 60 * (y - self.height // 2) / (self.height // 2))
            frame[y, :] = [c, c, c]

        # Perspective floor lines
        cv2.line(frame, (self.width // 2, self.height // 2), (0, self.height), (120, 120, 120), 2)
        cv2.line(frame, (self.width // 2, self.height // 2), (self.width, self.height), (120, 120, 120), 2)

        # Dynamic obstacle moving towards camera
        phase = (frame_count % 120) / 120.0
        obs_y = int(self.height // 2 + phase * (self.height // 2 - 40))
        obs_h = int(20 + phase * 60)
        obs_w = int(60 + phase * 180)
        obs_x = (self.width - obs_w) // 2

        cv2.rectangle(frame, (obs_x, obs_y), (obs_x + obs_w, obs_y + obs_h), (50, 180, 240), -1)
        cv2.rectangle(frame, (obs_x, obs_y), (obs_x + obs_w, obs_y + obs_h), (255, 255, 255), 2)
        return frame

    def get_edge_features(self) -> np.ndarray:
        """Returns flattened 8x8 edge density array (64 float values in [0.0, 1.0])."""
        with self._lock:
            return self.latest_edge_grid.flatten().copy()

    def get_edge_features_tensor(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """Returns 8x8 edge density as a torch tensor with shape (1, 64)."""
        feats = self.get_edge_features()
        return torch.from_numpy(feats).unsqueeze(0).to(device=device, dtype=torch.float32)

    def get_display_image(self) -> np.ndarray:
        """Creates high-contrast cyber-cyan edge display with real-time HUD telemetry."""
        with self._lock:
            if self.latest_raw_frame is None:
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            raw = self.latest_raw_frame.copy()
            edges = self.latest_edge_frame.copy()
            grid = self.latest_edge_grid.copy()

        # Stylized glowing cyan edges (B=255, G=240, R=0)
        edge_bgr = np.zeros_like(raw)
        edge_bgr[edges > 0] = [255, 235, 10]

        # Receptive field edge density heatmap overlay (scaled up)
        heatmap = cv2.resize((grid * 255.0).astype(np.uint8), (128, 128), interpolation=cv2.INTER_NEAREST)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)

        edge_ratio = float(np.count_nonzero(edges)) / float(edges.size) * 100.0

        if self.view_mode == "side_by_side":
            # Half-width side by side
            raw_half = cv2.resize(raw, (self.width // 2, self.height // 2))
            edge_half = cv2.resize(edge_bgr, (self.width // 2, self.height // 2))
            combined = np.hstack([raw_half, edge_half])

            # Embed 8x8 mini heatmap in bottom-right
            hm_h, hm_w = 80, 80
            hm_resized = cv2.resize(heatmap_color, (hm_w, hm_h), interpolation=cv2.INTER_NEAREST)
            combined[-hm_h - 10 : -10, -hm_w - 10 : -10] = hm_resized
            cv2.rectangle(combined, (combined.shape[1] - hm_w - 10, combined.shape[0] - hm_h - 10), (combined.shape[1] - 10, combined.shape[0] - 10), (255, 255, 255), 1)

            # HUD text
            cv2.putText(combined, "RAW 480p WEBCAM", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(combined, "OPENCV CANNY EDGES", (self.width // 2 + 15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 255), 1)
            cv2.putText(combined, f"FPS: {self.fps:.1f} | Edges: {edge_ratio:.1f}%", (15, combined.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(combined, "8x8 Edge Grid", (combined.shape[1] - hm_w - 15, combined.shape[0] - hm_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            # Obstacle detection alert
            center_density = float(grid[2:6, 2:6].mean())
            if center_density > 0.08:
                cv2.putText(combined, "! OBSTACLE EDGE DETECTED !", (self.width // 2 - 110, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            return combined

        elif self.view_mode == "overlay":
            # Direct edge overlay on raw camera
            overlay = raw.copy()
            overlay[edges > 0] = [0, 255, 255]
            cv2.putText(overlay, f"OPENCV EDGE OVERLAY | FPS: {self.fps:.1f}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return overlay

        else:
            # Pure glowing edges
            pure = edge_bgr.copy()
            cv2.putText(pure, f"MICRODUCK OPENCV EDGES | FPS: {self.fps:.1f}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 255), 2)
            return pure

    def update_display(self) -> int:
        """Pumps the OpenCV window GUI event loop (non-blocking). Returns pressed key code."""
        if not self.enable_display:
            return -1

        # Check for active display environment
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return -1

        img = self.get_display_image()
        try:
            cv2.imshow(self.window_name, img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("\t") or key == ord("v") or key == ord("V"):
                # Cycle view modes
                modes = ["side_by_side", "edges", "overlay"]
                idx = (modes.index(self.view_mode) + 1) % len(modes)
                self.view_mode = modes[idx]
                print(f"[OpenCV Camera]: View mode changed to '{self.view_mode}'")
            elif key == ord("+") or key == ord("="):
                self.upper_thresh = min(self.upper_thresh + 10, 255)
                self.auto_threshold = False
                print(f"[OpenCV Camera]: Canny Upper Threshold -> {self.upper_thresh}")
            elif key == ord("-") or key == ord("_"):
                self.upper_thresh = max(self.upper_thresh - 10, 10)
                self.auto_threshold = False
                print(f"[OpenCV Camera]: Canny Upper Threshold -> {self.upper_thresh}")
            return key
        except Exception:
            return -1

    def close(self):
        """Releases camera and closes any active GUI windows."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
