import threading
import time
from typing import Optional

import cv2
import numpy as np


class CameraStream:
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_timestamp: float = 0.0

    def start(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return True

    def _loop(self) -> None:
        assert self.cap is not None
        while self.running:
            ok, frame = self.cap.read()
            ts = time.time()
            if ok:
                with self.lock:
                    self.latest_frame = frame
                    self.latest_timestamp = ts
            else:
                time.sleep(0.005)

    def read(self) -> tuple[bool, Optional[np.ndarray], float]:
        with self.lock:
            if self.latest_frame is None:
                return False, None, 0.0
            return True, self.latest_frame.copy(), self.latest_timestamp

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
