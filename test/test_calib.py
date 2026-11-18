"""Synthetic test: project a board through a known camera, check we recover it.
Run:  python3 test/test_calib.py"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ros2_camera_calibration import calib

SIZE = (1920, 1080)
COLS, ROWS, SQ = 9, 6, 0.03
rng = np.random.default_rng(0)
obj1 = calib.board_points(COLS, ROWS, SQ)


def views(project, n=30):
    obj, img = [], []
    while len(obj) < n:
        rvec = rng.uniform(-0.5, 0.5, 3)
        tvec = np.array([rng.uniform(-0.15, 0.05), rng.uniform(-0.1, 0.05), rng.uniform(0.35, 0.7)])
        p = project(obj1, rvec, tvec).reshape(-1, 2)
        if (p[:, 0] > 5).all() and (p[:, 0] < SIZE[0] - 5).all() and (p[:, 1] > 5).all() and (p[:, 1] < SIZE[1] - 5).all():
            obj.append(obj1.copy())
            img.append((p + rng.normal(0, 0.15, p.shape)).astype(np.float32).reshape(-1, 1, 2))
    return obj, img


# pinhole truth: f = 1000, mild barrel distortion
Kp = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1.0]])
Dp = np.array([-0.2, 0.05, 0, 0, 0.0])
obj, img = views(lambda o, r, t: cv2.projectPoints(o.reshape(-1, 3), r, t, Kp, Dp)[0])
res = calib.calibrate(obj, img, SIZE, "both")
best = res[0]
print("pinhole data  ->", [(r["model"], round(r.get("rms", -1), 3)) for r in res])
assert best["model"] == "pinhole" and best["rms"] < 0.5  # tie -> simpler model wins
assert abs(best["K"][0, 0] - 1000) < 15, best["K"]

# fisheye truth
Kf = np.array([[600, 0, 960], [0, 600, 540], [0, 0, 1.0]])
Df = np.array([0.05, -0.01, 0.0, 0.0]).reshape(4, 1)
obj, img = views(lambda o, r, t: cv2.fisheye.projectPoints(o.reshape(1, -1, 3).astype(np.float64), r, t, Kf, Df)[0])
res = calib.calibrate(obj, img, SIZE, "both")
print("fisheye data  ->", [(r["model"], round(r.get("rms", -1), 3), r.get("error")) for r in res])
fish = [r for r in res if r["model"] == "fisheye"][0]
assert fish["rms"] < 0.5 and abs(fish["K"][0, 0] - 600) < 10, fish
print("FOV h,v (nan = edges not covered by test board):", [round(x, 1) for x in fish["fov_deg"]])
y = calib.to_camera_info_yaml(fish, "test")
assert y["distortion_model"] == "equidistant" and len(y["distortion_coefficients"]["data"]) == 4
print("OK")
