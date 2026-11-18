"""Pure OpenCV calibration logic (no ROS), so it can be tested anywhere."""
import math

import cv2
import numpy as np


def board_points(cols, rows, square):
    """3D corner positions of the board (z = 0), in metres. cols, rows = inner corners."""
    p = np.zeros((1, cols * rows, 3), np.float32)
    p[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return p


def sharpness(gray):
    """Variance of the Laplacian: low = blurred."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def find_corners(gray, cols, rows):
    """Return refined corners (N,1,2) float32, or None if the board isn't found."""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)


def coverage_cell(corners, size, grid=3):
    """Which cell of a grid x grid picture the board centre is in (for coverage hints)."""
    w, h = size
    cx, cy = corners.reshape(-1, 2).mean(axis=0)
    return min(int(cy / h * grid), grid - 1), min(int(cx / w * grid), grid - 1)


def _rms_per_view(obj, img, rvecs, tvecs, project):
    errs = []
    for o, i, r, t in zip(obj, img, rvecs, tvecs):
        proj = project(o, r, t)
        errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, axis=1)))))
    return errs


def calibrate_pinhole(obj, img, size):
    """Standard model, 5 coefficients (k1 k2 p1 p2 k3). Returns dict."""
    obj3 = [o.reshape(-1, 3) for o in obj]
    img2 = [i.reshape(-1, 1, 2) for i in img]
    rms, K, D, rv, tv = cv2.calibrateCamera(obj3, img2, size, None, None)
    per = _rms_per_view(obj3, img2, rv, tv, lambda o, r, t: cv2.projectPoints(o, r, t, K, D)[0])
    return {"model": "pinhole", "rms": float(rms), "K": K, "D": D.reshape(-1), "per_view": per,
            "distortion_model": "plumb_bob", "size": size}


def calibrate_fisheye(obj, img, size):
    """Kannala-Brandt fisheye model, 4 coefficients. Returns dict, or raises cv2.error."""
    obj3 = [o.reshape(1, -1, 3).astype(np.float64) for o in obj]
    img2 = [i.reshape(1, -1, 2).astype(np.float64) for i in img]
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    # OpenCV 4 keeps these in cv2.fisheye, OpenCV 5 in cv2
    c = cv2.fisheye if hasattr(cv2.fisheye, "CALIB_FIX_SKEW") else cv2
    flags = c.CALIB_RECOMPUTE_EXTRINSIC | c.CALIB_FIX_SKEW
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    rms, K, D, rv, tv = cv2.fisheye.calibrate(obj3, img2, size, K, D, flags=flags, criteria=crit)
    per = _rms_per_view(obj3, img2, rv, tv, lambda o, r, t: cv2.fisheye.projectPoints(o, r, t, K, D)[0])
    return {"model": "fisheye", "rms": float(rms), "K": K, "D": D.reshape(-1), "per_view": per,
            "distortion_model": "equidistant", "size": size}


def drop_worst(obj, img, per_view, factor=2.0, keep_min=8):
    """Drop views whose error is > factor x the median (blurred or mis-detected)."""
    med = float(np.median(per_view))
    keep = [i for i, e in enumerate(per_view) if e <= factor * med]
    if len(keep) < keep_min:
        return obj, img, 0
    return [obj[i] for i in keep], [img[i] for i in keep], len(obj) - len(keep)


def calibrate(obj, img, size, model="both"):
    """Run the requested model(s); fit, drop outlier views, refit. Returns list of results, best first."""
    results = []
    fns = {"pinhole": calibrate_pinhole, "fisheye": calibrate_fisheye}
    for name in (["pinhole", "fisheye"] if model == "both" else [model]):
        try:
            r = fns[name](obj, img, size)
            o2, i2, dropped = drop_worst(obj, img, r["per_view"])
            if dropped:
                r = fns[name](o2, i2, size)
            r["views_used"] = len(o2)
            r["views_dropped"] = dropped
            r["fov_deg"] = field_of_view(r)
            results.append(r)
        except cv2.error as e:
            results.append({"model": name, "error": str(e).strip().splitlines()[-1]})
    # ties go to the simpler pinhole model: fisheye must be >5 % better to win
    score = lambda r: r.get("rms", float("inf")) * (0.95 if r["model"] == "pinhole" else 1.0)
    return sorted(results, key=score)


def _fisheye_theta(r_d, D):
    """Angle from the optical axis for a distorted radius r_d (Kannala-Brandt), by Newton."""
    th = r_d
    for _ in range(50):
        t2 = th * th
        f = th * (1 + D[0] * t2 + D[1] * t2 ** 2 + D[2] * t2 ** 3 + D[3] * t2 ** 4) - r_d
        d = 1 + 3 * D[0] * t2 + 5 * D[1] * t2 ** 2 + 7 * D[2] * t2 ** 3 + 9 * D[3] * t2 ** 4
        th -= f / d
    # not converged, or the fitted curve is not increasing: the edge was never observed
    return th if (abs(f) < 1e-9 and 0 <= th < math.pi) else float("nan")


def field_of_view(r):
    """Horizontal, vertical FOV in degrees, measured through the lens model (edge midpoints).
    Works past 180 deg for fisheye (a tan() formula cannot)."""
    w, h = r["size"]
    K, D = np.asarray(r["K"], float), np.asarray(r["D"], float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def angle(dx, dy):   # pixel offset from the centre -> angle from the axis (radians)
        x, y = dx / fx, dy / fy
        if r["model"] == "fisheye":
            return _fisheye_theta(math.hypot(x, y), D)
        u = cv2.undistortPoints(np.array([[[dx + cx, dy + cy]]], np.float64), K, D).reshape(2)
        return math.atan(math.hypot(*u))

    hfov = angle(-cx, 0) + angle(w - 1 - cx, 0)
    vfov = angle(0, -cy) + angle(0, h - 1 - cy)
    return math.degrees(hfov), math.degrees(vfov)


def to_camera_info_yaml(r, name):
    """ROS camera_info dict (the format camera_info_manager / usb_cam read)."""
    w, h = r["size"]
    K = np.asarray(r["K"], float)
    P = np.hstack([K, np.zeros((3, 1))])
    return {
        "image_width": int(w), "image_height": int(h), "camera_name": name,
        "camera_matrix": {"rows": 3, "cols": 3, "data": [float(x) for x in K.reshape(-1)]},
        "distortion_model": r["distortion_model"],
        "distortion_coefficients": {"rows": 1, "cols": len(r["D"]), "data": [float(x) for x in r["D"]]},
        "rectification_matrix": {"rows": 3, "cols": 3, "data": [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]},
        "projection_matrix": {"rows": 3, "cols": 4, "data": [float(x) for x in P.reshape(-1)]},
    }
