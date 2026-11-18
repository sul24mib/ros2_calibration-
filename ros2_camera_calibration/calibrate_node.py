#!/usr/bin/env python3
"""ROS 2 checkerboard calibration node (pinhole or fisheye). Headless: works over SSH.

  ros2 run ros2_camera_calibration calibrate_node --ros-args \
      -p image_topic:=/image_raw -p board_cols:=9 -p board_rows:=6 -p square_size:=0.025

Hold the board in front of the camera; the node saves a view whenever it sees the board
sharp, still and in a new position. Watch /calibration/debug_image (rqt_image_view).
When enough views are saved:   ros2 service call /calibrate/calibrate std_srvs/srv/Trigger
Other services:  /calibrate/capture (save this frame now), /calibrate/reset (start over).
"""
import os

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from . import calib


class CalibrateNode(Node):
    def __init__(self):
        super().__init__("calibrate")
        P = self.declare_parameter
        self.topic = P("image_topic", "/image_raw").value
        self.cols = P("board_cols", 9).value            # inner corners, not squares
        self.rows = P("board_rows", 6).value
        self.square = P("square_size", 0.025).value     # metres (only matters for extrinsics)
        self.model = P("model", "both").value           # pinhole | fisheye | both
        self.min_views = P("min_views", 20).value
        self.auto = P("auto_capture", True).value
        self.min_move_px = P("min_move_px", 40.0).value  # new view must differ this much
        self.min_sharp = P("min_sharpness", 50.0).value
        self.out_dir = P("output_dir", "calibration_out").value
        self.cam_name = P("camera_name", "camera").value
        if self.model not in ("pinhole", "fisheye", "both"):
            raise ValueError("model must be pinhole, fisheye or both")

        self.bridge = CvBridge()
        self.obj1 = calib.board_points(self.cols, self.rows, self.square)
        self.obj, self.img, self.cells = [], [], set()
        self.size = None
        self.last_corners = None
        self.last_gray = None
        self.last_found = None
        self.prev_center = None

        self.create_subscription(Image, self.topic, self.on_image, qos_profile_sensor_data)
        self.debug_pub = self.create_publisher(Image, "/calibration/debug_image", 1)
        self.create_service(Trigger, "~/capture", self.srv_capture)
        self.create_service(Trigger, "~/calibrate", self.srv_calibrate)
        self.create_service(Trigger, "~/reset", self.srv_reset)
        self.get_logger().info(
            f"listening on {self.topic}; board {self.cols}x{self.rows} inner corners, model={self.model}")

    # -- capture ---------------------------------------------------------
    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        if self.size is not None and size != self.size:
            self.get_logger().warn(f"image size changed {self.size} -> {size}; use /reset", throttle_duration_sec=5)
            return
        self.size = size
        corners = calib.find_corners(gray, self.cols, self.rows)
        self.last_gray, self.last_found = gray, corners

        note = "no board"
        if corners is not None:
            sharp = calib.sharpness(gray)
            c = corners.reshape(-1, 2).mean(axis=0)
            still = self.prev_center is not None and np.linalg.norm(c - self.prev_center) < 3.0
            self.prev_center = c
            note = f"board found, sharp={sharp:.0f}"
            if self.auto and still and sharp >= self.min_sharp and self._is_new(corners):
                self._add(corners)
                note = "SAVED"
            cv2.drawChessboardCorners(frame, (self.cols, self.rows), corners, True)
        else:
            self.prev_center = None
        cv2.putText(frame, f"{len(self.obj)}/{self.min_views} views  cells {len(self.cells)}/9  {note}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

    def _is_new(self, corners):
        if self.last_corners is None:
            return True
        return float(np.mean(np.linalg.norm(corners.reshape(-1, 2) - self.last_corners.reshape(-1, 2), axis=1))) \
            > self.min_move_px

    def _add(self, corners):
        self.obj.append(self.obj1.copy())
        self.img.append(corners.copy())
        self.last_corners = corners
        self.cells.add(calib.coverage_cell(corners, self.size))
        self.get_logger().info(f"saved view {len(self.obj)}/{self.min_views}")

    # -- services --------------------------------------------------------
    def srv_capture(self, _req, res):
        if self.last_found is None:
            res.success, res.message = False, "no board in the last frame"
        else:
            self._add(self.last_found)
            res.success, res.message = True, f"{len(self.obj)} views"
        return res

    def srv_reset(self, _req, res):
        self.obj, self.img, self.cells, self.last_corners, self.size = [], [], set(), None, None
        res.success, res.message = True, "cleared"
        return res

    def srv_calibrate(self, _req, res):
        if len(self.obj) < 8:
            res.success, res.message = False, f"need at least 8 views, have {len(self.obj)}"
            return res
        results = calib.calibrate(self.obj, self.img, self.size, self.model)
        lines = []
        for r in results:
            if "error" in r:
                lines.append(f"{r['model']}: FAILED ({r['error']})")
            else:
                lines.append(f"{r['model']}: rms {r['rms']:.3f} px, FOV {r['fov_deg'][0]:.1f} x "
                             f"{r['fov_deg'][1]:.1f} deg, views {r['views_used']} (dropped {r['views_dropped']})")
        ok = [r for r in results if "error" not in r]
        if not ok:
            res.success, res.message = False, "; ".join(lines)
            return res
        os.makedirs(self.out_dir, exist_ok=True)
        for r in ok:
            path = os.path.join(self.out_dir, f"{self.cam_name}_{r['model']}.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(calib.to_camera_info_yaml(r, self.cam_name), f, sort_keys=False)
        best = ok[0]
        msg = f"BEST: {best['model']}. " + " | ".join(lines) + f" | saved to {os.path.abspath(self.out_dir)}"
        if best["rms"] > 1.0:
            msg += " | WARNING: rms > 1 px, capture more views incl. corners and tilted"
        self.get_logger().info(msg)
        res.success, res.message = True, msg
        return res


def main():
    rclpy.init()
    node = CalibrateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
