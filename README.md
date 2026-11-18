# ros2_camera_calibration

Checkerboard camera calibration for ROS 2, pinhole **or** fisheye (OpenCV). Headless, so it works over SSH.
Tested logic on OpenCV 4.x and 5.x; target Ubuntu 22.04 (Humble) / 24.04 (Jazzy).

## Install
```bash
sudo apt install python3-opencv python3-yaml ros-$ROS_DISTRO-cv-bridge
cd ~/ros2_ws/src && git clone <this repo>
cd ~/ros2_ws && colcon build --packages-select ros2_camera_calibration && source install/setup.bash
```

## Use
```bash
ros2 run ros2_camera_calibration calibrate_node --ros-args \
  -p image_topic:=/image_raw -p board_cols:=9 -p board_rows:=6 -p square_size:=0.025 -p model:=both
rqt_image_view /calibration/debug_image          # live corners + view counter
ros2 service call /calibrate/calibrate std_srvs/srv/Trigger   # after >= 20 views
```
Hold the board still at different positions, distances and tilts, **including the corners**.
The node saves a view when the board is sharp, still and in a new place.
Services: `/calibrate/capture` (save now), `/calibrate/calibrate`, `/calibrate/reset`.

`board_cols` / `board_rows` count **inner corners** (a 10x7-square board is 9x6).

## Output
`calibration_out/<camera_name>_pinhole.yaml` and/or `_fisheye.yaml` in `camera_info` format
(`plumb_bob` or `equidistant`). The reply lists RMS error (px) and FOV for each model.
`model:=both` fits both and ranks them; pinhole wins ties (fisheye must be >5 % better).
RMS above 1 px means capture more views. FOV prints `nan` if the board never reached the image edges.

## Which model?
Mild barrel distortion, under about 120 degrees: pinhole. A circular image or about 150 degrees and up: fisheye.
Run `both` and compare if unsure.

## Test (no ROS needed)
`python3 test/test_calib.py`  (synthetic board through a known pinhole and fisheye lens)
