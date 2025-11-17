# Camera Parameter Examples

This directory contains example camera intrinsic parameter files for use with the `video-pointcloud` command.

## File Format

Camera parameters should be provided as a JSON file with the following structure:

```json
{
    "fx": 591.0,       // focal length in x direction (pixels)
    "fy": 591.0,       // focal length in y direction (pixels)
    "cx": 320.0,       // principal point x coordinate (pixels)
    "cy": 240.0,       // principal point y coordinate (pixels)
    "width": 640,      // original image width (optional)
    "height": 480,     // original image height (optional)
    "depth_scale": 1.0 // scale factor for depth values (optional, default 1.0)
}
```

## Available Examples

- **femto_bolt.json**: Default parameters for Orbbec Femto Bolt RGB camera (640x480)
- **femto_bolt_hd.json**: Parameters for Femto Bolt in HD mode (1280x720)

## Important Notes

1. **Calibration**: The example files contain typical values. For best results, calibrate your specific camera using standard calibration techniques (e.g., OpenCV camera calibration).

2. **Image Resizing**: When providing `width` and `height`, the system automatically scales intrinsic parameters if the processed image resolution differs from the original.

3. **Depth Scale**: The `depth_scale` parameter converts depth values to metric units. For Depth-Anything-3, the output is typically in meters, so use `1.0` unless you need unit conversion.

## Usage Example

```bash
# Basic usage with Femto Bolt camera
da3 video-pointcloud video.mp4 \
    --camera-params examples/camera_params/femto_bolt.json \
    --fps 2.0 \
    --export-dir ./output

# High-quality with outlier removal
da3 video-pointcloud video.mp4 \
    --camera-params examples/camera_params/femto_bolt.json \
    --fps 5.0 \
    --voxel-size 0.005 \
    --remove-outliers \
    --nb-neighbors 30 \
    --std-ratio 1.5 \
    --export-dir ./output
```

## How to Obtain Camera Parameters

### From Orbbec SDK (Femto Bolt)

```python
from pyorbbecsdk import Pipeline, Config, OBSensorType

pipeline = Pipeline()
config = Config()
profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
profile = profile_list.get_default_video_stream_profile()

# Get intrinsics
camera_param = pipeline.get_camera_param()
color_intrinsic = camera_param.rgb_intrinsic

print(f"fx: {color_intrinsic.fx}")
print(f"fy: {color_intrinsic.fy}")
print(f"cx: {color_intrinsic.cx}")
print(f"cy: {color_intrinsic.cy}")
```

### From OpenCV Calibration

```python
import cv2
import json

# After calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# Save parameters
params = {
    "fx": mtx[0, 0],
    "fy": mtx[1, 1],
    "cx": mtx[0, 2],
    "cy": mtx[1, 2],
    "width": image_width,
    "height": image_height,
    "depth_scale": 1.0
}

with open("camera_params.json", "w") as f:
    json.dump(params, f, indent=4)
```

### From ROS Camera Info

If you have ROS camera_info messages:

```python
# K matrix from camera_info
# K = [fx,  0, cx,
#       0, fy, cy,
#       0,  0,  1]

params = {
    "fx": camera_info.K[0],
    "fy": camera_info.K[4],
    "cx": camera_info.K[2],
    "cy": camera_info.K[5],
    "width": camera_info.width,
    "height": camera_info.height,
    "depth_scale": 1.0
}
```
