# Point Cloud Scanning System with Loop Closure

Complete system for 360° object scanning (e.g., cars) using multiple MP4 videos with loop closure optimization.

## Features

- ✅ Multi-video processing (merge multiple 360° scans)
- ✅ Loop closure detection and optimization
- ✅ Pose graph optimization for drift correction
- ✅ High-quality mesh reconstruction
- ✅ Cross-platform (Ubuntu & Windows)
- ✅ GPU-accelerated processing

## Directory Structure

```
pointcloud_scan/
├── src/
│   ├── loop_closure.py          # Loop closure detection
│   ├── pose_optimization.py     # Pose graph optimization
│   ├── multi_video_processor.py # Multi-video pipeline
│   ├── mesh_reconstruction.py   # Mesh generation
│   └── utils.py                 # Utilities
├── config/
│   └── scan_config.yaml         # Configuration file
├── output/                      # Output directory
├── docs/
│   └── THEORY.md                # Comprehensive theory
└── README.md                    # This file
```

## Quick Start

### 1. Process Multiple Videos

```bash
# Process multiple 360° videos of a car
python -m pointcloud_scan.src.multi_video_processor \
    --videos video1.mp4 video2.mp4 video3.mp4 \
    --output-dir pointcloud_scan/output/car_scan \
    --camera-params camera_params.json \
    --enable-loop-closure \
    --generate-mesh
```

### 2. Using the CLI

```bash
# Add to main CLI
da3 scan-360 \
    --videos video1.mp4 video2.mp4 \
    --output-dir pointcloud_scan/output/car_scan \
    --camera-params params.json \
    --fps 5.0 \
    --loop-closure-threshold 0.8
```

## Theory

See [docs/THEORY.md](docs/THEORY.md) for comprehensive explanation of:
- Loop closure detection algorithms
- Pose graph optimization
- g2o optimization framework
- Mesh reconstruction techniques

## Requirements

```bash
pip install open3d scipy scikit-learn networkx
```

For g2o optimization (optional, for best results):
```bash
# Ubuntu
sudo apt-get install libg2o-dev

# Or use Python wrapper
pip install g2opy
```

## Examples

### Example 1: Car Scanning
```bash
python -m pointcloud_scan.src.multi_video_processor \
    --videos front.mp4 side.mp4 back.mp4 \
    --output-dir output/my_car \
    --camera-params femto_bolt.json \
    --fps 3.0 \
    --voxel-size 0.005 \
    --loop-threshold 0.85 \
    --mesh-method poisson \
    --mesh-depth 9
```

### Example 2: Quick Test
```bash
python -m pointcloud_scan.src.multi_video_processor \
    --videos test.mp4 \
    --output-dir output/test \
    --fps 2.0 \
    --no-mesh
```

## Configuration

Edit `config/scan_config.yaml` to customize:
- Loop closure parameters
- Optimization settings
- Mesh generation options
- GPU settings

## Output Files

```
output/car_scan/
├── point_cloud_raw.ply           # Before loop closure
├── point_cloud_optimized.ply     # After loop closure
├── mesh.ply                      # Reconstructed mesh
├── mesh.obj                      # OBJ format (with textures)
├── loop_closures.json            # Detected loops
├── pose_graph.json               # Optimized poses
├── metadata.json                 # Processing info
└── visualization/
    ├── trajectory_before.png     # Before optimization
    └── trajectory_after.png      # After optimization
```
