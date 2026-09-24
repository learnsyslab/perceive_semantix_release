# Changelog

All notable changes to this project are documented in this file.
Entries are listed by the date they were published, newest first.

## 2026-09-24

### Breaking changes

- **ROS 2 Humble → Jazzy.** The pixi environment now uses the `robostack-jazzy` channel and Python 3.12
  (previously `robostack-humble` and Python 3.11). Existing environments and `colcon` builds need to be
  recreated (`pixi install`, then `colcon build` from a clean `build/`/`install/`).
- **Default camera topics changed** from `/spectacular_ai/*` to `/femtobolt/color/camera_info`,
  `/femtobolt/color/image_raw` and `/femtobolt/depth/image_raw`, so the node works with any camera driver or
  SLAM stack that publishes under the camera's namespace. To use the example ROS bag, pass the
  `/spectacular_ai/*` topics explicitly (see README).
- **Background map is now an OctoMap.** The background geometry is stored as an OctoMap instead of a
  downsampled point cloud, and the ROS node publishes it as `octomap_msgs/Octomap` on `map/background_octomap`
  (previously `sensor_msgs/PointCloud2` on `map/background`).
  - ROS parameter `publishing_rate_background_pointcloud` was renamed to `publishing_rate_background`.
  - Config option `downsample_background_voxel_size` was replaced by a `background` section
    (`resolution_m`, `max_range_m`, `prob_hit`, `prob_miss`, `store_color`, `lazy_eval`, `rerun_chunk_size_m`).

### Added

- **Occlusion handling:** objects hidden behind closer geometry in the current depth image are no longer
  treated as missing (config option `occlusion_margin_m`). Invalid depth measurements are not counted as
  occlusions.
- **3D exploration priority map:** set the ROS parameter `exploration_map/dimensions` to `3` for a voxelized
  priority map (published as `PointCloud2` on `map/exploration`); `2` keeps the ground-projected occupancy grid.
- **Rerun launch modes:** choose how the Rerun viewer is used with the ROS parameter `rerun_mode`
  (`spawn`, `serve_grpc`, `connect_grpc` or `disabled`), or with the config options `rerun_launch_mode` and
  `rerun_grpc_connect_url`. This allows running headless or viewing remotely.
- ROS parameters `target_image_width` and `target_image_height` to set the input image size after rotation.
- Camera frames are buffered until a matching (possibly delayed) SLAM pose is available on `/tf`.
- The pixi environment sources `install/setup.sh` on activation, so the ROS package is available in
  `pixi shell` after the first build.

### Fixed

- Camera pose was not rotated consistently with the image when `image_rotations_clockwise` was set.
- Occupancy map was published flipped.
- Crash when logging object associations to Rerun with missing (`None`) entries.
- Example ROS bag commands in the README use the correct topics.

### Changed

- Updated and pinned Rerun (`rerun-sdk` 0.34), with a workaround for spawning the viewer.
- Updated `pixi.lock` for all environments.

## 2026-01-30

- Initial public release.
