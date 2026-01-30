import argparse
import logging
from dataclasses import MISSING
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from perceive_semantix_lib import Config, DebugConfig, DetectionConfig, InputData, InputDataStamped, Scene


def main():
    """Run Perceive Semantix on prerecorded camera inputs (RGB-D + pose + intrinsics) stored on disk."""
    parser = argparse.ArgumentParser(
        description="Example interface to run Perceive Semantix on prerecorded camera inputs (RGB-D + pose + intrinsics) stored on disk."
        "\n The 'native' format (used by default) expects data organized as:"
        "\nThe input data should organized in a folder structure as follows:"
        "\n<scene_directory>/"
        "\n    input/"
        "\n        color/"
        "\n            <timestamp>.png"
        "\n            ..."
        "\n        depth/"
        "\n            <timestamp>.npy"
        "\n            ..."
        "\n        camera.csv"
        "\n    ..."
        "\nThe camera.csv file should contain a header and one row per timestamp with the following format (K=intrinsics, T=extrinsics):"
        "\n    timestamp, K_00, K_01, K_02, K_10, K_11, K_12, K_20, K_21, K_22, T_00, T_01, T_02, T_03, T_10, T_11, T_12, T_13, T_20, T_21, T_22, T_23, T_30, T_31, T_32, T_33",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("path", type=Path, help="Path to the scene directory")
    parser.add_argument("--init", "-i", type=Path, help="Path to initial map to load", default=None)
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--format", choices=["native", "3rscan"], default="native", help="Input data format (default: native)"
    )
    parser.add_argument(
        "--rotate90deg",
        choices=[-1, 0, 1],
        type=int,
        default=0,
        help="Rotate input images by +/-90 degrees clockwise (1: +90deg, -1: -90deg, 0: no rotation)",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("perceive_semantix_lib").setLevel(logging.DEBUG)
    else:
        logging.getLogger("perceive_semantix_lib").setLevel(logging.INFO)

    if args.format == "native":
        from load_native import check_database as native_check_database
        from load_native import iterate_database as native_iterate_database

        check_database = native_check_database
        iterate_database = native_iterate_database
        scene_path: Path = args.path / "input"
        img_width_height = (360, 640)
        blur_threshold = MISSING  # use default value instead

    elif args.format == "3rscan":
        from load_3rscan import check_database as rscan_check_database
        from load_3rscan import iterate_database as rscan_iterate_database

        check_database = rscan_check_database
        iterate_database = rscan_iterate_database
        scene_path: Path = args.path
        img_width_height = (540, 960)
        blur_threshold = 20.0

    else:
        raise SystemExit(f"Unsupported format: {args.format}")

    # Check database consistency
    if not scene_path.exists():
        raise SystemExit(f"Scene path not found: {scene_path.resolve()}")
    if not check_database(scene_path):
        raise SystemExit(f"Scene database is not consistent: {scene_path.resolve()}")

    # Setup scene
    config = Config(
        debug=DebugConfig(
            store_scene_each_frame=False,
        ),
        detection=DetectionConfig(
            object_class_names_file=(
                Path(__file__).parent.parent / "ros2" / "perceive_semantix_ros2" / "object_classes.txt"
            ),
            **{k: v for k, v in [("blur_threshold", blur_threshold)] if v is not MISSING},  # type: ignore
        ),
        input_image_height=img_width_height[1],
        input_image_width=img_width_height[0],
    )
    scene = Scene(config=config, initial_scene_path=args.init)

    pose_transfrom = np.eye(4)
    pose_transfrom[:3, :3] = Rotation.from_euler("z", -90 * args.rotate90deg, degrees=True).as_matrix()

    # Feed input data into scene
    logging.info(f"Processing scene from database at {scene_path.resolve()}")
    for time_sec, color_img, depth_img, camera_intrinsics, camera_pose in iterate_database(scene_path):
        if args.rotate90deg in [-1, 1]:
            camera_intrinsics[(0, 0)], camera_intrinsics[(1, 1)] = camera_intrinsics[(1, 1)], camera_intrinsics[(0, 0)]
            camera_intrinsics[(0, 2)], camera_intrinsics[(1, 2)] = camera_intrinsics[(1, 2)], camera_intrinsics[(0, 2)]
            color_img = np.rot90(color_img, -args.rotate90deg).copy()
            depth_img = np.rot90(depth_img, -args.rotate90deg).copy()
            camera_pose = camera_pose @ pose_transfrom

        input = InputDataStamped(
            time_sec=time_sec,
            data=InputData(
                camera_intrinsics=camera_intrinsics,
                color=color_img,
                depth=depth_img,
                pose=camera_pose,
            ),
        )
        scene.step(input)


if __name__ == "__main__":
    main()
