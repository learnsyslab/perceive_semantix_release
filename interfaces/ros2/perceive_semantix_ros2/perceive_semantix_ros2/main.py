import logging
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import cv2
import numpy as np
from jaxtyping import Float, UInt8
from pyoctomap import ColorOcTree, OcTree  # pyright: ignore[reportAttributeAccessIssue]
from scipy.spatial.transform import Rotation
from typing_extensions import Literal

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped
from message_filters import ApproximateTimeSynchronizer
from message_filters import Subscriber as MF_Subscriber
from nav_msgs.msg import OccupancyGrid as ROSOccupancyGrid
from octomap_msgs.msg import Octomap
from rcl_interfaces.msg import FloatingPointRange, IntegerRange, ParameterDescriptor, ParameterType
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time as RosTime
from sensor_msgs.msg import CameraInfo, PointCloud2
from sensor_msgs.msg import Image as ROSImage
from tf2_ros.transform_listener import TransformListener

from perceive_semantix_interfaces.msg import (
    ObjectDetection,  # pyright: ignore[reportAttributeAccessIssue]
    ObjectDetectionArray,  # pyright: ignore[reportAttributeAccessIssue]
)
from perceive_semantix_interfaces.srv import EnableHeatmap  # pyright: ignore[reportAttributeAccessIssue]
from perceive_semantix_lib import Config, DebugConfig, DetectionConfig, InputData, InputDataStamped, Scene, SceneObject
from perceive_semantix_lib.core.geometry import GeometryType, PointCloud, TensorPointCloud
from perceive_semantix_lib.core.occupancy_grid import OccupancyGrid, SparseVoxelGrid
from perceive_semantix_lib.grid_maps import ExplorationConfig, ExplorationPriorityMap, OccupancyMap

from perceive_semantix_ros2.logging_handler import ROS2LoggingHandler
from perceive_semantix_ros2.ros2_pointcloud import array_to_pointcloud2, merge_rgb_fields, xyz_field_to_pointcloud2
from perceive_semantix_ros2.tf_message_filter import TfMessageFilter, TfSampleTrackingBuffer
from perceive_semantix_ros2.utils import scale_intrinsics


def _strip_octomap_file_header(data: bytes) -> bytes:
    r"""Strip pyoctomap's ASCII file-format header (up to and including `data\n`), leaving the raw payload.

    `octomap_msgs/Octomap.data` expects just the payload, not pyoctomap's file-format header.
    """
    _octomap_file_header_terminator = b"\ndata\n"
    idx = data.find(_octomap_file_header_terminator)
    if idx == -1:
        raise ValueError("Unexpected octomap binary format: no 'data' header terminator found")
    return data[idx + len(_octomap_file_header_terminator) :]


class PerceiveSemantixNode(Node):
    def __init__(self) -> None:
        super().__init__("perceive_semantix")

        # fmt: off
        self.declare_parameter("topic_camera_info"                      , "femtobolt/color/camera_info" , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("topic_color"                            , "femtobolt/color/image_raw"   , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("topic_depth"                            , "femtobolt/depth/image_raw"   , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("image_rotations_clockwise"              , -1                            , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_INTEGER, description="Number of clockwise 90 degree rotations to apply to input images (and the camera pose).", integer_range=[IntegerRange(from_value=-1, to_value=1, step=1)]))
        self.declare_parameter("target_image_width"                     , Config.input_image_width      , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_INTEGER, description="Width the input images are resized to, after image_rotations_clockwise is applied.", integer_range=[IntegerRange(from_value=1, to_value=10000, step=1)]))
        self.declare_parameter("target_image_height"                    , Config.input_image_height     , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_INTEGER, description="Height the input images are resized to, after image_rotations_clockwise is applied.", integer_range=[IntegerRange(from_value=1, to_value=10000, step=1)]))
        self.declare_parameter("camera_depth_scale_to_m"                , 1000.0                        , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_DOUBLE, description="Scale factor to convert depth image values to meters"))
        self.declare_parameter("global_frame"                           , "map"                         , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("min_scene_update_rate_hz"               , 0.5                           , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_DOUBLE, description="Minimum rate (in Hz) at which the scene is updated even if no new input data is available."))
        self.declare_parameter("publishing_rate_background"             , -1.0                          , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_DOUBLE, description="Rate (in Hz) at which the background octomap is published. Set to -1 to publish every time the scene is updated. Set to 0 to disable publishing."))
        self.declare_parameter("occupancy_map/publishing_rate"          , -1.0                          , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_DOUBLE, description="Rate (in Hz) at which the occupancy map is published. Set to -1 to publish every time the scene is updated. Set to 0 to disable publishing."))
        self.declare_parameter("occupancy_map/resolution"               , 0.1                           , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_DOUBLE, description="Resolution (in meters) of the occupancy map grid cells.", floating_point_range=[FloatingPointRange(from_value=0.001, to_value=100.0, step=0.001)]))
        self.declare_parameter("occupancy_map/floor_height"             , 0.3                           , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_DOUBLE, description="Minimum height (in meters) above which points are considered occupied."))
        self.declare_parameter("occupancy_map/robot_height"             , 2.0                           , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_DOUBLE, description="Maximum height (in meters) below which points are considered occupied."))
        self.declare_parameter("exploration_map/dimensions"             , 2                             , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_INTEGER, description="Whether exploration map is 2D (ground-projected) or 3D (voxelized).", integer_range=[IntegerRange(from_value=2, to_value=3, step=1)]))
        self.declare_parameter("exploration_map/publishing_rate"        , -1.0                          , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_DOUBLE, description="Rate (in Hz) at which the exploration map is published. Set to -1 to publish every time the scene is updated. Set to 0 to disable publishing."))
        self.declare_parameter("exploration_map/resolution"             , 0.1                           , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_DOUBLE, description="Resolution (in meters) of the exploration map grid cells.", floating_point_range=[FloatingPointRange(from_value=0.001, to_value=100.0, step=0.001)]))
        self.declare_parameter("objects/point_cloud/publishing_rate"    , -1.0                          , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_DOUBLE, description="Rate (in Hz) at which the objects point cloud is published. Set to -1 to publish every time the scene is updated. Set to 0 to disable publishing."))
        self.declare_parameter("objects/description/publishing_rate"    , -1.0                          , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_DOUBLE, description="Rate (in Hz) at which detected objects are published as ObjectDetectionArray messages. Set to -1 to publish every time the scene is updated. Set to 0 to disable publishing."))
        self.declare_parameter("initial_scene_path"                     , ""                            , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_STRING, description="Path to the initial scene file."))
        self.declare_parameter("store_output"                           , True                          , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_BOOL, description="Whether to store the output."))
        self.declare_parameter("output_path"                            , ""                            , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_STRING, description="Path to store the output. If empty a new output dir 'scene_yyyymmdd_hhmmss' is created."))
        self.declare_parameter("rerun_mode"                             , "spawn"                       , ParameterDescriptor(read_only=True, type=ParameterType.PARAMETER_STRING, description="One of 'spawn', 'serve_grpc', 'connect_grpc', or 'disabled'"))

        topic_camera_info: str                              = str(self.get_parameter("topic_camera_info").get_parameter_value().string_value)
        topic_color: str                                    = str(self.get_parameter("topic_color").get_parameter_value().string_value)
        topic_depth: str                                    = str(self.get_parameter("topic_depth").get_parameter_value().string_value)
        self.rotations_image_clockwise: int                 = int(self.get_parameter("image_rotations_clockwise").get_parameter_value().integer_value)
        target_image_width: int                             = int(self.get_parameter("target_image_width").get_parameter_value().integer_value)
        target_image_height: int                            = int(self.get_parameter("target_image_height").get_parameter_value().integer_value)
        self.camera_depth_scale_to_m: float                 = float(self.get_parameter("camera_depth_scale_to_m").get_parameter_value().double_value)
        self.global_frame: str                              = str(self.get_parameter("global_frame").get_parameter_value().string_value)
        self.min_scene_update_rate_hz: float                = float(self.get_parameter("min_scene_update_rate_hz").get_parameter_value().double_value)
        self.max_scene_update_rate_hz: float                = 5.0
        self.publishing_rate_background: float              = float(self.get_parameter("publishing_rate_background").get_parameter_value().double_value)
        self.publishing_rate_occupancy_map: float           = float(self.get_parameter("occupancy_map/publishing_rate").get_parameter_value().double_value)
        occupancy_map_resolution: float                     = float(self.get_parameter("occupancy_map/resolution").get_parameter_value().double_value)
        occupancy_map_floor_height: float                   = float(self.get_parameter("occupancy_map/floor_height").get_parameter_value().double_value)
        occupancy_map_robot_height: float                   = float(self.get_parameter("occupancy_map/robot_height").get_parameter_value().double_value)
        self.publishing_rate_exploration_map: float         = float(self.get_parameter("exploration_map/publishing_rate").get_parameter_value().double_value)
        exploration_map_dimensions: Literal[2, 3]           = int(self.get_parameter("exploration_map/dimensions").get_parameter_value().integer_value) # type: ignore
        exploration_map_resolution: float                   = float(self.get_parameter("exploration_map/resolution").get_parameter_value().double_value)
        self.publishing_rate_pcd_objects: float             = float(self.get_parameter("objects/point_cloud/publishing_rate").get_parameter_value().double_value)
        self.publishing_rate_descr_objects: float           = float(self.get_parameter("objects/description/publishing_rate").get_parameter_value().double_value)
        initial_scene_path: Optional[Path]                  = Path(self.get_parameter("initial_scene_path").get_parameter_value().string_value) if self.get_parameter("initial_scene_path").get_parameter_value().string_value else None
        store_output: bool                                  = bool(self.get_parameter("store_output").get_parameter_value().bool_value)
        output_path: Optional[Path]                         = Path(self.get_parameter("output_path").get_parameter_value().string_value) if self.get_parameter("output_path").get_parameter_value().string_value else None

        def verify_publishing_rate(param_name: str, rate: float) -> None:
            if rate < 0 and rate != -1:
                self.get_logger().error(msg := f"{param_name} must be non-negative or -1 (but is {rate})")
                raise ValueError(msg)

        verify_publishing_rate("publishing_rate_background", self.publishing_rate_background)
        verify_publishing_rate("publishing_rate_occupancy_map", self.publishing_rate_occupancy_map)
        verify_publishing_rate("publishing_rate_exploration_map", self.publishing_rate_exploration_map)
        verify_publishing_rate("objects/point_cloud/publishing_rate", self.publishing_rate_pcd_objects)
        verify_publishing_rate("objects/description/publishing_rate", self.publishing_rate_descr_objects)

        rerun_mode: str = str(self.get_parameter("rerun_mode").get_parameter_value().string_value)
        if rerun_mode not in ["spawn", "serve_grpc", "connect_grpc", "disabled"]:
            self.get_logger().error(msg := f"rerun_mode must be one of 'spawn', 'serve_grpc', 'connect_grpc', or 'disabled' (but is '{rerun_mode}')")
            raise ValueError(msg)
        # fmt: on

        self.max_message_delay = 1 / 30
        self.tf_wait_timeout = Duration(seconds=1.0)
        # Poses should land at (near enough) the exact stamp of the color image they belong to;
        # a larger gap means tf is interpolating across a frame the pose source never estimated.
        self.tf_max_deviation = Duration(seconds=0.00001)

        self.sub_info = MF_Subscriber(self, CameraInfo, topic_camera_info)
        self.sub_color = MF_Subscriber(self, ROSImage, topic_color)
        self.sub_depth = MF_Subscriber(self, ROSImage, topic_depth)
        self.callback_synchronizer = ApproximateTimeSynchronizer(
            [self.sub_info, self.sub_color, self.sub_depth], 5, self.max_message_delay
        )

        self.tf_buffer = TfSampleTrackingBuffer(root_frame=self.global_frame)
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Buffers frames whose pose isn't ready yet instead of letting the next frame overwrite them.
        self.tf_filter = TfMessageFilter(
            self,
            self.callback_synchronizer,
            self.tf_buffer,
            target_frame=self.global_frame,
            source_frame_from_messages=lambda info_msg, color_msg, depth_msg: color_msg.header.frame_id,
            stamp_from_messages=lambda info_msg, color_msg, depth_msg: RosTime.from_msg(color_msg.header.stamp),
            queue_size=10,
            max_wait=self.tf_wait_timeout,
            max_deviation=self.tf_max_deviation,
            on_drop=lambda msg: self.get_logger().warning(msg),
        )
        self.tf_filter.registerCallback(self._callback_sync)
        self.input: Optional[tuple[InputDataStamped, RosTime]] = None

        self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)  # pyright: ignore[reportAttributeAccessIssue]
        logging_handler = ROS2LoggingHandler(self)
        logging_handler.setLevel(logging.DEBUG)
        logging.getLogger("perceive_semantix_lib").setLevel(logging.DEBUG)
        logging.getLogger("perceive_semantix_lib").addHandler(logging_handler)
        # disable propagation to root logger, otherwise the default handler from root logger (stream) will duplicate the logs
        logging.getLogger("perceive_semantix_lib").propagate = False

        debug_config_args: dict[str, Any] = {
            "store_scene_each_frame": store_output,
            "enable_rerun": rerun_mode != "disabled",
        }
        if output_path is not None:
            debug_config_args["output_root"] = output_path
        if rerun_mode != "disabled":
            debug_config_args["rerun_launch_mode"] = rerun_mode
        scene_config = Config(
            input_image_width=target_image_width,
            input_image_height=target_image_height,
            debug=DebugConfig(
                **debug_config_args,
            ),
            detection=DetectionConfig(
                object_class_names_file=(
                    Path(get_package_share_directory("perceive_semantix_ros2")) / "object_classes/object_classes.txt"
                )
            ),
        )
        self.scene: Scene = Scene(scene_config, initial_scene_path=initial_scene_path)

        if self.publishing_rate_pcd_objects != 0:
            self.publisher_objects_pcd = self.create_publisher(PointCloud2, "map/objects", 1)
        else:
            self.publisher_objects_pcd = None

        if self.publishing_rate_descr_objects != 0:
            self.publisher_objects_descr = self.create_publisher(ObjectDetectionArray, "map/objects_description", 1)
        else:
            self.publisher_objects_descr = None

        if self.publishing_rate_background != 0:
            self.publisher_background = self.create_publisher(Octomap, "map/background_octomap", 1)
        else:
            self.publisher_background = None

        if self.publishing_rate_occupancy_map != 0:
            self.occupancy_map = OccupancyMap(
                self.scene,
                resolution=occupancy_map_resolution,
                occupied_height_bounds=(occupancy_map_floor_height, occupancy_map_robot_height),
            )
            qos_profile = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.publisher_occupancy_map = self.create_publisher(ROSOccupancyGrid, "map", qos_profile)
        else:
            self.occupancy_map = None
            self.publisher_occupancy_map = None

        if self.publishing_rate_exploration_map != 0:
            self.exploration_map = ExplorationPriorityMap(
                object_list=self.scene.detector.object_classes.get_classes_arr(),
                config=ExplorationConfig(
                    resolution_meters=exploration_map_resolution,
                    map_padding_meters=exploration_map_resolution * 3,
                    dimensions=exploration_map_dimensions,
                ),
            )
            if exploration_map_dimensions == 3:
                self.publisher_exploration_map = self.create_publisher(PointCloud2, "map/exploration", 1)
            else:
                self.publisher_exploration_map = self.create_publisher(ROSOccupancyGrid, "map/exploration", 1)
            self.service_exploration_map = self.create_service(
                EnableHeatmap, "set_exploration_query", self._enable_exploration_map_callback
            )
        else:
            self.publisher_exploration_map = None
            self.service_exploration_map = None

    def _callback_sync(
        self, transform_msg: TransformStamped, info_msg: CameraInfo, color_msg: ROSImage, depth_msg: ROSImage
    ) -> None:
        data = InputData(
            camera_intrinsics=self._process_intrinsics(info_msg),
            color=self._process_color(color_msg),
            depth=self._process_depth(depth_msg),
            pose=self._process_pose(transform_msg),
        )
        time = RosTime.from_msg(color_msg.header.stamp)
        self.input = (
            InputDataStamped(time_sec=time.seconds_nanoseconds()[0] + time.seconds_nanoseconds()[1] / 1e9, data=data),
            time,
        )

    def _process_intrinsics(self, info_msg: CameraInfo) -> Float[np.ndarray, "3 3"]:
        k = np.array(info_msg.k).reshape(3, 3)
        width = info_msg.width
        height = info_msg.height

        if self.rotations_image_clockwise in [-1, 1]:
            unrotated_input_height = self.scene.config.input_image_width
            unrotated_input_width = self.scene.config.input_image_height
        else:
            unrotated_input_height = self.scene.config.input_image_height
            unrotated_input_width = self.scene.config.input_image_width
        height_downsample_ratio = unrotated_input_height / height
        width_downsample_ratio = unrotated_input_width / width
        k = scale_intrinsics(k, height_downsample_ratio, width_downsample_ratio)

        if self.rotations_image_clockwise in [-1, 1]:
            k[(0, 0)], k[(1, 1)] = k[(1, 1)], k[(0, 0)]
            k[(0, 2)], k[(1, 2)] = k[(1, 2)], k[(0, 2)]

        return k.astype(np.float32)

    def _process_color(self, color_msg: ROSImage) -> UInt8[np.ndarray, "N M 3"]:
        if color_msg.encoding not in ["rgb8", "bgr8"]:
            self.get_logger().error(
                msg := f"Received color image with currently unsupported encoding '{color_msg.encoding}'."
            )
            raise ValueError(msg)
        color = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(color_msg.height, color_msg.width, 3)
        if self.rotations_image_clockwise in [-1, 1]:
            color = cv2.resize(
                color,
                (self.scene.config.input_image_height, self.scene.config.input_image_width),
                interpolation=cv2.INTER_LINEAR,
            )
            color = np.rot90(color, -self.rotations_image_clockwise)
        else:
            color = cv2.resize(
                color,
                (self.scene.config.input_image_width, self.scene.config.input_image_height),
                interpolation=cv2.INTER_LINEAR,
            )
        if color_msg.encoding == "bgr8":
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = color.astype(np.uint8)
        return color

    def _process_depth(self, depth_msg: ROSImage) -> Float[np.ndarray, "N M 1"]:
        if depth_msg.encoding == "32FC1":
            depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width)
        else:
            depth = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(depth_msg.height, depth_msg.width)
        if self.rotations_image_clockwise in [-1, 1]:
            depth = cv2.resize(
                depth.astype(float),
                (self.scene.config.input_image_height, self.scene.config.input_image_width),
                interpolation=cv2.INTER_NEAREST,
            )
            depth = np.rot90(depth, -self.rotations_image_clockwise)
        else:
            depth = cv2.resize(
                depth.astype(float),
                (self.scene.config.input_image_width, self.scene.config.input_image_height),
                interpolation=cv2.INTER_NEAREST,
            )
        depth = depth.astype(float) / self.camera_depth_scale_to_m
        depth = np.expand_dims(depth, -1)
        depth[~np.isfinite(depth)] = 0.0
        return depth

    def _process_pose(self, transform_msg: TransformStamped) -> Float[np.ndarray, "4 4"]:
        pose = np.eye(4, dtype=float)
        pose[:3, :3] = Rotation.from_quat(
            [
                transform_msg.transform.rotation.x,
                transform_msg.transform.rotation.y,
                transform_msg.transform.rotation.z,
                transform_msg.transform.rotation.w,
            ]
        ).as_matrix()
        # HACK: incoming transform's local axes are (X backward, Y left, Z down) instead of the
        # optical-frame convention (X right, Y down, Z forward) the image-rotation fix below
        # assumes. Verify visually (e.g. in rerun) and adjust/remove once the upstream frame
        # convention is corrected.
        axis_fix = np.array(
            [
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        axis_fix = np.eye(3)
        pose[:3, :3] = pose[:3, :3] @ axis_fix
        pose[:3, 3] = np.array(
            [
                transform_msg.transform.translation.x,
                transform_msg.transform.translation.y,
                transform_msg.transform.translation.z,
            ]
        )
        if self.rotations_image_clockwise:
            image_rotation = np.eye(4)
            image_rotation[:3, :3] = Rotation.from_euler(
                "z", -90 * self.rotations_image_clockwise, degrees=True
            ).as_matrix()
            pose = pose @ image_rotation
        return pose

    def _publish_scene_objects_pcd(self, objects: Sequence[SceneObject[GeometryType]], time: RosTime) -> None:
        if self.publisher_objects_pcd is None:
            return
        if not all(isinstance(obj.geometry, PointCloud | TensorPointCloud) for obj in objects):
            self.get_logger().error("ROS publishing of other geometries than PointClouds not implemented")
            return
        num_points = sum(len(obj.geometry) for obj in objects)
        structured_array = np.zeros(
            num_points,
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("r", np.uint32),
                ("g", np.uint32),
                ("b", np.uint32),
                ("s", np.uint32),
            ],
        )
        offset = 0
        for obj in objects:
            num_points_obj = len(obj.geometry)
            obj_points = obj.geometry.numpy_points()
            obj_colors = (obj.geometry.numpy_colors() * 255).astype(np.uint32)
            structured_array["x"][offset : offset + num_points_obj] = obj_points[:, 0]
            structured_array["y"][offset : offset + num_points_obj] = obj_points[:, 1]
            structured_array["z"][offset : offset + num_points_obj] = obj_points[:, 2]
            structured_array["r"][offset : offset + num_points_obj] = obj_colors[:, 0]
            structured_array["g"][offset : offset + num_points_obj] = obj_colors[:, 1]
            structured_array["b"][offset : offset + num_points_obj] = obj_colors[:, 2]
            structured_array["s"][offset : offset + num_points_obj] = np.array(
                obj.stationarity_confidence * 255
            ).astype(np.uint32)
            offset += num_points_obj
        structured_array = merge_rgb_fields(structured_array)
        msg = array_to_pointcloud2(structured_array, frame_id=self.global_frame, stamp=RosTime.to_msg(time))
        self.publisher_objects_pcd.publish(msg)

    def _publish_scene_objects_description(self, objects: Iterable[SceneObject[GeometryType]], time: RosTime) -> None:
        if self.publisher_objects_descr is None:
            return
        msg = ObjectDetectionArray()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = RosTime.to_msg(time)

        for obj in objects:
            obj_msg = ObjectDetection()
            obj_msg.label = obj.class_name
            obj_msg.statonarity_score = obj.stationarity_confidence
            pos = obj.geometry.get_center()
            obj_msg.position.x = float(pos[0])
            obj_msg.position.y = float(pos[1])
            obj_msg.position.z = float(pos[2])
            t = obj.last_observed_time()
            if t < 0:
                obj_msg.last_time_seen = RosTime().to_msg()
            else:
                obj_msg.last_time_seen = RosTime(seconds=int(t), nanoseconds=int((t - int(t)) * 1e9)).to_msg()
            obj_msg.num_detections = obj.number_of_observations()
            msg.objects.append(obj_msg)
        self.publisher_objects_descr.publish(msg)

    def _publish_background_octomap(self, tree: Union[OcTree, ColorOcTree], time: RosTime) -> None:
        if self.publisher_background is None:
            return
        msg = Octomap()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = RosTime.to_msg(time)
        msg.id = tree.getTreeType()
        msg.resolution = float(tree.getResolution())
        # Binary format is compact but can't encode color; use the full format for ColorOcTree.
        if isinstance(tree, ColorOcTree):
            msg.binary = False
            raw = tree.write()
        else:
            msg.binary = True
            raw = tree.writeBinary()
        # data is int8[]; reinterpret (not naively list) the raw bytes as signed int8.
        binary_payload = _strip_octomap_file_header(raw)
        msg.data = np.frombuffer(binary_payload, dtype=np.uint8).astype(np.int8).tolist()
        self.publisher_background.publish(msg)

    def _publish_occupancy_map(self, occupancy_grid: OccupancyGrid, time: RosTime) -> None:
        if self.publisher_occupancy_map is None:
            return
        msg = self._convert_occupancy_to_ros(occupancy_grid, time, scale_positive_to_100=True)
        self.publisher_occupancy_map.publish(msg)

    def _publish_exploration_map(self, exploration_grid: Union[OccupancyGrid, SparseVoxelGrid], time: RosTime) -> None:
        if self.publisher_exploration_map is None:
            return
        if isinstance(exploration_grid, SparseVoxelGrid):
            msg = self._convert_sparse_voxel_grid_to_pointcloud2(exploration_grid, time)
        else:
            msg = self._convert_occupancy_to_ros(exploration_grid, time)
        self.publisher_exploration_map.publish(msg)

    def _convert_sparse_voxel_grid_to_pointcloud2(self, grid: SparseVoxelGrid, time: RosTime) -> PointCloud2:
        # values are int8 priority scores in [-128, 127]; remap to a [0, 1] probability field "p".
        positions = grid.origin + (grid.coords + 0.5) * grid.resolution
        probabilities = (grid.values.astype(np.float32) + 128.0) / 255.0
        return xyz_field_to_pointcloud2(
            positions.astype(np.float32), "p", probabilities, stamp=RosTime.to_msg(time), frame_id=self.global_frame
        )

    def _convert_occupancy_to_ros(
        self, grid: OccupancyGrid, time: RosTime, scale_positive_to_100: bool = False
    ) -> ROSOccupancyGrid:
        msg = ROSOccupancyGrid()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = RosTime.to_msg(time)
        msg.info.resolution = float(grid.resolution)
        msg.info.width = int(grid.grid.shape[0])
        msg.info.height = int(grid.grid.shape[1])
        msg.info.origin.position.x = float(grid.origin[0])
        msg.info.origin.position.y = float(grid.origin[1])
        msg.info.origin.position.z = 0.0
        data = grid.grid.T.flatten(order="C").astype(np.int8)
        if scale_positive_to_100:
            data[data > 0] *= 100
        msg.data = data.tolist()
        return msg

    def _enable_exploration_map_callback(
        self, request: EnableHeatmap.Request, response: EnableHeatmap.Response
    ) -> EnableHeatmap.Response:
        response.success = True
        if self.exploration_map.query is not None and self.exploration_map.query == request.input:
            self.get_logger().info(f"Received unchanged exploration map query: {self.exploration_map.query}")
            return response
        self.exploration_map.set_query(request.input)
        self.get_logger().info(f"Set exploration map query to: {self.exploration_map.query}")
        return response

    def run(self) -> None:
        self.get_logger().info(f"Node '{self.get_name()}' started.")

        # Keep track of two times: when replaying old data the current time does not match the messages times
        last_update_own_clock: Optional[RosTime] = None
        last_update_msg_clock: Optional[RosTime] = None

        published_object_pcds_outdated: bool = False
        published_object_descr_outdated: bool = False
        published_background_outdated: bool = False
        published_occupancy_map_outdated: bool = False
        published_exploration_map_outdated: bool = False
        last_object_pcds_published_time: RosTime = self.get_clock().now()
        last_object_descr_published_time: RosTime = self.get_clock().now()
        last_background_published_time: RosTime = self.get_clock().now()
        last_occupancy_map_published_time: RosTime = self.get_clock().now()
        last_exploration_map_published_time: RosTime = self.get_clock().now()

        latest_robot_position: Optional[tuple[float, float]] = None

        def set_outdated() -> None:
            nonlocal published_object_pcds_outdated
            nonlocal published_object_descr_outdated
            nonlocal published_background_outdated
            nonlocal published_occupancy_map_outdated
            nonlocal published_exploration_map_outdated
            published_object_pcds_outdated = True
            published_object_descr_outdated = True
            published_background_outdated = True
            published_occupancy_map_outdated = True
            published_exploration_map_outdated = True

        def below_max_rate(now: RosTime, last_time: Optional[RosTime]) -> bool:
            if last_time is None:
                return True
            return (now - last_time).nanoseconds / 1e9 >= 1 / self.max_scene_update_rate_hz

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)
            self.tf_filter.update()
            now = self.get_clock().now()
            if below_max_rate(now, last_update_own_clock) and self.input is not None:
                last_update_own_clock = now
                last_update_msg_clock = self.input[1]
                set_outdated()
                self.get_logger().debug("Stepping perception scene")
                self.scene.step(self.input[0])
                self.get_logger().debug("Perception scene step complete")
                if self.input[0].data is not None:
                    latest_robot_position = tuple(self.input[0].data.pose[:2, 3])
                self.input = None

            if last_update_own_clock is not None and last_update_msg_clock is not None:
                time_since_last_update = now - last_update_own_clock
                if time_since_last_update.nanoseconds / 1e9 > 1 / self.min_scene_update_rate_hz:
                    last_update_msg_clock += time_since_last_update
                    last_update_own_clock = now
                    set_outdated()
                    self.get_logger().debug("Stepping perception scene (without new input data)")
                    self.scene.step(InputDataStamped(time_sec=last_update_msg_clock.nanoseconds / 1e9, data=None))
                    self.get_logger().debug("Perception scene step complete (without new input data)")

            if (
                self.publishing_rate_pcd_objects != 0
                and published_object_pcds_outdated
                and (
                    self.publishing_rate_pcd_objects == -1
                    or (now - last_object_pcds_published_time).nanoseconds / 1e9 > 1 / self.publishing_rate_pcd_objects
                )
            ):
                self._publish_scene_objects_pcd(tuple(self.scene.get_objects()), time=now)
                last_object_pcds_published_time = self.get_clock().now()
                published_object_pcds_outdated = False

            if (
                self.publishing_rate_descr_objects != 0
                and published_object_descr_outdated
                and (
                    self.publishing_rate_descr_objects == -1
                    or (now - last_object_descr_published_time).nanoseconds / 1e9
                    > 1 / self.publishing_rate_descr_objects
                )
            ):
                self._publish_scene_objects_description(self.scene.get_objects(), time=now)
                last_object_descr_published_time = self.get_clock().now()
                published_object_descr_outdated = False

            if (
                self.publishing_rate_background != 0
                and published_background_outdated
                and (
                    self.publishing_rate_background == -1
                    or (now - last_background_published_time).nanoseconds / 1e9 > 1 / self.publishing_rate_background
                )
            ):
                self._publish_background_octomap(self.scene.get_background(), time=now)
                last_background_published_time = self.get_clock().now()
                published_background_outdated = False

            if (
                self.occupancy_map is not None
                and self.publishing_rate_occupancy_map != 0
                and published_occupancy_map_outdated
                and (
                    self.publishing_rate_occupancy_map == -1
                    or (now - last_occupancy_map_published_time).nanoseconds / 1e9
                    > 1 / self.publishing_rate_occupancy_map
                )
            ):
                self.get_logger().debug("Publishing occupancy map")
                if (
                    occupancy_grid := self.occupancy_map.get_occupancy_grid(included_position=latest_robot_position)
                ) is not None:
                    self._publish_occupancy_map(occupancy_grid, time=now)
                last_occupancy_map_published_time = self.get_clock().now()
                published_occupancy_map_outdated = False
                self.get_logger().debug("Published occupancy map")

            if (
                self.exploration_map is not None
                and self.publishing_rate_exploration_map != 0
                and published_exploration_map_outdated
                and (
                    self.publishing_rate_exploration_map == -1
                    or (now - last_exploration_map_published_time).nanoseconds / 1e9
                    > 1 / self.publishing_rate_exploration_map
                )
            ):
                self.get_logger().debug("Publishing exploration map")
                if (exploration_grid := self.exploration_map.get(list(self.scene.get_objects()))) is not None:
                    self._publish_exploration_map(exploration_grid, time=now)
                last_exploration_map_published_time = self.get_clock().now()
                published_exploration_map_outdated = False
                self.get_logger().debug("Published exploration map")


def main(args=None):  # noqa: D103
    rclpy.init(args=args)
    node = PerceiveSemantixNode()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
