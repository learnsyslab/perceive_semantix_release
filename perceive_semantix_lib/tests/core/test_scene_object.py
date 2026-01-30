import numpy as np
import pytest
import torch
from open3d import utility

from perceive_semantix_lib.core.geometry.point_cloud import PointCloud
from perceive_semantix_lib.core.scene_object import SceneObject


@pytest.fixture
def example_pointcloud_scene_object():
    """Create an example SceneObject with random points and visual feature."""
    point_cloud = PointCloud()
    point_cloud.pcd.points = utility.Vector3dVector(np.random.rand(100, 3).astype(np.float64))
    visual_feature = torch.rand(128)
    scene_object = SceneObject(
        class_id=1,
        class_name="example",
        class_confidence=0.9,
        observation_time=1234567890.0,
        goemetry=point_cloud,
        bounding_box=point_cloud.get_bounding_box(),
        visual_feature=visual_feature,
    )
    return scene_object


def test_scene_object_serialization(example_pointcloud_scene_object):
    """Test serialization and deserialization of SceneObject."""
    scene_object = example_pointcloud_scene_object
    obj_dict = scene_object.to_dict()

    loaded_scene_object = SceneObject.from_dict(obj_dict, device=str(scene_object.visual_feature.device))
    assert scene_object.class_id == loaded_scene_object.class_id
    assert scene_object.class_name == loaded_scene_object.class_name
    assert scene_object.class_confidence == loaded_scene_object.class_confidence
    assert scene_object.observation_times == loaded_scene_object.observation_times
    assert np.allclose(
        np.asarray(scene_object.geometry.pcd.points),
        np.asarray(loaded_scene_object.geometry.pcd.points),
    )
    assert torch.allclose(
        scene_object.visual_feature,
        loaded_scene_object.visual_feature,
    )
