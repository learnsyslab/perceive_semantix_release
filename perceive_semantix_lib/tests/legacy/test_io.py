from pathlib import Path

import pytest

from perceive_semantix_lib.core.utils.disk_storage import DiskStorage


@pytest.fixture
def scene_object_file() -> Path:  # noqa: D103
    return Path(__file__).parent.parent.parent.parent / "example_data" / "premapped_scenes" / "scene_office_legacy.pkl"


def test_load_legacy_scene(scene_object_file):
    """Test loading a legacy scene from disk storage."""
    object_tracker, background_tracker, last_frame_time_sec, previous_camera_pose = DiskStorage.load_scene(
        scene_object_file
    )
