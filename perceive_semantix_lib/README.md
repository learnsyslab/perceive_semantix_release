# perceive_semantix_lib

[![PyPI - Version](https://img.shields.io/pypi/v/perceive_semantix_lib.svg)](https://pypi.org/project/perceive_semantix_lib)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/perceive_semantix_lib.svg)](https://pypi.org/project/perceive_semantix_lib)

-----

## Table of Contents

- [Installation](#installation)
- [License](#license)
- [Code Outline](#code-outline)

## Installation

```console
pip install perceive_semantix_lib
```

## License

`perceive_semantix_lib` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Code Outline

### `Scene` Class

```python
class Scene:
    Attributes:
        Config
        ObjectDetector
        ObjectTracker
        |- dict[UUID, SceneObjects]
        |- active_objects: list[UUID]
        |- inactive_objects: list[UUID]
        ...

    Methods:
        step(self, posed_rgbd: InputDataStamped)
        ...
```

The `Scene` class represents the core world model, maintaining and updating object hypotheses as new frames are processed.

---

### `Scene.step(posed_rgbd: InputDataStamped)`

Processes a new input frame (`posed_rgbd`) and updates the scene accordingly.
The main steps are:

1. **Detection**

    Detect visible objects in the current frame using `ObjectDetector`, producing [`detections`](src/perceive_semantix_lib/core/scene_belief.py#L294).

2. **Projection**

   Project existing scene objects onto the camera plane to determine which ones are expected to be visible ([`SceneObject.project_onto_camera`](src/perceive_semantix_lib/core/scene_belief.py#L309)).

3. **Matching**

   Establish correspondences between detections and existing objects:
   - Initialize adjacency structures ([`ObjectDetectionAdjacency`](src/perceive_semantix_lib/core/scene_belief.py#L347)).
   - Match detections to objects assuming static positions ([`_match_detections_to_objects`](src/perceive_semantix_lib/core/scene_belief.py#L350)).
   - Match remaining detections allowing object movement ([`match_detections_to_objects_semantically_greedy_icp`](src/perceive_semantix_lib/core/scene_belief.py#L378)).
   - Match existing objects to each other ([`match_objects_to_objects`](src/perceive_semantix_lib/core/scene_belief.py#L396)).
   - Attempt re-identification of inactive or missing objects ([`match_objects_to_objects`](src/perceive_semantix_lib/core/scene_belief.py#L415)).

4. **Merging & Creation**

   Merge matched objects and instantiate new ones as needed, based on the populated adjacency structure ([`merge_from_adjacency`](src/perceive_semantix_lib/core/scene_belief.py#L423)).

5. **Post-processing**

   Log the outcome and, if necessary, filter the scene to reduce noise and maintain consistency.

---

### Extensions

#### [`OccupancyMap`](src/perceive_semantix_lib/grid_maps/occupancy_map.py)

Takes in a `Scene` to compute a 2D occupancy grid.

#### [`ExplorationPriorityMap`](src/perceive_semantix_lib/grid_maps/exploration_priority_map.py)

Implements the exploration priority map as described in https://arxiv.org/abs/2509.19851 based on a list of active objects provided by `Scene.get_objects()`. 
