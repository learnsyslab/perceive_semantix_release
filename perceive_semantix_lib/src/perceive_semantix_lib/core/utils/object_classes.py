import json
from pathlib import Path

import numpy as np


class ObjectClasses:
    """Manages object classes and their associated colors, allowing for exclusion of background classes.

    This class facilitates the creation or loading of a color map from a specified file containing
    class names. It also manages background classes based on configuration, allowing for their
    inclusion or exclusion.

    Attributes:
        classes_file_path (str): Path to the file containing class names, one per line.

    Usage:
        obj_classes = ObjectClasses(classes_file_path, skip_bg=True)
        model.set_classes(obj_classes.get_classes_arr())
        some_class_color = obj_classes.get_class_color(index or class_name)

    """

    def __init__(self, classes_file_path, bg_classes, skip_bg):
        self.classes_file_path = Path(classes_file_path)
        self.bg_classes = bg_classes
        self.skip_bg = skip_bg
        self.classes, self.class_to_color = self._load_or_create_colors()

    def _load_or_create_colors(self):
        with open(self.classes_file_path, "r") as f:
            all_classes = [cls.strip() for cls in f.readlines()]

        # Filter classes based on the skip_bg parameter
        if self.skip_bg:
            classes = [cls for cls in all_classes if cls not in self.bg_classes]
        else:
            classes = all_classes

        colors_file_path = self.classes_file_path.parent / f"{self.classes_file_path.stem}_colors.json"
        if colors_file_path.exists():
            with open(colors_file_path, "r") as f:
                class_to_color = json.load(f)
            # Ensure color map only includes relevant classes
            class_to_color = {cls: class_to_color[cls] for cls in classes if cls in class_to_color}
        else:
            class_to_color = {class_name: list(np.random.rand(3).tolist()) for class_name in classes}
            with open(colors_file_path, "w") as f:
                json.dump(class_to_color, f)

        return classes, class_to_color

    def __getitem__(self, i: int):
        """Retrieve the i-th class name."""
        return self.classes[i]

    def get_classes_arr(self):
        """Return the list of class names, excluding background classes if configured to do so."""
        return self.classes

    def get_bg_classes_arr(self):
        """Return the list of background class names, if configured to do so."""
        return self.bg_classes

    def get_class_color(self, key):
        """Retrieve the color associated with a given class name or index.

        Args:
            key (int or str): The index or name of the class.

        Returns:
            list: The color (RGB values) associated with the class.

        """
        if isinstance(key, int):
            if key < 0 or key >= len(self.classes):
                raise IndexError("Class index out of range.")
            class_name = self.classes[key]
        elif isinstance(key, str):
            class_name = key
            if class_name not in self.classes:
                raise ValueError(f"{class_name} is not a valid class name.")
        else:
            raise ValueError("Key must be an integer index or a string class name.")
        # Default color for undefined classes
        return self.class_to_color.get(class_name, [0, 0, 0])

    def get_class_color_dict_by_index(self):
        """Return a dictionary of class colors, just like self.class_to_color, but indexed by class index."""
        return {str(i): self.get_class_color(i) for i in range(len(self.classes))}

    @classmethod
    def create_sentinel(cls) -> "ObjectClasses":
        return _ObjectClassesSentinel()


class _ObjectClassesSentinel(ObjectClasses):
    """Sentinel that raises on any attribute access before initialization."""

    def __init__(self):
        pass

    def __getattribute__(self, name: str):
        # Allow calling magic methods like __class__. This is required for runtime typechecking.
        if name.startswith("__"):
            return object.__getattribute__(self, name)
        raise RuntimeError(f"Attempted to access '{name}' on ObjectClasses before it was initialized.")


# module-level object to be initalized in objected_detector.py. If accessed before the
# initialization, the sentinal object will raise a RuntimeError
OBJECT_CLASSES: ObjectClasses = ObjectClasses.create_sentinel()


def init_global_object_classes(classes: ObjectClasses) -> None:
    """Initialize the global object class list."""
    global OBJECT_CLASSES
    OBJECT_CLASSES = classes


def get_global_object_classes() -> ObjectClasses:
    """Get the global object classes."""
    global OBJECT_CLASSES
    return OBJECT_CLASSES
