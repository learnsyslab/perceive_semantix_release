import array
from sys import byteorder as BYTEORDER

import numpy as np

from sensor_msgs.msg import PointCloud2, PointField

# Taken from
# https://github.com/Box-Robotics/ros2_numpy/blob/jazzy/ros2_numpy/point_cloud2.py

type_mappings = [
    (PointField.INT8, np.dtype("int8")),
    (PointField.UINT8, np.dtype("uint8")),
    (PointField.INT16, np.dtype("int16")),
    (PointField.UINT16, np.dtype("uint16")),
    (PointField.INT32, np.dtype("int32")),
    (PointField.UINT32, np.dtype("uint32")),
    (PointField.FLOAT32, np.dtype("float32")),
    (PointField.FLOAT64, np.dtype("float64")),
]
pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)


def dtype_to_fields(dtype):
    """Convert a numpy record datatype into a list of PointFields."""
    fields = []
    for field_name in dtype.names:
        np_field_type, field_offset = dtype.fields[field_name]
        pf = PointField()
        pf.name = field_name
        if np_field_type.subdtype:
            item_dtype, shape = np_field_type.subdtype
            pf.count = int(np.prod(shape))
            np_field_type = item_dtype
        else:
            pf.count = 1

        pf.datatype = nptype_to_pftype[np_field_type]
        pf.offset = field_offset
        fields.append(pf)
    return fields


def merge_rgb_fields(cloud_arr):
    """Take an array with named np.uint8 fields 'r', 'g', and 'b', and returns an array in which they have been merged into a single np.float32 'rgb' field.

    The first byte of this field is the 'r' uint8, the second is the 'g', uint8, and the third is the 'b' uint8. This is the way that pcl likes to handle RGB colors for some reason.
    """
    r = np.asarray(cloud_arr["r"], dtype=np.uint32)
    g = np.asarray(cloud_arr["g"], dtype=np.uint32)
    b = np.asarray(cloud_arr["b"], dtype=np.uint32)
    rgb_arr = np.array((r << 16) | (g << 8) | (b << 0), dtype=np.uint32)

    # not sure if there is a better way to do this. i'm changing the type of
    # the array from uint32 to float32, but i don't want any conversion to take
    # place -jdb
    rgb_arr.dtype = np.float32  # type: ignore

    # create a new array, without r, g, and b, but with rgb float32 field
    new_dtype = []
    for field_name in cloud_arr.dtype.names:
        field_type, field_offset = cloud_arr.dtype.fields[field_name]
        if field_name not in ("r", "g", "b"):
            new_dtype.append((field_name, field_type))
    new_dtype.append(("rgb", np.float32))
    new_cloud_arr = np.zeros(cloud_arr.shape, new_dtype)

    # fill in the new array
    for field_name in new_cloud_arr.dtype.names:  # type: ignore
        if field_name == "rgb":
            new_cloud_arr[field_name] = rgb_arr
        else:
            new_cloud_arr[field_name] = cloud_arr[field_name]

    return new_cloud_arr


def array_to_pointcloud2(cloud_arr, stamp=None, frame_id=None):
    """Convert a numpy record array to a sensor_msgs.msg.PointCloud2."""
    # make it 2d (even if height will be 1)
    cloud_arr = np.atleast_2d(cloud_arr)

    cloud_msg = PointCloud2()

    if stamp is not None:
        cloud_msg.header.stamp = stamp
    if frame_id is not None:
        cloud_msg.header.frame_id = frame_id
    cloud_msg.height = cloud_arr.shape[0]
    cloud_msg.width = cloud_arr.shape[1]
    cloud_msg.fields = dtype_to_fields(cloud_arr.dtype)
    cloud_msg.is_bigendian = BYTEORDER != "little"
    cloud_msg.point_step = cloud_arr.dtype.itemsize
    cloud_msg.row_step = cloud_msg.point_step * cloud_arr.shape[1]
    cloud_msg.is_dense = all([np.isfinite(cloud_arr[fname]).all() for fname in cloud_arr.dtype.names])

    # The PointCloud2.data setter will create an array.array object for you if you don't
    # provide it one directly. This causes very slow performance because it iterates
    # over each byte in python.
    # Here we create an array.array object using a memoryview, limiting copying and
    # increasing performance.
    memory_view = memoryview(cloud_arr)
    if memory_view.nbytes > 0:
        array_bytes = memory_view.cast("B")
    else:
        # Casting raises a TypeError if the array has no elements
        array_bytes = b""
    as_array = array.array("B")
    as_array.frombytes(array_bytes)
    cloud_msg.data = as_array
    return cloud_msg
