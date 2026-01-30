from setuptools import find_packages, setup

package_name = "perceive_semantix_ros2"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/object_classes", ["object_classes.txt"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Benjamin Bogenberger",
    maintainer_email="benjamin.bogenberger@tum.de",
    description="ROS2 package for the perceive semantix library",
    license="MIT",
    entry_points={
        "console_scripts": ["perceive_semantix_node = " + package_name + ".main:main"],
    },
)
