# perceive semantix interface

This packages here is intended to be used as a dependency within ROS2 workspaces which are managed by [Pixi](https://pixi.sh/dev/tutorials/ros2/).

To add the dependency

1. Enable `pixi-build` preview feature by adding to your `pixi.toml`
    ```
    [workspace]
    ...
    preview = ["pixi-build"]
    ```
    or to your `pyproject.toml`
    ```
    [tool.pixi.workspace]
    ...
    preview = ["pixi-build"]
    ```
2. Add the dependency by running
    ```
    pixi add --git https://github.com/utiasDSL/perceive_semantix.git <pkg_name> --subdir src/<pkg_name>
    ```
    where `<pkg_name>` is for example "`perceive_semantix_interfaces`". This will add the following line to your `pixi.toml`/`pyproject.toml`
    ```
    perceive_semantix_interfaces = { git = "https://github.com/utiasDSL/perceive_semantix.git", subdirectory = "interfaces/ros2/perceive_semantix_interfaces" }
    ```
