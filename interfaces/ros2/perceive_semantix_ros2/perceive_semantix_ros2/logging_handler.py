import logging

from rclpy.node import Node


class ROS2LoggingHandler(logging.Handler):
    def __init__(self, node: Node):
        super().__init__()
        self.node = node

    def emit(self, record):
        # Convert Python log levels to ROS 2 log levels
        if record.levelno >= logging.CRITICAL:
            self.node.get_logger().fatal(record.getMessage())
        elif record.levelno >= logging.ERROR:
            self.node.get_logger().error(record.getMessage())
        elif record.levelno >= logging.WARNING:
            self.node.get_logger().warn(record.getMessage())
        elif record.levelno >= logging.INFO:
            self.node.get_logger().info(record.getMessage())
        else:
            self.node.get_logger().debug(record.getMessage())
