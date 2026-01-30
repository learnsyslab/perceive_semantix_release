# perceive_semantix_interfaces/msgs/ObjectDetection.msg

# Label of the detected object (e.g., "car", "person")
string label

# Stationarity score of the object (e.g., a value between 0 and 1)
float32 statonarity_score

# Position of the object in 3D space
geometry_msgs/Point position

# Last time the object was seen (in seconds since epoch)
builtin_interfaces/Time last_time_seen

# Number of detections (how often the object was detected)
int32 num_detections