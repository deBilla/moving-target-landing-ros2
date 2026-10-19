# ROS 2 Jazzy + OpenCV (with ArUco) for the moving-target landing simulation.
# Headless: the simulator renders the downward camera itself, so no Gazebo/GPU is needed.
FROM ros:jazzy-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-opencv \
        python3-numpy \
        python3-scipy \
        python3-matplotlib \
        python3-pytest \
        ros-jazzy-cv-bridge \
        ros-jazzy-rosbag2-storage-mcap \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
COPY ros2_ws/src /ws/src
RUN . /opt/ros/jazzy/setup.sh && colcon build --symlink-install

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
