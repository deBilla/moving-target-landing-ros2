#!/bin/bash
# Run one closed-loop ROS 2 simulation in Docker.
#   docker/run_ros_sim.sh <scenario> <mode> [seed]
# Results land in ./results/ros_run_<scenario>_<mode>.json and ./results/frames_<scenario>_<mode>/
set -e
cd "$(dirname "$0")/.."
SCENARIO=${1:-straight}
MODE=${2:-corrected}
SEED=${3:-0}
# --network none + own domain id: parallel runs must not discover each other over DDS
DOMAIN=${ROS_DOMAIN_ID:-$(( (RANDOM % 100) + 1 ))}
docker run --rm --network none -e ROS_DOMAIN_ID=$DOMAIN -e ROS_LOCALHOST_ONLY=1 \
  -v "$PWD/ros2_ws/src:/ws/src" \
  -v "$PWD/results:/results" \
  mtl_landing:jazzy bash -c "
    cd /ws && colcon build --symlink-install >/dev/null && source install/setup.bash &&
    ros2 launch mtl_landing landing_sim.launch.py scenario:=$SCENARIO mode:=$MODE seed:=$SEED"
