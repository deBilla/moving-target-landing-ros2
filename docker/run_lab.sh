#!/bin/bash
# Interactive sim lab: every simulator / estimator / controller parameter on a slider.
#   docker/run_lab.sh            then open http://localhost:8765
set -e
cd "$(dirname "$0")/.."
PORT=${PORT:-8765}
echo "Sim lab: http://localhost:$PORT  (Ctrl+C to stop)"
docker run --rm -it -p "$PORT:8765" \
  -v "$PWD/ros2_ws/src:/ws/src" -v "$PWD/tools:/tools" \
  -e PYTHONPATH=/ws/src/mtl_landing \
  --entrypoint python3 mtl_landing:jazzy /tools/sim_lab/server.py
