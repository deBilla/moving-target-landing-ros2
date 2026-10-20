# Slim image for the interactive sim lab. The simulator core is plain Python
# (numpy + OpenCV with ArUco), so this doesn't need ROS.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/ros2_ws/src/mtl_landing LAB_PUBLIC=1 PORT=8765

RUN pip install --no-cache-dir "numpy==2.2.6" "opencv-contrib-python-headless==4.10.0.84" \
    && useradd --uid 10001 --no-create-home lab

WORKDIR /app
COPY ros2_ws/src/mtl_landing/mtl_landing ros2_ws/src/mtl_landing/mtl_landing
COPY tools/sim_lab tools/sim_lab

USER 10001
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/health', timeout=2)"
CMD ["python", "tools/sim_lab/server.py"]
