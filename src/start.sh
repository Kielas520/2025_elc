#!/bin/bash
cd LCD-show/
echo "520" | sudo -S ./MPI5001-show
echo "520" | sudo -S chmod 777 /home/sunrise/2025_elc/src/main.py
/usr/bin/python3 /home/sunrise/2025_elc/src/main.py

# v4l2-ctl --list-devices  # 列出所有视频设备
# sudo rdk-miniboot-update