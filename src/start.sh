#!/bin/bash
echo "520" | sudo -S chmod 777 /home/kielas/2025_elc/src/start.sh
echo "520" | sudo -S chmod 777 /home/kielas/2025_elc/src/main.py
v4l2-ctl --list-devices  # 列出所有视频设备
# sudo rdk-miniboot-update
/usr/bin/python3 /home/kielas/2025_elc/src/main.py