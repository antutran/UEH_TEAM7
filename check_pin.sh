#!/bin/bash
docker exec crc_bringup bash -c 'source /opt/ros/humble/setup.bash && ros2 topic echo /voltage --once' | awk '/data:/ {v=$2; pct=(v-9.6)/(12.6-9.6)*100; if(pct<0)pct=0; if(pct>100)pct=100; printf "⚡ PIN: %.1f%% (%.2fV)\n", pct, v}'
