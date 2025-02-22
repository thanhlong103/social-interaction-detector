#!/bin/bash

#========== GROUP DETECTION ===========
(
  cd ~/social-interaction-detector/socin_robot_ws || exit
  source install/setup.bash
  ros2 run fusing_people fused_group
) &

#========== CAMERA HUMAN TRACKER ============
(
  cd ~/social-interaction-detector/socin_robot_ws || exit
  source install/setup.bash
  cd ~/social-interaction-detector/socin_robot_ws/src/vision_people_tracker/src 
  source tf/bin/activate
  python3 vision_people_tracker.py
) &

# Launch Rviz2
rviz2 -d ~/social-interaction-detector/rviz2/fused.rviz &

# Wait for all background processes to finish
wait
