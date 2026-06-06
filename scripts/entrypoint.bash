#!/usr/bin/env bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
source /root/catkin_ws/setup.bash

rosrun learning_machines_task1 learning_robobo_controller.py "$@"
