cmake_minimum_required(VERSION 2.8.3)
project(learning_machines_task1)

# Specify rospy as dependency
find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_python_setup()
catkin_package()
