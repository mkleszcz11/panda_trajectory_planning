# panda_trajectory_planning

## Usefull comands

### For the real robot


### For simulation
Run franka panda simulation (moveit and gazebo)
```bash
roslaunch panda_moveit_config demo_gazebo.launch
```

First:
roslaunch franka_gazebo panda.launch

Then:
rosservice call /controller_manager/load_controller "name: 'position_joint_trajectory_controller'"
