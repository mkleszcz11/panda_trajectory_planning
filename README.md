# panda_trajectory_planning

## Usefull comands

### For the real robot
`roslaunch panda_moveit_config franka_control.launch robot_ip:=172.16.0.3 load_gripper:=1`

Go to home pos:
``

### For simulation
Run franka panda simulation (moveit and gazebo)
```bash
roslaunch panda_moveit_config demo_gazebo.launch
```
Alternatively you can run a custom setup (table included):
```bash
roslaunch franka_test custom_setup.launch
```

### ?
First:
roslaunch franka_gazebo panda.launch

Then:
rosservice call /controller_manager/load_controller "name: 'position_joint_trajectory_controller'"

### Move to home position
```bash
rosrun 
```

### Get current pose


