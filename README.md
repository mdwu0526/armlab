# 6-DOF Serial Link Robotic Manipulator

A project for ROB 550: Robotics Systems Lab course taught in University of Michigan, Ann Arbor. An overview of this project:

##### Acting
- 6-DOF rigid-body coordinate transforms using homogeneous coordinate transforms
- Forward kinematics modeling of a manipulator
- Inverse kinematics modeling of a manipulator
- Grasping

##### Perception 
- 3D image/workspace calibration
- Object detection with OpenCV
- Depth camera sensors

##### Reasoning
- Path planning & path smoothing
- State machines

The whole pipeline can be explained by the following figure: 
<img src="pipeline.PNG" width="500">

### Running the code

1. `./launch_armlab.sh`
1. `./control_station.py -c config/rx200_dh.csv`

### Directories and Codebase 

| Files                 | Description   |
| -------------         | -------------  |
| `control_station.py`    | Main program.  Sets up the GUI and how botlab interacts with the user.|
| `rxarm.py`             | Implements the Rxarm and Joint class |
| `test/kinematics.py`   | Script to verify FK and IK and test functions.|
| `config/rx200_config.csv` | This file sets up the Denavit Hartenberg parameters for the RX200 arm.|
| `block_detect.py` | Implements the block detector which utilizes openCV.|
| `state_machine.py` | Implements the StateMachine class|
| `kinematics.py`| Implements functions for forward and inverse kinematics.|
| `armlab_gui.ui`| This file defines the GUI interface, created using QtCreator. To compile a new u<span>i.p</span>y file run, `pyuic4 mainWindow.ui -o ui.py`|
| `ui.py`| Output from QtCreator with GUI implementation in Python.|
| `utils/camera_cal.py`| Standalone program to generate camera distortion parameters and camera intrinsic matrix after calibrating with a checkerboard.|

### Lab setup

<img src="lab_setup.jpg" width="500">

### Teach 'n Play (on an Operations board)

<img src="teachnplay.gif" width="500">

### Collaborators
[Max Wu](https://www.linkedin.com/in/maxwu0526/), [Ashwin Saxena](https://www.linkedin.com/in/ashwin-s-6aa596169/) and [Ian Stewart](https://www.linkedin.com/in/icstewar/).


