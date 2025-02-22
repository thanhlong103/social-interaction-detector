# Human Group Interaction Recognition using PCA and Interest Point Estimation

## Overview
Understanding human group interactions is essential for robots navigating shared spaces, allowing them to move safely and naturally. Traditional motion planning methods focus on obstacle avoidance but lack the ability to interpret social cues.

This project presents a framework for **Human Group Interaction Recognition** using **Principal Component Analysis (PCA) and Interest Point Estimation**. The system uses a **monocular RGB-D camera** to estimate **3D human positions**, resolve **front/back ambiguity**, and identify **interaction zones** based on spatial formations. PCA determines dominant interaction directions, while Interest Point Estimation enhances engagement area detection within groups.

## Features
- **Real-time group interaction recognition** using an RGB-D camera.
- **PCA and Interest Point Estimation** to model engagement zones.
- **ROS 2-based implementation** for robotic applications.
- **High accuracy and efficiency** in dynamic human-populated environments.

## Project Structure
```
├── data/                     # Contains collected test data
├── rviz2/                    # RViz2 configuration for visualization
└── socin_robot_ws/           # Main workspace containing ROS2 packages
    ├── fusing_people/        # ROS2 package for group detection
    ├── people_msgs/          # Custom ROS2 message for human information (Position, Orientation, ID)
    └── vision_people_tracker/# MoveNet model and ROS2 node for human skeleton detection
```

## Prerequisites
Ensure you have the following installed:
- **Ubuntu 20.04**
- **ROS 2 Foxy**
- **Colcon** (ROS 2 build system)
- **Python 3**
- **OpenCV, NumPy, TensorFlow, Pyrealsense2, Shapely and Scikit-learn** (for vision-based tracking)

## Installation
1. Clone the repository:
   ```bash
   cd ~
   git clone https://github.com/thanhlong103/social-interaction-detector
   ```
2. Navigate to the ROS 2 workspace:
   ```bash
   cd social-interaction-detector/socin_robot_ws
   ```
3. Build the ROS 2 workspace:
   ```bash
   colcon build --symlink-install
   ```
4. Set the correct permissions:
   ```bash
   cd ..
   sudo chmod +x ./run.sh
   ```

## Usage
1. **Monitor the human skeleton detection topic:**
   ```bash
   cd ~/social-interaction/detector
   ./run.sh
   ```
2. **Monitor the human pose detection topic:**
   ```bash
   ros2 topic echo /people_vision
   ```

## Expected Output
- The system will detect human positions and orientations.
- It will estimate interaction zones and visualize engagement areas in **RViz2**.
- ROS 2 topics will publish human tracking data.

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

## License
This project is open-source under the **MIT License**.

## Contact
For inquiries, please reach out us via GitHub issues or email long.nguyen.210085@student.fulbright.edu.vn.

---

*By integrating social perception into robotic systems, this work bridges the gap between physical safety and social intelligence, paving the way for more adaptive, human-aware robots.*
