---
id: installation-setup
title: "Installation and Setup Guide"
sidebar_position: 1
---

# Installation and Setup Guide

## Overview

This guide provides step-by-step instructions for setting up the complete Physical AI & Humanoid Robotics development environment. The setup includes ROS 2, NVIDIA Isaac Sim, Gazebo, Unity integration, and all necessary dependencies for developing and simulating humanoid robots.

**Important**: This setup requires significant computational resources. Please ensure your system meets the hardware requirements outlined in the [Hardware Requirements](./hardware-requirements.md) document before proceeding.

## Prerequisites

### System Requirements

- **OS**: Ubuntu 22.04 LTS (recommended)
- **CPU**: Intel i7-13700K or AMD Ryzen 9 7950X (or equivalent)
- **GPU**: NVIDIA RTX 4070 Ti (12GB VRAM) or higher (RTX 3090/4090 recommended)
- **RAM**: 64GB DDR5 (32GB minimum)
- **Storage**: 500GB SSD (1TB recommended)
- **Network**: Stable internet connection for downloads

### Pre-Installation Checklist

- [ ] Verify system meets hardware requirements
- [ ] Backup important data before system modifications
- [ ] Ensure administrator/root access
- [ ] Stable internet connection available
- [ ] Sufficient disk space (minimum 200GB free)

## Phase 1: System Preparation

### 1.1 Update System

First, update your Ubuntu system:

```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install basic utilities
sudo apt install -y wget curl git vim htop build-essential cmake
```

### 1.2 Install NVIDIA Drivers

For optimal performance with Isaac Sim and GPU acceleration:

```bash
# Check current driver
nvidia-smi

# Install latest NVIDIA drivers (skip if already installed)
sudo apt install nvidia-driver-535 -y

# Reboot after driver installation
sudo reboot
```

### 1.3 Install CUDA Toolkit

```bash
# Download CUDA toolkit (adjust version as needed)
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run

# Run installer (do NOT install driver if already installed)
sudo sh cuda_12.3.0_545.23.06_linux.run

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Phase 2: ROS 2 Installation

### 2.1 Install ROS 2 Humble Hawksbill

```bash
# Set locale
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Add ROS 2 apt repository
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe

# Add ROS 2 GPG key and repository
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 packages
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install ros-humble-ros-base
sudo apt install python3-ros-dev-tools
```

### 2.2 Install ROS 2 Dependencies

```bash
# Install additional ROS 2 packages
sudo apt install -y \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-gazebo-ros2-control \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-joint-state-broadcaster \
  ros-humble-robot-state-publisher \
  ros-humble-xacro \
  ros-humble-teleop-twist-keyboard \
  ros-humble-cv-bridge \
  ros-humble-tf2-tools \
  ros-humble-tf2-geometry-msgs \
  ros-humble-tf2-eigen \
  ros-humble-vision-opencv \
  ros-humble-image-transport \
  ros-humble-compressed-image-transport \
  ros-humble-image-pipeline \
  ros-humble-depth-image-proc \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup \
  ros-humble-nav2-simple-commander \
  ros-humble-geometry2 \
  ros-humble-tf-transformations \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool
```

### 2.3 Initialize rosdep

```bash
# Initialize rosdep
sudo rosdep init
rosdep update
```

## Phase 3: NVIDIA Isaac Installation

### 3.1 Install Isaac ROS Packages

```bash
# Create ROS 2 workspace
mkdir -p ~/isaac_ws/src
cd ~/isaac_ws

# Clone Isaac ROS packages
git clone -b humble https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git src/isaac_ros_common
git clone -b humble https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git src/isaac_ros_visual_slam
git clone -b humble https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag.git src/isaac_ros_apriltag
git clone -b humble https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation.git src/isaac_ros_pose_estimation
git clone -b humble https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_gxf.git src/isaac_ros_gxf
git clone -b humble https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline.git src/isaac_ros_image_pipeline
git clone -b humble https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_bezier_curve_generator.git src/isaac_ros_bezier_curve_generator
git clone -b humble https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_managed_nitros.git src/isaac_ros_managed_nitros
```

### 3.2 Install Isaac ROS Dependencies

```bash
cd ~/isaac_ws
rosdep install --from-paths src --ignore-src -r -y
```

### 3.3 Build Isaac ROS Packages

```bash
cd ~/isaac_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select \
  isaac_ros_common \
  isaac_ros_visual_slam \
  isaac_ros_apriltag \
  isaac_ros_pose_estimation \
  isaac_ros_gxf \
  gxf_isaac_ros_core \
  gxf_isaac_ros_messages \
  gxf_isaac_ros_types \
  gxf_isaac_ros_utilities \
  gxf_isaac_gems
```

## Phase 4: Isaac Sim Installation

### 4.1 Install Omniverse Launcher

```bash
# Download Omniverse Launcher
wget https://developer.download.nvidia.com/devzone/secure/omniverse/downloads/Omniverse_Launcher_Linux.dmg?XX_HASH_XX=1234567890abcdef -O Omniverse_Launcher_Linux.dmg

# Note: Isaac Sim is typically installed via Omniverse Launcher GUI
# For automated installation, you may need to register with NVIDIA Developer account
```

### 4.2 Alternative: Docker Installation

If using Docker for Isaac Sim:

```bash
# Install Docker
sudo apt install docker.io
sudo usermod -aG docker $USER

# Install nvidia-docker2
sudo apt install nvidia-docker2
sudo systemctl restart docker

# Pull Isaac Sim Docker image
docker pull nvcr.io/nvidia/isaac-sim:4.0.0
```

### 4.3 Isaac Sim Configuration

```bash
# Create Isaac Sim configuration directory
mkdir -p ~/.nvidia-omniverse
mkdir -p ~/Documents/Isaac-Sim-Configs

# Add Isaac Sim to PATH (if installed locally)
echo 'export ISAAC_SIM_PATH="$HOME/.local/share/ov/pkg/isaac_sim-4.0.0"' >> ~/.bashrc
echo 'export PYTHONPATH=${ISAAC_SIM_PATH}/python:${PYTHONPATH}' >> ~/.bashrc
source ~/.bashrc
```

## Phase 5: Gazebo Installation

### 5.1 Install Gazebo Garden

```bash
# Install Gazebo Garden
sudo apt install software-properties-common
sudo add-apt-repository ppa:ignitionrobotics/release-latest
sudo apt update
sudo apt install gz-garden

# Alternative: Install via binary packages
sudo apt install -y \
  gazebo \
  libgazebo-dev \
  ros-humble-gazebo-ros \
  ros-humble-gazebo-ros2-control \
  ros-humble-gazebo-plugins
```

### 5.2 Install Gazebo Models

```bash
# Create directory for Gazebo models
mkdir -p ~/.gazebo/models

# Download common models
cd ~/.gazebo/models
wget https://github.com/osrf/gazebo_models/archive/main.tar.gz
tar -xzf main.tar.gz
mv gazebo_models-main/* .
rm -rf gazebo_models-main main.tar.gz
```

## Phase 6: Unity Integration Setup

### 6.1 Install Unity Hub and Editor

```bash
# Download Unity Hub
wget https://public-cdn.cloud.unity3d.com/hub/prod/UnityHub.AppImage
chmod +x UnityHub.AppImage

# Run Unity Hub (GUI application)
./UnityHub.AppImage

# Install Unity 2022.3 LTS from Unity Hub
# Select "Desktop Game Development" module
# Select "Linux Build Support" module
```

### 6.2 Install ROS TCP Connector

```bash
# Create Unity project directory
mkdir -p ~/unity_projects/physical_ai_unity

# Clone ROS TCP Connector for Unity
cd ~/unity_projects/physical_ai_unity
git clone https://github.com/Unity-Technologies/ROS-TCP-Connector.git

# The ROS TCP Connector will be imported into Unity projects
```

### 6.3 Install Unity Robotics Hub

In Unity Hub:
1. Go to "Projects" tab
2. Click "Add" to create new project
3. Install "Unity Robotics Hub" from Package Manager
4. Import "Unity Robot Generator" and "Synthetic Data" packages

## Phase 7: Python Environment Setup

### 7.1 Create Virtual Environment

```bash
# Install Python tools
sudo apt install python3-pip python3-venv python3-dev

# Create virtual environment
python3 -m venv ~/physical_ai_env
source ~/physical_ai_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools
```

### 7.2 Install Python Dependencies

```bash
# Activate virtual environment
source ~/physical_ai_env/bin/activate

# Install core Python packages
pip install numpy scipy matplotlib pandas
pip install opencv-python open3d
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install openai-whisper
pip install transformers accelerate
pip install gymnasium[box2d]
pip install stable-baselines3[extra]
pip install tensorboard
```

### 7.3 Install Additional Robotics Libraries

```bash
# Activate environment
source ~/physical_ai_env/bin/activate

# Install robotics-specific libraries
pip install transforms3d
pip install pybullet
pip install mujoco
pip install dm-control
pip install robosuite
pip install habitat-sim
pip install pyquaternion
pip install trimesh
pip install networkx
```

## Phase 8: Environment Configuration

### 8.1 Create Environment Setup Script

```bash
# Create setup script
cat > ~/physical_ai_setup.sh << 'EOF'
#!/bin/bash

# Physical AI & Humanoid Robotics Environment Setup Script

# Source ROS 2
source /opt/ros/humble/setup.bash

# Source Isaac workspace if built
if [ -f "$HOME/isaac_ws/install/setup.bash" ]; then
    source $HOME/isaac_ws/install/setup.bash
fi

# Set environment variables
export GAZEBO_MODEL_PATH=$HOME/.gazebo/models:$GAZEBO_MODEL_PATH
export GAZEBO_RESOURCE_PATH=$HOME/.gazebo:$GAZEBO_RESOURCE_PATH
export ROS_DOMAIN_ID=0  # Set appropriate domain ID

# Activate Python environment
source $HOME/physical_ai_env/bin/activate

echo "Physical AI environment ready!"
echo "ROS 2 Humble is sourced"
echo "Isaac packages are available" if Isaac workspace exists
echo "Python environment activated"
EOF

chmod +x ~/physical_ai_setup.sh
```

### 8.2 Add to Shell Profile

```bash
# Add to bash profile
echo 'source ~/physical_ai_setup.sh' >> ~/.bashrc
source ~/.bashrc
```

## Phase 9: Testing the Installation

### 9.1 Test ROS 2 Installation

```bash
# Test ROS 2 installation
ros2 --version

# Test basic functionality
ros2 run demo_nodes_cpp talker &
sleep 2
ros2 run demo_nodes_py listener &
sleep 5
pkill -f talker
pkill -f listener
```

### 9.2 Test Gazebo Installation

```bash
# Test Gazebo
gz sim --version

# Launch simple Gazebo world (in separate terminal)
gz sim -v 4 -r shapes.sdf
```

### 9.3 Test Isaac ROS Installation

```bash
# Source workspace
source ~/isaac_ws/install/setup.bash

# Check available Isaac packages
ros2 pkg list | grep isaac
```

### 9.4 Test Python Environment

```bash
# Test Python environment
source ~/physical_ai_env/bin/activate
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python3 -c "import numpy as np; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Phase 10: Workspace Initialization

### 10.1 Create Main Workspace

```bash
# Create main project workspace
mkdir -p ~/physical_ai_workspace/src
cd ~/physical_ai_workspace

# Create basic directory structure
mkdir -p ~/physical_ai_workspace/{config,launch,models,worlds,scripts,docs}

# Initialize as ROS 2 workspace
touch ~/physical_ai_workspace/src/.gitkeep
```

### 10.2 Create First Package

```bash
# Create a basic humanoid robot package
cd ~/physical_ai_workspace/src
ros2 pkg create --build-type ament_python humanoid_robot_bringup --dependencies rclpy std_msgs geometry_msgs sensor_msgs

# Create basic launch directory structure
mkdir -p ~/physical_ai_workspace/src/humanoid_robot_bringup/launch
mkdir -p ~/physical_ai_workspace/src/humanoid_robot_bringup/config
mkdir -p ~/physical_ai_workspace/src/humanoid_robot_bringup/worlds
```

## Phase 11: Hardware Interface Setup (Optional)

### 11.1 Install Real Robot Interfaces

If working with physical robots:

```bash
# Install real robot interfaces (example for common platforms)
sudo apt install ros-humble-ros2-control ros-humble-ros2-controllers
sudo apt install ros-humble-joy ros-humble-teleop-twist-joy
sudo apt install ros-humble-interactive-marker-twist-server

# Install specific robot drivers as needed
# Example for common platforms:
# sudo apt install ros-humble-unitree-*  # For Unitree robots
# sudo apt install ros-humble-robotis-*  # For Robotis robots
```

### 11.2 Install Sensor Drivers

```bash
# Install common sensor drivers
sudo apt install ros-humble-librealsense2
sudo apt install ros-humble-realsense2-camera
sudo apt install ros-humble-pointcloud-to-laserscan
sudo apt install ros-humble-robot-localization
```

## Phase 12: Verification and Troubleshooting

### 12.1 Run Comprehensive Test

```bash
# Create a test script to verify all components
cat > ~/physical_ai_workspace/test_setup.py << 'EOF'
#!/usr/bin/env python3

import sys
import subprocess
import torch
import cv2
import numpy as np

def test_ros2():
    """Test ROS 2 availability"""
    try:
        result = subprocess.run(['ros2', '--version'], capture_output=True, text=True)
        print(f"✓ ROS 2: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("✗ ROS 2: Not found")
        return False

def test_cuda():
    """Test CUDA availability"""
    if torch.cuda.is_available():
        print(f"✓ CUDA: Available (Device: {torch.cuda.get_device_name()})")
        return True
    else:
        print("✗ CUDA: Not available")
        return False

def test_opencv():
    """Test OpenCV"""
    try:
        print(f"✓ OpenCV: {cv2.__version__}")
        return True
    except ImportError:
        print("✗ OpenCV: Not installed")
        return False

def test_gazebo():
    """Test Gazebo"""
    try:
        result = subprocess.run(['gz', 'sim', '--version'], capture_output=True, text=True)
        print(f"✓ Gazebo: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("✗ Gazebo: Not found")
        return False

def main():
    print("Physical AI & Humanoid Robotics Environment Verification")
    print("=" * 55)

    tests = [
        ("ROS 2 Installation", test_ros2),
        ("CUDA Availability", test_cuda),
        ("OpenCV Installation", test_opencv),
        ("Gazebo Installation", test_gazebo),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        print(f"\nTesting {name}...")
        if test_func():
            passed += 1

    print(f"\n{'='*55}")
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Environment is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check installation guide.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
EOF

chmod +x ~/physical_ai_workspace/test_setup.py

# Run verification
source ~/physical_ai_setup.sh
python3 ~/physical_ai_workspace/test_setup.py
```

### 12.2 Common Issues and Solutions

#### Issue: CUDA Not Detected
```bash
# Check if CUDA is properly installed
nvidia-smi
nvcc --version

# Ensure environment variables are set
echo $CUDA_HOME
echo $PATH | grep cuda
```

#### Issue: ROS 2 Packages Not Found
```bash
# Source the setup files
source /opt/ros/humble/setup.bash
source ~/isaac_ws/install/setup.bash  # if Isaac workspace exists

# Check if packages are built
ls ~/isaac_ws/install/
```

#### Issue: Gazebo Not Launching
```bash
# Check Gazebo installation
which gz
ldd $(which gz)  # Check for missing dependencies
```

## Phase 13: Post-Installation Configuration

### 13.1 Create Startup Script

```bash
# Create comprehensive startup script
cat > ~/launch_physical_ai_env.sh << 'EOF'
#!/bin/bash

# Physical AI Development Environment Launcher

echo "🚀 Launching Physical AI & Humanoid Robotics Development Environment..."

# Source ROS 2
source /opt/ros/humble/setup.bash

# Source Isaac workspace if available
if [ -f "$HOME/isaac_ws/install/setup.bash" ]; then
    source $HOME/isaac_ws/install/setup.bash
    echo "📦 Isaac packages sourced"
fi

# Source main workspace
if [ -f "$HOME/physical_ai_workspace/install/setup.bash" ]; then
    source $HOME/physical_ai_workspace/install/setup.bash
    echo "🏭 Physical AI workspace sourced"
fi

# Activate Python environment
source $HOME/physical_ai_env/bin/activate

# Set ROS domain
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}

# Display environment info
echo "✅ Environment Ready!"
echo "ROS 2: $(ros2 --version)"
echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "Python: $(python3 --version)"

# Launch terminator or other tools if needed
exec "$@"
EOF

chmod +x ~/launch_physical_ai_env.sh
```

### 13.2 Create Desktop Entry (Optional)

```bash
# Create desktop entry for easy access
cat > ~/.local/share/applications/physical-ai-env.desktop << 'EOF'
[Desktop Entry]
Name=Physical AI Development Environment
Comment=Complete environment for Physical AI & Humanoid Robotics development
Exec=gnome-terminal -- bash -c "~/launch_physical_ai_env.sh; exec bash"
Icon=applications-science
Terminal=true
Type=Application
Categories=Development;Science;
EOF
```

## Final Verification

Run a complete system test to ensure everything works together:

```bash
# Source environment
source ~/physical_ai_setup.sh

# Test complete pipeline
echo "Testing complete Physical AI environment..."

# Create a simple test to verify all components work together
cd ~/physical_ai_workspace
mkdir -p src/test_physical_ai
cd src/test_physical_ai

# Create a simple test package
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.8)
project(test_physical_ai)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
EOF

cat > package.xml << 'EOF'
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>test_physical_ai</name>
  <version>0.0.1</version>
  <description>Test package for Physical AI environment verification</description>
  <maintainer email="user@example.com">User Name</maintainer>
  <license>Apache License 2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>sensor_msgs</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
EOF

# Build the test package
cd ~/physical_ai_workspace
source /opt/ros/humble/setup.bash
colcon build --packages-select test_physical_ai

if [ $? -eq 0 ]; then
    echo "✅ Environment setup verification completed successfully!"
    echo "🎯 You're ready to start developing Physical AI & Humanoid Robotics applications!"
else
    echo "❌ Environment setup verification failed. Please check installation steps."
fi
```

## Next Steps

After completing this installation:

1. **Explore the tutorials**: Start with the introductory tutorials in the documentation
2. **Run simulations**: Test your installation with basic Gazebo and Isaac Sim examples
3. **Create your first robot**: Design a simple humanoid robot model
4. **Experiment with AI**: Try running basic AI models with your simulated robot

Congratulations! You now have a complete Physical AI & Humanoid Robotics development environment ready for advanced research and development.