---
id: gazebo-setup
title: "Gazebo Simulation Environment Setup"
sidebar_position: 2
---

# Gazebo Simulation Environment Setup

## Introduction

Gazebo is a powerful 3D simulation environment that provides physics simulation, realistic rendering, and sensor simulation capabilities. For Physical AI and humanoid robotics, Gazebo serves as a critical testing ground where complex behaviors can be developed and validated before deployment to real hardware.

This section will guide you through setting up Gazebo, understanding its core components, and creating simulation environments for humanoid robots.

## Installing Gazebo

### Prerequisites

Before installing Gazebo, ensure you have:
- Ubuntu 22.04 LTS (recommended for ROS 2 Humble)
- ROS 2 Humble Hawksbill installed
- NVIDIA RTX GPU with appropriate drivers (for high-fidelity rendering)

### Installation

For ROS 2 Humble, install Gazebo Garden:

```bash
# Add the Gazebo repository
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/gazebo-tooling/gazebodistro/master/repo/focal/gazebo-stable.list > /etc/apt/sources.list.d/gazebo-stable.list
sudo apt update

# Install Gazebo Garden
sudo apt install gazebo-garden
```

For ROS 2 integration, install the ROS 2 Gazebo packages:

```bash
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control
```

## Core Components of Gazebo

### 1. Gazebo Server (gzserver)

The Gazebo server runs the physics simulation and handles all the computational aspects of the simulation. It can run without a GUI for faster execution.

### 2. Gazebo Client (gzclient)

The client provides the visual interface for the simulation. It connects to the server to display the simulation in real-time.

### 3. Models and Worlds

- **Models**: Represent objects in the simulation (robots, furniture, etc.)
- **Worlds**: Define the environment, including physics properties, lighting, and initial model positions

## Basic Gazebo Concepts

### Worlds

A world file defines the simulation environment. Here's a basic example:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="default">
    <!-- Physics engine -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.3 0.3 -1</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.7 0.7 0.7 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Include robot -->
    <include>
      <uri>model://my_humanoid_robot</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### Models

Models define the objects in your simulation. Here's an example model directory structure:

```
~/.gazebo/models/my_humanoid_robot/
├── model.config
└── model.sdf
```

**model.config**:
```xml
<?xml version="1.0"?>
<model>
  <name>my_humanoid_robot</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <author>
    <name>Your Name</name>
    <email>your.email@example.com</email>
  </author>
  <description>A simple humanoid robot model for simulation.</description>
</model>
```

## ROS 2 Integration

### Gazebo ROS Packages

Gazebo integrates with ROS 2 through several packages:

- `gazebo_ros`: Core ROS 2 interface for Gazebo
- `gazebo_ros_pkgs`: Additional ROS 2 plugins for Gazebo
- `gazebo_ros2_control`: ROS 2 control interface for Gazebo

### Launching Gazebo with ROS 2

Here's how to launch Gazebo with ROS 2 integration:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('my_robot_gazebo'),
                'worlds',
                'my_world.sdf'
            ])
        }.items()
    )

    return LaunchDescription([
        gazebo
    ])
```

## Creating a Humanoid Robot Simulation

### Robot Configuration for Gazebo

To make your humanoid robot work in Gazebo, you need to add Gazebo-specific plugins to your URDF:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_robot">

  <!-- Include your basic URDF here -->
  <!-- ... your links and joints ... -->

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin name="gazebo_ros2_control" filename="libgazebo_ros2_control.so">
      <parameters>$(find-pkg-share my_robot_description)/config/my_robot_controllers.yaml</parameters>
    </plugin>
  </gazebo>

  <!-- Sensor plugins -->
  <gazebo reference="head_camera">
    <sensor name="camera" type="camera">
      <camera>
        <horizontal_fov>1.089</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
        </image>
        <clip>
          <near>0.1</near>
          <far>10.0</far>
        </clip>
      </camera>
      <always_on>true</always_on>
      <update_rate>30</update_rate>
      <visualize>true</visualize>
    </sensor>
  </gazebo>

</robot>
```

### Controller Configuration

Create a controller configuration file (`my_robot_controllers.yaml`):

```yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    left_leg_controller:
      type: position_controllers/JointGroupPositionController

    right_leg_controller:
      type: position_controllers/JointGroupPositionController

    left_arm_controller:
      type: position_controllers/JointGroupPositionController

    right_arm_controller:
      type: position_controllers/JointGroupPositionController

left_leg_controller:
  ros__parameters:
    joints:
      - left_hip_joint
      - left_knee_joint
      - left_ankle_joint

right_leg_controller:
  ros__parameters:
    joints:
      - right_hip_joint
      - right_knee_joint
      - right_ankle_joint

left_arm_controller:
  ros__parameters:
    joints:
      - left_shoulder_joint
      - left_elbow_joint

right_arm_controller:
  ros__parameters:
    joints:
      - right_shoulder_joint
      - right_elbow_joint
```

## Common Gazebo Commands

### Starting Gazebo
```bash
# Start Gazebo server only (no GUI)
gz sim -s

# Start Gazebo with GUI
gz sim

# Start with a specific world
gz sim -r my_world.sdf
```

### Using ROS 2 with Gazebo
```bash
# Launch with ROS 2 integration
ros2 launch my_robot_gazebo my_robot_world.launch.py

# Check available topics
ros2 topic list | grep gazebo

# Send commands to joints
ros2 topic pub /left_leg_controller/commands std_msgs/Float64MultiArray "data: [0.5, 0.0, -0.5]"
```

## Best Practices for Gazebo Simulation

### 1. Physics Tuning

- Start with realistic physics parameters
- Adjust time step for stability vs. performance
- Use appropriate solver parameters for humanoid dynamics

### 2. Model Optimization

- Use simplified collision geometries for performance
- Balance visual quality with simulation speed
- Optimize mesh complexity for real-time rendering

### 3. Sensor Simulation

- Configure sensors to match real hardware specifications
- Add realistic noise models
- Validate sensor outputs against real sensors

### 4. Environment Design

- Create diverse environments for robust testing
- Include challenging scenarios for thorough validation
- Document environment parameters for reproducibility

## Troubleshooting Common Issues

### Performance Issues
- Reduce physics update rate if simulation is too slow
- Simplify collision meshes for faster physics calculations
- Use less complex rendering settings for development

### Joint Control Problems
- Verify joint limits and types match your robot
- Check controller configuration files
- Ensure proper transmission elements in URDF

### Sensor Data Issues
- Verify sensor plugins are properly configured
- Check topic names and message types
- Validate sensor noise parameters

## Advanced Features

### Multi-Robot Simulation
Gazebo supports multiple robots in the same environment:

```xml
<world name="multi_robot_world">
  <!-- Robot 1 -->
  <include>
    <name>robot1</name>
    <uri>model://humanoid_robot</uri>
    <pose>0 0 1 0 0 0</pose>
  </include>

  <!-- Robot 2 -->
  <include>
    <name>robot2</name>
    <uri>model://humanoid_robot</uri>
    <pose>2 0 1 0 0 0</pose>
  </include>
</world>
```

### Dynamic Environments
You can create dynamic environments that change during simulation for more realistic testing scenarios.

## Hands-on Exercise

Create a complete Gazebo simulation environment that includes:
1. A basic humanoid robot model
2. A simple world with obstacles
3. Sensor simulation (camera and IMU)
4. ROS 2 control interface
5. A launch file to start the entire simulation

This will give you hands-on experience with setting up complete simulation environments for humanoid robots.