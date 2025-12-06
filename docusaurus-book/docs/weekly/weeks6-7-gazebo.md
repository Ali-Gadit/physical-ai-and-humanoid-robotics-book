---
id: weeks6-7-gazebo
title: "Weeks 6-7 - Robot Simulation with Gazebo"
sidebar_position: 3
---

# Weeks 6-7: Robot Simulation with Gazebo

## Overview

During Weeks 6-7, students focus on Gazebo simulation environment setup, physics simulation including gravity and collisions, and sensor simulation for robot development. This phase builds on the ROS 2 foundations learned in previous weeks and prepares students for advanced simulation techniques used in humanoid robotics.

## Learning Objectives

By the end of Weeks 6-7, students will be able to:

1. Set up and configure Gazebo simulation environments
2. Understand physics simulation including gravity, collisions, and material properties
3. Simulate various sensors including LiDAR, depth cameras, and IMUs
4. Create high-fidelity rendering and human-robot interaction scenarios
5. Integrate simulation environments with ROS 2 for seamless testing
6. Understand the principles of Sim-to-Real transfer

## Week 6: Gazebo Simulation Environment Setup

### Day 26: Introduction to Gazebo and Environment Setup

#### Gazebo Installation and Configuration

Gazebo is a 3D simulation environment that provides physics simulation, realistic rendering, and sensor simulation capabilities. For Physical AI and humanoid robotics, Gazebo serves as a critical testing ground where complex behaviors can be developed and validated before deployment to real hardware.

**Installation Prerequisites:**
- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill
- NVIDIA RTX GPU (recommended for high-fidelity rendering)

**Basic Installation:**
```bash
# Install Gazebo Garden
sudo apt update && sudo apt install gazebo-garden

# Install ROS 2 Gazebo packages
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control
```

#### Core Components of Gazebo

1. **Gazebo Server (gzserver)**: Runs the physics simulation and handles all computational aspects
2. **Gazebo Client (gzclient)**: Provides the visual interface for the simulation
3. **Models and Worlds**: Represent objects and environments in the simulation

### Day 27: World Files and Physics Configuration

#### World File Structure

A world file defines the simulation environment including physics properties, lighting, and initial model positions:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="default">
    <!-- Physics engine configuration -->
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
      <uri>model://my_robot</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### Day 28: Robot Models and URDF Integration

#### Creating Robot Models for Simulation

Robot models in Gazebo can be defined using SDF (Simulation Description Format) or imported from URDF (Unified Robot Description Format). For humanoid robots, URDF is typically used and converted to SDF internally.

**URDF to Gazebo Integration:**
```xml
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_robot">
  <!-- Include Gazebo-specific plugins -->
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

### Day 29: Launch Files and ROS 2 Integration

#### Launching Gazebo with ROS 2

Integration with ROS 2 allows for real-time control and monitoring of simulated robots:

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

### Day 30: Testing and Validation

#### Simulation Testing Procedures

1. **Environment Validation**: Verify world physics and lighting
2. **Robot Behavior**: Test joint movements and sensor outputs
3. **ROS 2 Communication**: Confirm topic publishing/subscribing
4. **Performance Testing**: Monitor simulation stability and frame rate

## Week 7: Physics Simulation and Sensor Integration

### Day 31: Advanced Physics Simulation

#### Physics Engine Configuration

Gazebo supports multiple physics engines with different characteristics:

**ODE (Open Dynamics Engine):**
- Default physics engine for Gazebo
- Good balance of speed and accuracy
- Well-suited for most humanoid robotics applications

**Physics Configuration Parameters:**
```xml
<physics name="humanoid_physics" type="ode">
  <max_step_size>0.0005</max_step_size>
  <real_time_update_rate>2000.0</real_time_update_rate>
  <ode>
    <solver>
      <type>quick</type>
      <iters>50</iters>
      <sor>1.0</sor>
    </solver>
    <constraints>
      <cfm>1e-5</cfm>
      <erp>0.1</erp>
    </constraints>
  </ode>
</physics>
```

#### Collision Detection and Response

For humanoid robots, accurate collision detection is crucial for stable walking and interaction:

```xml
<link name="foot_link">
  <collision name="collision">
    <geometry>
      <box size="0.15 0.08 0.02"/>
    </geometry>
  </collision>
  <surface>
    <contact>
      <ode>
        <soft_erp>0.1</soft_erp>    <!-- Error reduction for contacts -->
        <soft_cfm>0.001</soft_cfm>  <!-- Constraint force mixing -->
        <kp>1e+6</kp>              <!-- Contact stiffness -->
        <kd>100</kd>               <!-- Contact damping -->
      </ode>
    </contact>
    <friction>
      <ode>
        <mu>0.8</mu>   <!-- High friction for stable walking -->
        <mu2>0.8</mu2>
      </ode>
    </friction>
  </surface>
</link>
```

### Day 32: Sensor Simulation

#### Camera Sensors

Camera sensors simulate RGB, depth, and stereo cameras:

```xml
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <update_rate>30.0</update_rate>
    <camera name="head">
      <horizontal_fov>1.089</horizontal_fov>  <!-- 62.4 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <always_on>true</always_on>
    <visualize>true</visualize>
  </sensor>
</gazebo>
```

#### LiDAR Sensors

LiDAR sensors provide 2D or 3D distance measurements:

```xml
<gazebo reference="laser_link">
  <sensor name="laser" type="ray">
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-1.570796</min_angle>  <!-- -90 degrees -->
          <max_angle>1.570796</max_angle>    <!-- 90 degrees -->
        </horizontal>
      </scan>
      <range>
        <min>0.10</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <always_on>true</always_on>
    <visualize>true</visualize>
  </sensor>
</gazebo>
```

### Day 33: IMU and Force/Torque Sensors

#### IMU Sensors

IMU sensors provide orientation, angular velocity, and linear acceleration data:

```xml
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>false</visualize>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0017</stddev>  <!-- ~0.1 deg/s (1-sigma) -->
            <bias_mean>0.0004</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0017</stddev>
            <bias_mean>0.0004</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0017</stddev>
            <bias_mean>0.0004</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>  <!-- 1-sigma: 0.017 m/s^2 -->
            <bias_mean>0.0</bias_mean>
            <bias_stddev>0.0017</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
            <bias_mean>0.0</bias_mean>
            <bias_stddev>0.0017</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
            <bias_mean>0.0</bias_mean>
            <bias_stddev>0.0017</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

### Day 34: Physics Tuning for Humanoid Robots

#### Balance and Stability Considerations

Humanoid robots require special physics tuning for stable simulation:

```xml
<physics name="humanoid_physics" type="ode">
  <max_step_size>0.0005</max_step_size>
  <real_time_update_rate>2000.0</real_time_update_rate>
  <ode>
    <solver>
      <type>quick</type>
      <iters>100</iters>  <!-- More iterations for stability -->
      <sor>1.0</sor>
    </solver>
    <constraints>
      <cfm>1e-6</cfm>    <!-- Constraint Force Mixing -->
      <erp>0.1</erp>     <!-- Error Reduction Parameter -->
    </constraints>
  </ode>
</physics>
```

### Day 35: Integration Testing

#### Testing Simulation Components

1. **Physics Validation**: Test gravity, collisions, and joint constraints
2. **Sensor Validation**: Verify sensor data accuracy and noise models
3. **ROS 2 Integration**: Confirm all topics publish/subscriber correctly
4. **Performance Testing**: Monitor frame rate and computational load

## Hands-On Activities

### Week 6 Activities

1. **Gazebo Installation and Basic Setup**
   - Install Gazebo Garden and ROS 2 integration packages
   - Launch basic simulation with default robot
   - Explore Gazebo interface and controls

2. **World Creation Exercise**
   - Create a custom world file with obstacles
   - Add lighting and environmental features
   - Test world loading and physics behavior

3. **Robot Model Integration**
   - Create or import a simple robot model
   - Add Gazebo plugins for ROS 2 control
   - Test basic movement in simulation

### Week 7 Activities

1. **Physics Tuning Exercise**
   - Adjust physics parameters for different robot types
   - Test stability with various configurations
   - Document optimal settings for humanoid robots

2. **Sensor Simulation Implementation**
   - Add multiple sensor types to robot model
   - Verify sensor data in ROS 2 topics
   - Test sensor integration with perception nodes

3. **Simulation Validation**
   - Run comprehensive tests on simulation environment
   - Validate sensor accuracy against real-world data
   - Document any discrepancies or issues

## Assessment

### Week 6 Assessment
- **Lab Exercise**: Create and test a custom Gazebo world
- **Quiz**: Gazebo architecture and core concepts
- **Project**: Integrate a simple robot model with ROS 2

### Week 7 Assessment
- **Simulation Project**: Implement complete sensor suite for robot
- **Performance Test**: Validate physics simulation stability
- **Integration Challenge**: Connect simulation to perception system

## Resources

### Required Reading
- Gazebo Simulation Documentation
- ROS 2 with Gazebo Integration Guide
- Physics Simulation Best Practices

### Tutorials
- Gazebo Beginner Tutorials
- Sensor Simulation in Gazebo
- ROS 2 Control Integration

### Tools
- Gazebo Garden
- RViz2 for visualization
- rqt tools for monitoring
- TMUX for process management

## Next Steps

After completing Weeks 6-7, students will have mastered Gazebo simulation and be ready to move on to Weeks 8-10: NVIDIA Isaac Platform, where they'll learn to develop AI-powered perception and manipulation systems using NVIDIA's advanced robotics platform.