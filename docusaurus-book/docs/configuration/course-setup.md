---
id: course-setup
title: "Course Configuration and Setup Guide"
sidebar_position: 1
---

# Course Configuration and Setup Guide

## Overview

This guide provides comprehensive instructions for configuring the Physical AI & Humanoid Robotics course environment. The configuration process involves setting up the development environment, configuring simulation environments, establishing proper hardware interfaces, and preparing the complete toolchain for Physical AI development.

The course configuration ensures that all components work together seamlessly, from the foundational ROS 2 communication layer to the advanced NVIDIA Isaac AI platform for humanoid robotics applications.

## Prerequisites

### Hardware Requirements
- **Workstation**: RTX 4070 Ti (12GB VRAM) or higher (RTX 3090/4090 recommended)
- **CPU**: Intel i7-13700K or AMD Ryzen 9 7950X (or equivalent)
- **RAM**: 64GB DDR5 (32GB minimum)
- **Storage**: 1TB NVMe SSD (for simulation assets and datasets)
- **OS**: Ubuntu 22.04 LTS

### Software Prerequisites
- NVIDIA GPU drivers (535 or higher)
- CUDA Toolkit 12.0+
- Python 3.10+
- Git and version control tools

## Phase 1: Environment Configuration

### 1.1 System Environment Setup

#### Environment Variables Configuration

Create a comprehensive environment configuration file:

```bash
# Create environment configuration
mkdir -p ~/physical_ai_config
cat > ~/physical_ai_config/env_setup.sh << 'EOF'
#!/bin/bash

# Physical AI & Humanoid Robotics Environment Configuration

# ROS 2 Environment
export ROS_DISTRO=humble
export ROS_WS=~/physical_ai_ws
export ISAAC_WS=~/isaac_ws

# Source ROS 2
source /opt/ros/$ROS_DISTRO/setup.bash

# Source Isaac workspace if available
if [ -d "$ISAAC_WS/install" ]; then
    source $ISAAC_WS/install/setup.bash
fi

# Source main workspace
if [ -d "$ROS_WS/install" ]; then
    source $ROS_WS/install/setup.bash
fi

# Python environment
export PYTHONPATH=$ROS_WS/install/lib/python3.10/site-packages:$PYTHONPATH
export PYTHONPATH=$ISAAC_WS/install/lib/python3.10/site-packages:$PYTHONPATH

# Gazebo Configuration
export GAZEBO_MODEL_PATH=$HOME/.gazebo/models:$GAZEBO_MODEL_PATH
export GAZEBO_RESOURCE_PATH=$HOME/.gazebo:$GAZEBO_RESOURCE_PATH
export GAZEBO_PLUGIN_PATH=$ISAAC_WS/install/lib:$GAZEBO_PLUGIN_PATH

# Isaac Sim Configuration
export ISAAC_SIM_PATH=$HOME/.local/share/ov/pkg/isaac_sim-4.0.0
export OMNI_URL="omniverse://localhost/NVIDIA/Assets/Isaac/4.0"

# Unity Configuration
export UNITY_PROJECTS_PATH=$HOME/unity_projects

# Domain ID (can be changed for multi-robot scenarios)
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}

# Logging configuration
export RCUTILS_LOGGING_USE_STDOUT=1
export RCUTILS_LOGGING_BUFFERED_STREAM=1

# Performance settings
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "Physical AI environment configured:"
echo "  ROS Distribution: $ROS_DISTRO"
echo "  Main Workspace: $ROS_WS"
echo "  Isaac Workspace: $ISAAC_WS"
echo "  Domain ID: $ROS_DOMAIN_ID"
EOF

chmod +x ~/physical_ai_config/env_setup.sh
```

#### Add to Shell Profile

```bash
# Add to ~/.bashrc
echo 'source ~/physical_ai_config/env_setup.sh' >> ~/.bashrc
source ~/.bashrc
```

### 1.2 Workspace Directory Structure

Create the proper directory structure for the course:

```bash
# Create main course directory structure
mkdir -p ~/physical_ai_course/{src,config,launch,models,worlds,scripts,docs,experiments,data}

# Create module-specific directories
mkdir -p ~/physical_ai_course/modules/{module1_ros2,module2_digital_twin,module3_ai_brain,module4_vla}

# Create simulation directories
mkdir -p ~/physical_ai_course/simulations/{gazebo,isaac_sim,unity_exports}

# Create data directories
mkdir -p ~/physical_ai_course/data/{training,validation,test,synthetic}

# Create documentation directories
mkdir -p ~/physical_ai_course/docs/{lectures,labs,reports}
```

### 1.3 ROS 2 Workspace Setup

#### Initialize ROS 2 Workspace

```bash
# Create ROS 2 workspace
mkdir -p ~/physical_ai_ws/src
cd ~/physical_ai_ws

# Create colcon configuration
cat > colcon.meta << 'EOF'
{
    "names": {
        "isaac_ros_common": {
            "install": true
        },
        "isaac_ros_visual_slam": {
            "install": true
        },
        "isaac_ros_apriltag": {
            "install": true
        },
        "isaac_ros_gxf": {
            "install": true
        }
    }
}
EOF

# Initialize workspace
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

## Phase 2: Isaac Sim Configuration

### 2.1 Isaac Sim Environment Setup

```bash
# Create Isaac Sim configuration directory
mkdir -p ~/.nvidia-omniverse/config/kit/isaac-sim/

# Create Isaac Sim settings file
cat > ~/.nvidia-omniverse/config/kit/isaac-sim/standalone_app_settings.json << 'EOF'
{
    "app": {
        "window": {
            "height": 900,
            "width": 1600,
            "title": "Isaac Sim - Physical AI Course"
        },
        "renderer": {
            "max_render_width": 3840,
            "max_render_height": 2160,
            "refresh_rate": 60,
            "msaa": 4
        },
        "physics": {
            "solver_type": "TGS",
            "solver_position_iteration_count": 16,
            "solver_velocity_iteration_count": 8,
            "max_depenetration_velocity": 1000.0
        },
        "omnigraph": {
            "nodes": {
                "enabled": true,
                "debug_mode": false
            }
        }
    },
    "exts": {
        "disabled": [
            "omni.kit.renderer.core.stats",
            "omni.kit.menu.view"
        ]
    }
}
EOF
```

### 2.2 Isaac Sim Python Configuration

```bash
# Create Isaac Sim Python environment configuration
cat > ~/physical_ai_config/isaac_sim_env.py << 'EOF'
"""
Isaac Sim Environment Configuration for Physical AI Course
"""
import os
import carb
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
import numpy as np

# Isaac Sim configuration constants
ISAAC_SIM_CONFIG = {
    "stage_units_in_meters": 1.0,
    "physics_dt": 1.0/60.0,
    "rendering_dt": 1.0/60.0,
    "max_substeps": 8,

    # Camera settings
    "camera_resolution": (1280, 720),
    "camera_fov": 60.0,

    # Simulation settings
    "enable_fabric": True,
    "enable_scene_query_support": True,

    # Asset paths
    "assets_root": get_assets_root_path(),
    "robot_usd_path": "/Isaac/Robots/",
    "objects_path": "/Isaac/Props/",

    # Physics settings
    "gravity": -9.81,
    "solver_type": "TGS",  # TGS or PGSP
    "bounce_threshold": 2.0,
    "friction_combine_mode": "average",
    "restitution_combine_mode": "average"
}

def configure_isaac_sim():
    """Configure Isaac Sim for Physical AI applications"""
    # Set up physics parameters
    physics_settings = {
        "solver_type": ISAAC_SIM_CONFIG["solver_type"],
        "bounce_threshold": ISAAC_SIM_CONFIG["bounce_threshold"],
        "friction_combine_mode": ISAAC_SIM_CONFIG["friction_combine_mode"],
        "restitution_combine_mode": ISAAC_SIM_CONFIG["restitution_combine_mode"]
    }

    # Configure world
    world = World(
        stage_units_in_meters=ISAAC_SIM_CONFIG["stage_units_in_meters"],
        physics_dt=ISAAC_SIM_CONFIG["physics_dt"],
        rendering_dt=ISAAC_SIM_CONFIG["rendering_dt"],
        max_substeps=ISAAC_SIM_CONFIG["max_substeps"],
        backend="numpy",
        device="cuda"
    )

    return world

def setup_humanoid_environment():
    """Setup environment for humanoid robotics applications"""
    # Configure simulation environment for humanoid robots
    world = configure_isaac_sim()

    # Add common humanoid environments
    # Create flat ground
    world.scene.add_default_ground_plane()

    # Add lighting
    from omni.isaac.core.utils.prims import create_prim

    create_prim(
        prim_path="/World/Light",
        prim_type="DistantLight",
        position=np.array([0, 0, 10]),
        attributes={"color": np.array([0.8, 0.8, 0.8]), "intensity": 3000}
    )

    return world
EOF
```

## Phase 3: Gazebo Configuration

### 3.1 Gazebo Environment Setup

```bash
# Create Gazebo configuration directory
mkdir -p ~/.gazebo/{models,worlds,plugins}

# Create custom Gazebo plugins directory
mkdir -p ~/physical_ai_ws/src/gazebo_plugins/src
mkdir -p ~/physical_ai_ws/src/gazebo_plugins/include

# Create Gazebo world configuration
cat > ~/.gazebo/config << 'EOF'
[gazebo]
fullscreen = false
width = 1280
height = 1024
pos_x = 0
pos_y = 0
EOF
```

### 3.2 Physics Engine Configuration

```xml
<!-- Create default physics configuration -->
cat > ~/.gazebo/models/ground_plane/model.sdf << 'EOF'
<?xml version="1.0" ?>
<sdf version="1.7">
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
        <surface>
          <friction>
            <ode>
              <mu>1.0</mu>
              <mu2>1.0</mu2>
            </ode>
            <torsional>
              <coefficient>1.0</coefficient>
            </torsional>
          </friction>
          <bounce>
            <restitution_coefficient>0.01</restitution_coefficient>
            <threshold>100000</threshold>
          </bounce>
          <contact>
            <ode>
              <soft_cfm>0</soft_cfm>
              <soft_erp>0.2</soft_erp>
              <kp>1e+10</kp>
              <kd>1</kd>
              <max_vel>100.0</max_vel>
              <min_depth>0.001</min_depth>
            </ode>
          </contact>
        </surface>
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
          <specular>0.0 0.0 0.0 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF
```

## Phase 4: ROS 2 Package Configuration

### 4.1 Create Course Package Structure

```bash
# Create main course package
mkdir -p ~/physical_ai_ws/src/physical_ai_course
cd ~/physical_ai_ws/src/physical_ai_course

# Create package.xml
cat > package.xml << 'EOF'
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>physical_ai_course</name>
  <version>1.0.0</version>
  <description>Physical AI & Humanoid Robotics Course Package</description>
  <maintainer email="admin@physicalai.edu">Physical AI Course Admin</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>nav_msgs</depend>
  <depend>visualization_msgs</depend>
  <depend>tf2_ros</depend>
  <depend>tf2_geometry_msgs</depend>
  <depend>robot_state_publisher</depend>
  <depend>joint_state_publisher</depend>
  <depend>xacro</depend>
  <depend>gazebo_ros</depend>
  <depend>gazebo_plugins</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
EOF

# Create setup.py
cat > setup.py << 'EOF'
from setuptools import setup
import os
from glob import glob

package_name = 'physical_ai_course'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
        # Include all config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Include all URDF files
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.[xur][rsd]*')),
        # Include all worlds
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Physical AI Course Admin',
    maintainer_email='admin@physicalai.edu',
    description='Physical AI & Humanoid Robotics Course Package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'humanoid_controller = physical_ai_course.humanoid_controller:main',
            'sensor_processor = physical_ai_course.sensor_processor:main',
            'path_planner = physical_ai_course.path_planner:main',
            'voice_interface = physical_ai_course.voice_interface:main',
        ],
    },
)
EOF

# Create __init__.py
mkdir -p physical_ai_course
touch physical_ai_course/__init__.py
```

### 4.2 Create Launch File Configuration

```bash
# Create launch directory and files
mkdir -p ~/physical_ai_ws/src/physical_ai_course/launch

# Create main course launch file
cat > ~/physical_ai_ws/src/physical_ai_course/launch/course_environment.launch.py << 'EOF'
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    launch_rviz = LaunchConfiguration('launch_rviz', default='true')
    launch_gazebo = LaunchConfiguration('launch_gazebo', default='true')
    launch_isaac = LaunchConfiguration('launch_isaac', default='false')

    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_launch_rviz = DeclareLaunchArgument(
        'launch_rviz',
        default_value='true',
        description='Launch RViz if true'
    )

    declare_launch_gazebo = DeclareLaunchArgument(
        'launch_gazebo',
        default_value='true',
        description='Launch Gazebo if true'
    )

    declare_launch_isaac = DeclareLaunchArgument(
        'launch_isaac',
        default_value='false',
        description='Launch Isaac Sim if true'
    )

    # Set parameters globally
    set_parameters = [
        SetParameter(name='use_sim_time', value=use_sim_time),
        SetParameter(name='robot_description', value=''),
    ]

    # Gazebo launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        condition=IfCondition(launch_gazebo)
    )

    # RViz launch
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', [FindPackageShare('physical_ai_course'), '/rviz/course_config.rviz']],
        condition=IfCondition(launch_rviz)
    )

    # Course-specific nodes
    humanoid_controller = Node(
        package='physical_ai_course',
        executable='humanoid_controller',
        name='humanoid_controller',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_model': 'humanoid_v1'}
        ],
        output='screen'
    )

    sensor_processor = Node(
        package='physical_ai_course',
        executable='sensor_processor',
        name='sensor_processor',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Return the launch description
    return LaunchDescription(set_parameters + [
        declare_use_sim_time,
        declare_launch_rviz,
        declare_launch_gazebo,
        declare_launch_isaac,

        gazebo_launch,
        rviz_node,
        humanoid_controller,
        sensor_processor,
    ])
EOF
```

### 4.3 Create Configuration Files

```bash
# Create config directory
mkdir -p ~/physical_ai_ws/src/physical_ai_course/config

# Create robot configuration
cat > ~/physical_ai_ws/src/physical_ai_course/config/humanoid_robot.yaml << 'EOF'
/**:
  ros__parameters:
    # Robot physical parameters
    robot:
      mass: 50.0  # kg
      height: 1.5  # meters
      com_height: 0.85  # Center of mass height

    # Control parameters
    control:
      update_rate: 100.0  # Hz
      max_linear_velocity: 0.5  # m/s
      max_angular_velocity: 0.5  # rad/s
      position_tolerance: 0.05  # meters
      orientation_tolerance: 0.1  # radians

    # Joint limits and properties
    joints:
      hip_pitch_limit: 1.57  # radians
      hip_roll_limit: 0.5
      hip_yaw_limit: 0.78
      knee_limit: 2.35
      ankle_pitch_limit: 0.5
      ankle_roll_limit: 0.3

    # Balance control
    balance:
      com_tolerance: 0.1  # meters
      zmp_tolerance: 0.05  # meters
      balance_kp: 100.0
      balance_kd: 20.0

    # Walking parameters
    walking:
      step_length: 0.3  # meters
      step_width: 0.2
      step_height: 0.05
      walking_speed: 0.3  # m/s
      step_duration: 1.0  # seconds

    # Sensor parameters
    sensors:
      camera:
        fov: 60.0  # degrees
        resolution: [640, 480]
        frame_rate: 30.0
      lidar:
        range_min: 0.1
        range_max: 10.0
        resolution: 0.25  # degrees
        frame_rate: 10.0
      imu:
        linear_acceleration_stddev: 0.017
        angular_velocity_stddev: 0.001
        orientation_stddev: 0.001
EOF

# Create navigation configuration
cat > ~/physical_ai_ws/src/physical_ai_course/config/navigation.yaml << 'EOF'
bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition_bt_node
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Humanoid-specific controller
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
      time_steps: 50
      model_dt: 0.05
      batch_size: 1000
      vx_std: 0.2
      vy_std: 0.2
      wz_std: 0.3
      vx_max: 0.3  # Slower for humanoid stability
      vx_min: -0.1
      vy_max: 0.3
      wz_max: 0.8
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      stateful: True
      k_p: 1.0
      k_i: 0.0
      k_d: 0.0
      max_integral_error: 0.0
      # Humanoid-specific parameters
      step_size: 0.3  # Maximum step size for bipedal locomotion
      balance_constraint: 0.8  # Balance stability factor

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: True
      rolling_window: true
      width: 6
      height: 6
      resolution: 0.05
      # Humanoid-specific parameters
      robot_radius: 0.4  # Larger for humanoid safety
      footprint_padding: 0.1
      inflation_radius: 0.6
      cost_scaling_factor: 5.0

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: True
      robot_radius: 0.4
      resolution: 0.05
      # Humanoid-specific parameters
      track_unknown_space: false
      lethal_cost_threshold: 50
      inflation_radius: 0.8  # Larger for humanoid safety
      cost_scaling_factor: 3.0

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
      # Humanoid-specific parameters
      step_size: 0.3  # Path step size for bipedal constraints
      min_distance_from_obstacle: 0.5  # Safety distance for humanoid
EOF
```

## Phase 5: Isaac ROS Configuration

### 5.1 Isaac ROS Hardware Acceleration Setup

```bash
# Create Isaac ROS configuration
mkdir -p ~/physical_ai_ws/src/physical_ai_course/config/isaac_ros

# Create Isaac ROS perception configuration
cat > ~/physical_ai_ws/src/physical_ai_course/config/isaac_ros/perception.yaml << 'EOF'
/**:
  ros__parameters:
    # Visual SLAM parameters
    visual_slam_node:
      # Input topics
      camera_topic_left: "/camera/left/image_rect_color"
      camera_info_topic_left: "/camera/left/camera_info"
      camera_topic_right: "/camera/right/image_rect_color"
      camera_info_topic_right: "/camera/right/camera_info"
      imu_topic: "/imu/data"

      # Processing parameters
      enable_debug_mode: false
      enable_mapping: true
      enable_localization: true
      enable_point_cloud_output: true

      # Performance parameters
      max_num_points: 100000
      map_publish_period: 1.0
      tracking_rate: 30.0

      # Hardware acceleration
      enable_rectification: true
      enable_ir_rectification: true
      use_compressed_images: false

    # Stereo DNN parameters
    stereo_dnn_node:
      # Neural network parameters
      model_type: "detectnet"
      model_name: "resnet18_detector"
      confidence_threshold: 0.5
      max_objects: 100

      # Performance parameters
      input_width: 960
      input_height: 544
      batch_size: 1

      # Hardware acceleration
      input_layer_name: "input"
      output_layer_name: "output"
      threshold: 0.5

    # Apriltag parameters
    apriltag_node:
      # Input parameters
      image_input_topic: "/camera/image_rect"
      camera_info_input_topic: "/camera/camera_info"

      # Tag parameters
      families: "tag36h11"
      max_tag_id: 50
      tag_edge_size: 0.166

      # Detection parameters
      tag_buffer_size: 0
      det_buffer_size: 0

      # Performance parameters
      sharpening_factor: 0.0
      min_tag_perimeter: 100
EOF
```

### 5.2 Create RViz Configuration

```bash
mkdir -p ~/physical_ai_ws/src/physical_ai_course/rviz

cat > ~/physical_ai_ws/src/physical_ai_course/rviz/course_config.rviz << 'EOF'
Panels:
  - Class: rviz_common/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
        - /TF1/Frames1
        - /RobotModel1
        - /LaserScan1
        - /Image1
        - /PointCloud21
      Splitter Ratio: 0.5
    Tree Height: 897
  - Class: rviz_common/Selection
    Name: Selection
  - Class: rviz_common/Tool Properties
    Expanded:
      - /2D Goal Pose1
      - /Publish Point1
    Name: Tool Properties
    Splitter Ratio: 0.5886790156364441
  - Class: rviz_common/Views
    Expanded:
      - /Current View1
    Name: Views
    Splitter Ratio: 0.5
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz_default_plugins/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.029999999329447746
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 10
      Reference Frame: <Fixed Frame>
      Value: true
    - Class: rviz_default_plugins/TF
      Enabled: true
      Frame Timeout: 15
      Frames:
        All Enabled: true
      Marker Scale: 1
      Name: TF
      Show Arrows: true
      Show Axes: true
      Show Names: false
      Tree:
        {}
      Update Interval: 0
      Value: true
    - Alpha: 1
      Class: rviz_default_plugins/RobotModel
      Collision Enabled: false
      Description Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
      Enabled: true
      Links:
        All Links Enabled: true
        Expand Joint Details: false
        Expand Link Details: false
        Expand Tree: false
        Link Tree Style: Links in Alphabetic Order
      Name: RobotModel
      TF Prefix: ""
      Update Interval: 0
      Value: true
      Visual Enabled: true
    - Alpha: 1
      Autocompute Intensity Bounds: true
      Autocompute Value Bounds:
        Max Value: 10
        Min Value: -10
        Value: true
      Axis: Z
      Channel Name: intensity
      Class: rviz_default_plugins/LaserScan
      Color: 255; 255; 255
      Color Transformer: Intensity
      Decay Time: 0
      Enabled: true
      Invert Rainbow: false
      Max Color: 255; 255; 255
      Max Intensity: 0
      Min Color: 0; 0; 0
      Min Intensity: 0
      Name: LaserScan
      Position Transformer: XYZ
      Queue Size: 10
      Selectable: true
      Size (Pixels): 3
      Size (m): 0.009999999776482582
      Style: Flat Squares
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Best Effort
      Use Fixed Frame: true
      Use rainbow: true
      Value: true
    - Class: rviz_default_plugins/Image
      Enabled: true
      Max Value: 1
      Median window: 5
      Min Value: 0
      Name: Image
      Normalize Range: true
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Best Effort
      Value: true
    - Alpha: 1
      Autocompute Intensity Bounds: true
      Autocompute Value Bounds:
        Max Value: 10
        Min Value: -10
        Value: true
      Axis: Z
      Channel Name: intensity
      Class: rviz_default_plugins/PointCloud2
      Color: 255; 255; 255
      Color Transformer: RGB8
      Decay Time: 0
      Enabled: true
      Invert Rainbow: false
      Max Color: 255; 255; 255
      Max Intensity: 4096
      Min Color: 0; 0; 0
      Min Intensity: 0
      Name: PointCloud2
      Position Transformer: XYZ
      Queue Size: 10
      Selectable: true
      Size (Pixels): 3
      Size (m): 0.009999999776482582
      Style: Flat Squares
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Best Effort
      Use Fixed Frame: true
      Use rainbow: true
      Value: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: map
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
      Hide Inactive Objects: true
    - Class: rviz_default_plugins/SetGoal
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
    - Class: rviz_default_plugins/PublishPoint
      Single click: true
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
  Transformation:
    Current:
      Class: rviz_default_plugins/TF
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 10
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.05999999865889549
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05000000074505806
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.009999999776482582
      Pitch: 0.7853981852531433
      Target Frame: base_link
      Value: Orbit (rviz)
      Yaw: 0.7853981852531433
    Saved: ~
Window Geometry:
  Displays:
    collapsed: false
  Height: 1043
  Hide Left Dock: false
  Hide Right Dock: false
  Image:
    collapsed: false
  QMainWindow State: 000000ff00000000fd000000040000000000000156000003a0fc0200000008fb0000001200530065006c0065006300740069006f006e00000001e10000009b0000005c00fffffffb0000001e0054006f006f006c002000500072006f007000650072007400690065007302000001ed000001df00000185000000a3fb000000120056006900650077007300200054006f006f02000001df000002110000018500000122fb000000200054006f006f006c002000500072006f0070006500720074006900650073003203000002880000011d000002210000017afb000000100044006900730070006c006100790073010000003d000003a0000000c900fffffffb0000002000730065006c0065006300740069006f006e00200062007500660066006500720200000138000000aa0000023a00000294fb00000014005700690064006500530074006500720065006f02000000e6000000d2000003ee0000030bfb0000000c004b0069006e0065006300740200000186000001060000030c00000261000000010000010f000003a0fc0200000003fb0000001e0054006f006f006c002000500072006f00700065007200740069006500730100000041000000780000000000000000fb0000000a00560069006500770073010000003d000003a0000000a400fffffffb0000001200530065006c0065006300740069006f006e010000025a000000b200000000000000000000000200000490000000a9fc0100000001fb0000000a00560069006500770073030000004e00000080000002e100000197000000030000073d0000003efc0100000002fb0000000800540069006d006501000000000000073d000002eb00fffffffb0000000800540069006d00650100000000000004500000000000000000000005d3000003a000000004000000040000000800000008fc0000000100000002000000010000000a005400650073007400560069006500770000000000000000000000000000000003000000010000000100000002000000090058002e0059000000000000000000000000000000000000000000
  Width: 1853
  X: 67
  Y: 27
EOF
```

## Phase 6: Voice Interface Configuration

### 6.1 Create Voice Command Configuration

```bash
# Create voice interface configuration
mkdir -p ~/physical_ai_ws/src/physical_ai_course/config/voice

cat > ~/physical_ai_ws/src/physical_ai_course/config/voice/voice_commands.yaml << 'EOF'
/**:
  ros__parameters:
    # Voice recognition parameters
    voice_recognition:
      model_size: "base"  # tiny, base, small, medium, large
      language: "en"
      temperature: 0.0
      suppress_tokens: [-1]

    # Command processing
    command_processor:
      confidence_threshold: 0.7
      command_timeout: 5.0  # seconds
      wake_word_detection: true
      wake_words: ["robot", "humanoid", "assistant", "hey robot"]

    # Action mapping
    action_mappings:
      # Navigation commands
      "move forward": "navigation.move_forward"
      "go forward": "navigation.move_forward"
      "move backward": "navigation.move_backward"
      "go backward": "navigation.move_backward"
      "turn left": "navigation.turn_left"
      "turn right": "navigation.turn_right"
      "go to kitchen": "navigation.go_to_location:kitchen"
      "go to living room": "navigation.go_to_location:living_room"
      "go to bedroom": "navigation.go_to_location:bedroom"

      # Manipulation commands
      "pick up object": "manipulation.pick_up_object"
      "grasp object": "manipulation.grasp_object"
      "place object": "manipulation.place_object"
      "put down object": "manipulation.place_object"

      # Interaction commands
      "hello": "interaction.greet"
      "how are you": "interaction.respond"
      "what can you do": "interaction.capabilities"
      "introduce yourself": "interaction.introduce"

      # System commands
      "stop": "control.stop"
      "halt": "control.stop"
      "pause": "control.pause"
      "continue": "control.resume"

    # Location mappings
    location_mappings:
      kitchen: [3.0, 1.0, 0.0]
      living_room: [1.0, 2.0, 0.0]
      bedroom: [4.0, 3.0, 0.0]
      office: [2.0, 4.0, 0.0]
      entrance: [0.0, 0.0, 0.0]

    # Safety constraints
    safety_constraints:
      forbidden_commands: ["self_destruct", "harm", "damage"]
      speed_limits:
        navigation: 0.5  # m/s
        manipulation: 0.2  # m/s
      distance_limits:
        approach_human: 0.5  # minimum distance to humans
        navigation_boundary: 10.0  # max navigation distance
EOF
```

## Phase 7: Cognitive Planning Configuration

### 7.1 Create LLM Integration Configuration

```bash
# Create cognitive planning configuration
mkdir -p ~/physical_ai_ws/src/physical_ai_course/config/cognitive

cat > ~/physical_ai_ws/src/physical_ai_course/config/cognitive/planning.yaml << 'EOF'
/**:
  ros__parameters:
    # LLM parameters
    llm_integration:
      provider: "openai"  # openai, anthropic, huggingface
      model: "gpt-4-turbo"  # or specific model name
      api_key: ""  # Will be set via environment variable
      temperature: 0.3
      max_tokens: 1000
      timeout: 30.0

    # Planning parameters
    cognitive_planning:
      max_plan_length: 50  # maximum steps in plan
      planning_timeout: 60.0  # seconds
      validation_enabled: true
      safety_check_enabled: true
      context_window_size: 10  # number of previous interactions to consider

    # Action validation
    action_validator:
      check_collision_risk: true
      check_joint_limits: true
      check_balance_constraints: true
      check_manipulation_feasibility: true

    # Robot capabilities
    robot_capabilities:
      navigation:
        max_speed: 0.5
        min_turn_radius: 0.3
        terrain_types: ["flat", "carpet", "tile"]
      manipulation:
        max_payload: 2.0  # kg
        reach_distance: 1.2  # meters
        joint_limits: true
      perception:
        camera_range: 5.0
        detection_accuracy: 0.85
      interaction:
        speech_synthesis: true
        language_support: ["en", "es", "fr"]

    # Task decomposition
    task_decomposer:
      enable_subtask_generation: true
      max_subtasks: 10
      dependency_resolution: true
      resource_conflict_detection: true

    # Context management
    context_manager:
      enable_memory: true
      memory_retention_hours: 24
      entity_tracking: true
      spatial_reasoning: true
      temporal_reasoning: true
EOF
```

## Phase 8: Build and Test Configuration

### 8.1 Create Build Configuration

```bash
# Create build configuration
cat > ~/physical_ai_ws/colcon_build_config.yaml << 'EOF'
build:
  packages-select:
    - physical_ai_course
    - robot_state_publisher
    - joint_state_publisher
    - gazebo_ros
    - gazebo_plugins

  merge-install: true
  event-handlers:
    - console_cohesion+
    - console_package_list+
    - console_package_prepare+

  cmake-args:
    - -DCMAKE_BUILD_TYPE=Release

  python-executable: /usr/bin/python3

test:
  packages-select:
    - physical_ai_course

  return-code-on-test-failure: true
EOF
```

### 8.2 Create Test Configuration

```bash
# Create test configuration
mkdir -p ~/physical_ai_ws/src/physical_ai_course/test

cat > ~/physical_ai_ws/src/physical_ai_course/test/test_config.py << 'EOF'
"""
Configuration for Physical AI Course Tests
"""
import os
import sys
import unittest
from unittest.mock import Mock, patch

# Test configuration constants
TEST_CONFIG = {
    "timeout": 30.0,  # seconds for each test
    "tolerance": 0.01,  # tolerance for floating point comparisons
    "simulation_steps": 100,  # number of simulation steps for tests
    "test_world": "test_worlds/simple_room.sdf",
    "robot_model": "test_models/simple_humanoid.urdf",

    # Performance thresholds
    "min_frame_rate": 30.0,  # fps
    "max_response_time": 1.0,  # seconds
    "min_accuracy": 0.85,  # for perception tests
}

def get_test_environment():
    """Get test environment configuration"""
    env = {
        "use_sim_time": True,
        "test_mode": True,
        "random_seed": 42,
        "suppress_logs": True,
    }
    return env

class TestBase(unittest.TestCase):
    """Base class for all Physical AI course tests"""

    def setUp(self):
        """Set up test environment"""
        self.test_config = TEST_CONFIG
        self.test_env = get_test_environment()

        # Initialize test-specific configurations
        self.mock_robot = Mock()
        self.mock_world = Mock()

    def tearDown(self):
        """Clean up after tests"""
        pass

    def assertWithinTolerance(self, value, expected, tolerance=None):
        """Assert that value is within tolerance of expected"""
        if tolerance is None:
            tolerance = TEST_CONFIG["tolerance"]

        self.assertAlmostEqual(value, expected, delta=tolerance,
                              msg=f"Value {value} not within tolerance {tolerance} of expected {expected}")

if __name__ == '__main__':
    unittest.main()
EOF
```

## Phase 9: Validation and Verification

### 9.1 Create Configuration Validation Script

```bash
# Create configuration validation script
cat > ~/physical_ai_config/validate_config.py << 'EOF'
#!/usr/bin/env python3
"""
Configuration Validator for Physical AI Course
"""
import os
import sys
import yaml
import json
from pathlib import Path

def validate_ros2_configuration():
    """Validate ROS 2 configuration"""
    print("🔍 Validating ROS 2 configuration...")

    # Check if ROS 2 is sourced
    ros_distro = os.environ.get('ROS_DISTRO')
    if not ros_distro:
        print("❌ ROS 2 not sourced")
        return False
    else:
        print(f"✅ ROS 2 distribution: {ros_distro}")

    # Check workspace
    ros_ws = os.environ.get('ROS_WS')
    if not ros_ws or not os.path.exists(ros_ws):
        print("❌ ROS workspace not found")
        return False
    else:
        print(f"✅ ROS workspace: {ros_ws}")

    # Check if required packages are available
    required_packages = [
        'rclpy', 'std_msgs', 'geometry_msgs', 'sensor_msgs',
        'nav_msgs', 'tf2_ros', 'robot_state_publisher', 'gazebo_ros'
    ]

    for pkg in required_packages:
        try:
            __import__(pkg.replace('-', '_'))
            print(f"✅ Package available: {pkg}")
        except ImportError:
            print(f"❌ Package missing: {pkg}")
            return False

    return True

def validate_hardware_configuration():
    """Validate hardware configuration"""
    print("\n🔍 Validating hardware configuration...")

    # Check for NVIDIA GPU
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
        else:
            print("⚠️  NVIDIA GPU not detected (simulation may be limited)")
    except FileNotFoundError:
        print("⚠️  nvidia-smi not found (GPU acceleration may not be available)")

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA not available")
    except ImportError:
        print("⚠️  PyTorch not installed (install for CUDA validation)")

    return True

def validate_simulation_environment():
    """Validate simulation environment"""
    print("\n🔍 Validating simulation environment...")

    # Check Gazebo
    try:
        import subprocess
        result = subprocess.run(['gz', 'sim', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Gazebo available: {result.stdout.split()[1] if result.stdout.split() else 'unknown'}")
        else:
            print("❌ Gazebo not available")
            return False
    except FileNotFoundError:
        print("❌ Gazebo not found")
        return False

    # Check Isaac Sim path
    isaac_sim_path = os.environ.get('ISAAC_SIM_PATH')
    if isaac_sim_path and os.path.exists(isaac_sim_path):
        print(f"✅ Isaac Sim path configured: {isaac_sim_path}")
    else:
        print("⚠️  Isaac Sim path not configured or not found")

    return True

def validate_course_structure():
    """Validate course directory structure"""
    print("\n🔍 Validating course structure...")

    base_path = Path.home() / "physical_ai_course"

    required_dirs = [
        "src", "config", "launch", "models", "worlds", "scripts",
        "docs", "experiments", "data", "modules"
    ]

    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        if dir_path.exists():
            print(f"✅ Directory exists: {dir_name}")
        else:
            print(f"❌ Directory missing: {dir_name}")
            return False

    # Check module directories
    module_dirs = ["module1_ros2", "module2_digital_twin", "module3_ai_brain", "module4_vla"]
    for module_dir in module_dirs:
        module_path = base_path / "modules" / module_dir
        if module_path.exists():
            print(f"✅ Module directory exists: {module_dir}")
        else:
            print(f"❌ Module directory missing: {module_dir}")
            return False

    return True

def validate_isaac_ros_setup():
    """Validate Isaac ROS setup"""
    print("\n🔍 Validating Isaac ROS setup...")

    # Check Isaac workspace
    isaac_ws = os.environ.get('ISAAC_WS')
    if not isaac_ws or not os.path.exists(isaac_ws):
        print("⚠️  Isaac workspace not found")
        return True  # Not critical for basic operation

    # Check for Isaac ROS packages
    isaac_src = Path(isaac_ws) / "src"
    if isaac_src.exists():
        isaac_packages = list(isaac_src.glob("isaac_ros_*"))
        if isaac_packages:
            print(f"✅ Isaac ROS packages found: {len(isaac_packages)} packages")
        else:
            print("⚠️  No Isaac ROS packages found in workspace")
    else:
        print("⚠️  Isaac source directory not found")

    return True

def main():
    """Main validation function"""
    print("🧪 Running Physical AI Course Configuration Validation\n")

    validation_results = []

    # Run all validations
    validation_results.append(("ROS 2 Configuration", validate_ros2_configuration()))
    validation_results.append(("Hardware Configuration", validate_hardware_configuration()))
    validation_results.append(("Simulation Environment", validate_simulation_environment()))
    validation_results.append(("Course Structure", validate_course_structure()))
    validation_results.append(("Isaac ROS Setup", validate_isaac_ros_setup()))

    # Print summary
    print(f"\n📊 Validation Summary:")
    passed = sum(1 for _, result in validation_results if result)
    total = len(validation_results)

    for name, result in validation_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {name}")

    print(f"\n📈 Overall: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 All validations passed! Course environment is ready.")
        return 0
    else:
        print("⚠️  Some validations failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x ~/physical_ai_config/validate_config.py
```

## Phase 10: Final Integration Test

### 10.1 Create Integration Test Script

```bash
# Create integration test script
cat > ~/physical_ai_config/integration_test.sh << 'EOF'
#!/bin/bash

# Physical AI Course Integration Test Script

echo "🧪 Running Physical AI Course Integration Tests..."

# Source environment
source ~/physical_ai_config/env_setup.sh

# Test 1: ROS 2 Basic Functionality
echo -e "\n🔍 Test 1: ROS 2 Basic Functionality"
if command -v ros2 &> /dev/null; then
    echo "✅ ROS 2 command line tools available"
    ros2 --version
else
    echo "❌ ROS 2 not available"
    exit 1
fi

# Test 2: Workspace Build
echo -e "\n🔧 Test 2: Workspace Build"
cd ~/physical_ai_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select physical_ai_course --event-handlers console_direct+ || {
    echo "❌ Build failed"
    exit 1
}
echo "✅ Workspace builds successfully"

# Test 3: Gazebo Availability
echo -e "\n🎮 Test 3: Gazebo Availability"
if command -v gz &> /dev/null; then
    echo "✅ Gazebo command line tools available"
    gz --version
else
    echo "⚠️  Gazebo not available"
fi

# Test 4: Python Environment
echo -e "\n🐍 Test 4: Python Environment"
if python3 -c "import rclpy; import cv2; import torch; import numpy" &> /dev/null; then
    echo "✅ Required Python packages available"
else
    echo "❌ Required Python packages missing"
    exit 1
fi

# Test 5: Isaac ROS Packages (if available)
echo -e "\n🤖 Test 5: Isaac ROS Packages"
if [ -d "$ISAAC_WS/install" ]; then
    source $ISAAC_WS/install/setup.bash
    if ros2 pkg list | grep -i "isaac" &> /dev/null; then
        echo "✅ Isaac ROS packages available"
    else
        echo "⚠️  Isaac ROS packages not built"
    fi
else
    echo "⚠️  Isaac workspace not available"
fi

# Test 6: Launch System
echo -e "\n🚀 Test 6: Launch System Test"
cd ~/physical_ai_ws
source install/setup.bash
if ros2 launch physical_ai_course course_environment.launch.py use_sim_time:=false &> /dev/null &; then
    echo "✅ Launch system works (process started in background)"
    pkill -f "physical_ai_course"
else
    echo "⚠️  Launch system test failed"
fi

# Test 7: Configuration Files Exist
echo -e "\n⚙️  Test 7: Configuration Files"
config_files=(
    "config/humanoid_robot.yaml"
    "config/navigation.yaml"
    "config/isaac_ros/perception.yaml"
    "config/voice/voice_commands.yaml"
    "config/cognitive/planning.yaml"
    "launch/course_environment.launch.py"
    "rviz/course_config.rviz"
)

missing_configs=0
for config in "${config_files[@]}"; do
    if [ -f "src/physical_ai_course/$config" ]; then
        echo "✅ Config file exists: $config"
    else
        echo "❌ Config file missing: $config"
        ((missing_configs++))
    fi
done

if [ $missing_configs -gt 0 ]; then
    echo "❌ $missing_configs configuration files are missing"
    exit 1
fi

echo -e "\n🎉 All integration tests passed!"
echo "The Physical AI & Humanoid Robotics course environment is properly configured."
echo ""
echo "Next steps:"
echo "1. Run the validation script: python3 ~/physical_ai_config/validate_config.py"
echo "2. Try launching the course environment: ros2 launch physical_ai_course course_environment.launch.py"
echo "3. Proceed to Module 1 exercises in the course documentation"

exit 0
EOF

chmod +x ~/physical_ai_config/integration_test.sh
```

## Phase 11: Documentation and Setup Verification

### 11.1 Create Final Setup Verification

```bash
# Create setup verification document
cat > ~/physical_ai_course/docs/setup_verification.md << 'EOF'
# Setup Verification Guide

## Overview

This guide provides a comprehensive verification process to ensure your Physical AI & Humanoid Robotics course environment is properly configured. Complete all verification steps before proceeding with the course modules.

## Pre-Verification Checklist

- [ ] System meets hardware requirements (RTX 4070 Ti or better)
- [ ] Ubuntu 22.04 LTS is installed
- [ ] NVIDIA drivers are properly installed
- [ ] CUDA toolkit is installed and working
- [ ] All prerequisite software is installed

## Verification Steps

### 1. ROS 2 Environment

```bash
# Verify ROS 2 installation
source /opt/ros/humble/setup.bash
ros2 --version

# Verify workspace
cd ~/physical_ai_ws
source install/setup.bash
ros2 pkg list | grep physical_ai_course
```

**Expected Output**: ROS 2 version information and confirmation that `physical_ai_course` package is available.

### 2. Simulation Environment

```bash
# Test Gazebo
gz --version

# Test Isaac Sim (if installed)
python3 -c "import omni; print('Isaac Sim Python API available')" 2>/dev/null || echo "Isaac Sim not available"
```

**Expected Output**: Gazebo version information and Isaac Sim availability confirmation.

### 3. Hardware Acceleration

```bash
# Test GPU availability
nvidia-smi

# Test CUDA with PyTorch
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected Output**: GPU information and CUDA availability confirmation.

### 4. Course Package Build

```bash
# Build the course package
cd ~/physical_ai_ws
colcon build --packages-select physical_ai_course
source install/setup.bash
```

**Expected Output**: Successful build with no errors.

### 5. Launch System Test

```bash
# Test launch system
ros2 launch physical_ai_course course_environment.launch.py use_sim_time:=false
```

**Expected Output**: Launch system starts without errors (may need to Ctrl+C to stop).

### 6. Configuration Validation

```bash
# Run configuration validation
python3 ~/physical_ai_config/validate_config.py
```

**Expected Output**: All validation checks pass with "✅" indicators.

### 7. Integration Test

```bash
# Run integration test
~/physical_ai_config/integration_test.sh
```

**Expected Output**: All integration tests pass successfully.

## Troubleshooting Common Issues

### Issue: ROS 2 Commands Not Found
**Solution**: Ensure you've sourced the ROS 2 setup file:
```bash
source /opt/ros/humble/setup.bash
```

### Issue: Isaac Sim Not Available
**Solution**: Verify Isaac Sim installation and environment variables:
```bash
echo $ISAAC_SIM_PATH
ls -la $ISAAC_SIM_PATH
```

### Issue: GPU Acceleration Not Working
**Solution**: Check NVIDIA driver and CUDA installation:
```bash
nvidia-smi
nvcc --version
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Package Build Fails
**Solution**: Check for missing dependencies:
```bash
cd ~/physical_ai_ws
rosdep install --from-paths src --ignore-src -r -y
```

## Performance Benchmarks

### Minimum Performance Requirements:
- **ROS 2**: Command response time < 100ms
- **Gazebo**: Simulation at > 30 FPS with simple models
- **Isaac Sim**: Rendering at > 10 FPS with basic scenes
- **Python**: NumPy operations complete in < 100ms

### Testing Performance:
```bash
# Test ROS 2 performance
time ros2 topic list

# Test Python performance
python3 -c "import time; import numpy as np; start=time.time(); np.random.random((1000,1000)).dot(np.random.random((1000,1000))); print(f'Time: {time.time()-start:.3f}s')"
```

## Environment Readiness Confirmation

Once all verification steps pass, your environment is ready for the Physical AI & Humanoid Robotics course. You should be able to:

- ✅ Source ROS 2 and course workspace
- ✅ Build the course packages successfully
- ✅ Launch the course environment
- ✅ Access all required simulation tools
- ✅ Run basic ROS 2 commands
- ✅ Access GPU acceleration features

## Next Steps

After successful verification:

1. **Explore the documentation**: Review the course materials in the `docs/` directory
2. **Start with Module 1**: Begin with ROS 2 fundamentals
3. **Run basic examples**: Try the example code in the course package
4. **Join the community**: Participate in course forums and discussions

## Support

If you encounter issues during verification:

- Check the troubleshooting guide in `./support/troubleshooting.md`
- Review system requirements in `./hardware-requirements.md`
- Consult the course forums
- Contact technical support if needed

Your Physical AI & Humanoid Robotics development environment is now verified and ready for advanced robotics development!
EOF
```

Now I'll update the todo list to mark this task as completed:
