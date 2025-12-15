---
id: weeks6-7-gazebo
title: "Weeks 6-7 - Robot Simulation with Gazebo"
sidebar_position: 3
---

import BilingualChapter from '@site/src/components/BilingualChapter';

<BilingualChapter>
  <div className="english">
    # Weeks 6-7: Robot Simulation with Gazebo

    ## Overview

    Welcome to Weeks 6-7 of the Physical AI & Humanoid Robotics course! During these weeks, you'll explore the power of Gazebo as a simulation environment for robotics. Gazebo serves as the digital twin for physical robots, allowing you to test and validate your algorithms in a safe, controllable, and repeatable environment before deploying to real hardware.

    Simulation is crucial for Physical AI development because it provides:
    - **Safe Testing**: Validate algorithms without risk to hardware or humans
    - **Controlled Environments**: Reproducible conditions for debugging
    - **Physics Simulation**: Accurate modeling of physical laws and interactions
    - **Cost Efficiency**: Reduce hardware wear and accelerate development cycles
    - **Scalability**: Test with multiple robots and complex scenarios

    ## Learning Objectives

    By the end of Weeks 6-7, you will be able to:

    1. Understand the architecture and capabilities of Gazebo simulation
    2. Create and configure robot models for Gazebo simulation
    3. Design and build custom simulation environments
    4. Integrate Gazebo with ROS 2 for realistic robot simulation
    5. Configure physics properties and sensor models for accurate simulation
    6. Implement sensor simulation including cameras, LIDAR, and IMUs
    7. Validate robot behaviors in simulation before real-world deployment

    ## Week 6: Gazebo Fundamentals and Environment Setup

    ### Day 1: Introduction to Gazebo Simulation

    #### What is Gazebo?

    Gazebo is a 3D dynamic simulator with the ability to accurately and efficiently simulate populations of robots in complex indoor and outdoor environments. It provides:

    - **High-fidelity physics**: Accurate simulation of rigid-body dynamics
    - **Realistic rendering**: High-quality graphics for sensor simulation
    - **Extensible plugin system**: Custom plugins for sensors and controllers
    - **Large community**: Extensive models and examples available
    - **ROS integration**: Seamless integration with ROS and ROS 2

    #### Key Features of Gazebo

    **Physics Engine**: Based on ODE (Open Dynamics Engine) for realistic physics simulation
    - Collision detection and response
    - Friction and contact modeling
    - Gravity and environmental forces
    - Joint constraints and limits

    **Sensor Simulation**: Accurate modeling of various sensors:
    - Camera sensors (RGB, depth, stereo)
    - LIDAR and laser rangefinders
    - IMU and accelerometer sensors
    - Force/torque sensors
    - GPS and magnetometer sensors

    **Rendering**: High-quality visualization capabilities:
    - OpenGL-based graphics rendering
    - Dynamic lighting and shadows
    - Texture mapping and materials
    - Realistic visual effects

    ### Day 2: Gazebo Architecture and Components

    #### Core Architecture

    ```
    ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
    │   Gazebo GUI   │◄──►│  Gazebo Server   │◄──►│  Plugin System  │
    │  (gzclient)    │    │   (gzserver)     │    │                 │
    └─────────────────┘    └──────────────────┘    └─────────────────┘
             │                       │                        │
             ▼                       ▼                        ▼
    ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
    │ Visualization   │    │ Physics Engine  │    │ Sensors & Ctrl │
    │ & Interaction   │    │ (ODE, Bullet)   │    │ (Plugins)       │
    └─────────────────┘    └──────────────────┘    └─────────────────┘
    ```

    #### Gazebo Components

    **Gazebo Server (gzserver)**: Runs the physics simulation and handles models
    - Manages simulation time and physics calculations
    - Handles model spawning and destruction
    - Provides services for simulation control

    **Gazebo Client (gzclient)**: Provides the graphical user interface
    - 3D visualization of the simulation
    - Camera controls and scene interaction
    - Statistics and information display

    **Plugin System**: Extensible architecture for custom functionality
    - Sensor plugins for various sensor types
    - Controller plugins for robot control
    - World plugins for custom simulation logic
    - GUI plugins for custom interfaces

    ### Day 3: Setting Up Gazebo with ROS 2

    #### ROS 2 Integration

    Gazebo integrates with ROS 2 through several packages:

    - **gazebo_ros_pkgs**: Core ROS 2 packages for Gazebo integration
    - **gazebo_plugins**: Various sensor and controller plugins
    - **gazebo_dev**: Development tools and headers
    - **ros_gz**: Bridge packages for ROS 2 ↔ Gazebo communication

    #### Installation and Configuration

    ```bash
    # Install Gazebo Garden (recommended version)
    sudo apt update
    sudo apt install ros-humble-gazebo-*

    # Verify installation
    gz --version
    ```

    #### Basic Gazebo Commands

    ```bash
    # Start Gazebo server
    gz sim -r empty.sdf

    # Start Gazebo with GUI
    gz sim -g -r empty.sdf

    # List available worlds
    ls /usr/share/gazebo/worlds/
    ```

    ### Day 4: Creating Your First Gazebo World

    #### World File Structure

    Gazebo worlds are defined in SDF (Simulation Description Format) files:

    ```xml
    <?xml version="1.0" ?>
    <sdf version="1.7">
      <world name="default">
        <!-- Physics engine -->
        <physics name="1ms" type="ode">
          <max_step_size>0.001</max_step_size>
          <real_time_factor>1</real_time_factor>
          <real_time_update_rate>1000.0</real_time_update_rate>
          <gravity>0 0 -9.8</gravity>
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
                <ambient>0.8 0.8 0.8 1</ambient>
                <diffuse>0.8 0.8 0.8 1</diffuse>
                <specular>0.8 0.8 0.8 1</specular>
              </material>
            </visual>
          </link>
        </model>
      </world>
    </sdf>
    ```

    ### Day 5: Understanding Physics Simulation

    #### Physics Properties

    Accurate physics simulation is crucial for sim-to-real transfer:

    - **Gravity**: Typically -9.8 m/s² on Earth
    - **Friction**: Static and dynamic friction coefficients
    - **Restitution**: Bounciness of collisions (0.0-1.0)
    - **Damping**: Energy loss in joints and motion

    #### Tuning Physics Parameters

    ```xml
    <physics name="ode" type="ode">
      <!-- Time stepping -->
      <max_step_size>0.001</max_step_size>  <!-- Smaller = more accurate but slower -->
      <real_time_factor>1.0</real_time_factor>  <!-- Simulation speed multiplier -->
      <real_time_update_rate>1000.0</real_time_update_rate>  <!-- Hz -->

      <!-- Gravity -->
      <gravity>0 0 -9.8</gravity>

      <!-- Solver -->
      <ode>
        <solver>
          <type>quick</type>
          <iters>100</iters>  <!-- Iterations per step -->
          <sor>1.0</sor>      <!-- Successive over-relaxation -->
        </solver>
        <constraints>
          <cfm>0.0</cfm>      <!-- Constraint force mixing -->
          <erp>0.2</erp>      <!-- Error reduction parameter -->
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>
    ```

    ## Week 7: Advanced Simulation and Integration

    ### Day 6: Robot Model Integration in Gazebo

    #### Adding Robots to Gazebo

    Robots are integrated into Gazebo using URDF models with Gazebo-specific extensions:

    ```xml
    <!-- In your robot URDF/XACRO file -->
    <gazebo reference="base_link">
      <material>Gazebo/Blue</material>
      <mu1>0.2</mu1>
      <mu2>0.2</mu2>
      <kp>1000000.0</kp>  <!-- Contact stiffness -->
      <kd>100.0</kd>      <!-- Contact damping -->
    </gazebo>

    <!-- Adding differential drive controller -->
    <gazebo>
      <plugin filename="libgazebo_ros_diff_drive.so" name="diff_drive">
        <ros>
          <namespace>/my_robot</namespace>
          <remapping>cmd_vel:=cmd_vel</remapping>
          <remapping>odom:=odom</remapping>
        </ros>
        <update_rate>30</update_rate>
        <left_joint>left_wheel_joint</left_joint>
        <right_joint>right_wheel_joint</right_joint>
        <wheel_separation>0.3</wheel_separation>
        <wheel_diameter>0.15</wheel_diameter>
        <max_wheel_torque>20</max_wheel_torque>
        <max_wheel_acceleration>1.0</max_wheel_acceleration>
        <publish_odom>true</publish_odom>
        <publish_odom_tf>true</publish_odom_tf>
        <odometry_frame>odom</odometry_frame>
        <robot_base_frame>base_link</robot_base_frame>
      </plugin>
    </gazebo>
    ```

    #### Spawning Robots in Simulation

    ```bash
    # Spawn robot from URDF file
    ros2 run gazebo_ros spawn_entity.py -entity my_robot -file /path/to/robot.urdf

    # Spawn with position
    ros2 run gazebo_ros spawn_entity.py -entity my_robot -file /path/to/robot.urdf -x 1.0 -y 2.0 -z 0.0

    # Spawn via launch file
    <node pkg="gazebo_ros" exec="spawn_entity.py" args="-entity my_robot -file $(find-pkg-share my_robot_description)/urdf/robot.urdf">
    </node>
    ```

    ### Day 7: Sensor Simulation in Gazebo

    #### Camera Sensors

    ```xml
    <gazebo reference="camera_link">
      <sensor name="camera" type="camera">
        <update_rate>30</update_rate>
        <camera name="head">
          <horizontal_fov>1.3962634</horizontal_fov>  <!-- 80 degrees -->
          <image>
            <width>640</width>
            <height>480</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.1</near>
            <far>100</far>
          </clip>
        </camera>
        <always_on>true</always_on>
        <visualize>true</visualize>
        <plugin filename="libgazebo_ros_camera.so" name="camera_controller">
          <ros>
            <namespace>/my_robot</namespace>
            <remapping>~/image_raw:=camera/image_raw</remapping>
            <remapping>~/camera_info:=camera/camera_info</remapping>
          </ros>
          <camera_name>camera</camera_name>
          <frame_name>camera_link</frame_name>
        </plugin>
      </sensor>
    </gazebo>
    ```

    #### LIDAR Sensors

    ```xml
    <gazebo reference="lidar_link">
      <sensor name="lidar" type="gpu_lidar">
        <update_rate>10</update_rate>
        <ray>
          <scan>
            <horizontal>
              <samples>360</samples>
              <resolution>1.0</resolution>
              <min_angle>-3.14159</min_angle>  <!-- -π -->
              <max_angle>3.14159</max_angle>    <!-- π -->
            </horizontal>
          </scan>
          <range>
            <min>0.1</min>
            <max>30.0</max>
            <resolution>0.01</resolution>
          </range>
        </ray>
        <always_on>true</always_on>
        <visualize>true</visualize>
        <plugin filename="libgazebo_ros_gpu_lidar.so" name="gpu_lidar_plugin">
          <ros>
            <namespace>/my_robot</namespace>
            <remapping>~/out:=scan</remapping>
          </ros>
          <frame_name>lidar_link</frame_name>
        </plugin>
      </sensor>
    </gazebo>
    ```

    #### IMU Sensors

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
                <stddev>2e-4</stddev>
              </noise>
            </x>
            <y>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>2e-4</stddev>
              </noise>
            </y>
            <z>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>2e-4</stddev>
              </noise>
            </z>
          </angular_velocity>
          <linear_acceleration>
            <x>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>1.7e-2</stddev>
              </noise>
            </x>
            <y>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>1.7e-2</stddev>
              </noise>
            </y>
            <z>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>1.7e-2</stddev>
              </noise>
            </z>
          </linear_acceleration>
        </imu>
        <plugin filename="libgazebo_ros_imu_sensor.so" name="imu_plugin">
          <ros>
            <namespace>/my_robot</namespace>
            <remapping>~/out:=imu</remapping>
          </ros>
          <frame_name>imu_link</frame_name>
        </plugin>
      </sensor>
    </gazebo>
    ```

    ### Day 8: Environment Design and Modeling

    #### Creating Custom Worlds

    Designing realistic environments for humanoid robots:

    - **Human-scale environments**: Doorways, furniture, stairs
    - **Physics-appropriate materials**: Friction, restitution, damping
    - **Lighting conditions**: Indoor/outdoor variations
    - **Dynamic elements**: Moving objects, people

    #### Building Complex Environments

    ```xml
    <!-- Example: Living room environment -->
    <model name="living_room">
      <pose>0 0 0 0 0 0</pose>
      <link name="floor">
        <collision name="collision">
          <geometry>
            <box>
              <size>5 4 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>5 4 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.2 1</ambient>
            <diffuse>0.8 0.6 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Furniture -->
    <model name="sofa">
      <pose>2 1 0 0 0 0</pose>
      <!-- Sofa model definition -->
    </model>
    ```

    ### Day 9: Physics Tuning for Humanoid Robots

    #### Balancing and Stability

    Humanoid robots require special attention to physics parameters:

    - **Center of Mass**: Accurate CoM placement for stable walking
    - **Inertia Tensors**: Realistic mass distribution
    - **Joint Damping**: Appropriate for natural movement
    - **Foot Contact**: Accurate friction for walking stability

    #### Walking Dynamics

    ```xml
    <!-- Example joint configuration for humanoid leg -->
    <joint name="hip_joint" type="revolute">
      <parent>torso</parent>
      <child>thigh</child>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="1.57" effort="100" velocity="2"/>
      <dynamics damping="5.0" friction="0.1"/>  <!-- Tune for natural movement -->
    </joint>

    <gazebo reference="thigh">
      <self_collide>false</self_collide>
      <kinematic>false</kinematic>
      <gravity>true</gravity>
      <mu1>0.9</mu1>  <!-- High friction for feet -->
      <mu2>0.9</mu2>
      <fdir1>1 0 0</fdir1>  <!-- Direction of friction -->
      <max_vel>0.04</max_vel>
      <min_depth>0.001</min_depth>
    </gazebo>
    ```

    ### Day 10: Simulation Validation and Testing

    #### Validating Simulation Accuracy

    Key metrics for simulation validation:

    - **Kinematic accuracy**: Joint positions match expected values
    - **Dynamic behavior**: Movement patterns similar to real robot
    - **Sensor data**: Matches real-world sensor characteristics
    - **Timing**: Simulation time matches real-time when possible

    #### Sim-to-Real Transfer Considerations

    Minimize the sim-to-real gap:

    - **Model accuracy**: Detailed URDF with correct masses/inertias
    - **Sensor noise**: Include realistic noise models
    - **Actuator dynamics**: Model motor delays and limitations
    - **Environmental factors**: Include friction, air resistance
    - **Calibration**: Align simulation parameters with real robot

    ## Hands-On Projects

    ### Week 6 Project: Basic Simulation Environment

    1. Install and configure Gazebo with ROS 2 integration
    2. Create a simple world with ground plane and lighting
    3. Import a basic robot model (e.g., differential drive)
    4. Test basic movement and sensor readings
    5. Set up a launch file for your simulation environment

    ### Week 7 Project: Advanced Humanoid Simulation

    1. Create a humanoid robot model with appropriate URDF
    2. Add realistic sensors (camera, IMU, LIDAR)
    3. Design a human-scale environment
    4. Implement walking or basic movement patterns
    5. Validate sensor data accuracy and physics behavior

    ## Assessment

    ### Week 6 Assessment
    - **Quiz**: Gazebo architecture and core concepts
    - **Lab Exercise**: Create and run a basic simulation environment
    - **Configuration Challenge**: Set up robot model in Gazebo with ROS 2

    ### Week 7 Assessment
    - **Implementation**: Build advanced humanoid robot simulation
    - **Validation Exercise**: Compare simulation vs. real-world behavior
    - **Troubleshooting**: Diagnose and fix simulation problems

    ## Resources

    ### Required Reading
    - "Gazebo Tutorial Series" - OSRF Documentation
    - "Simulation-Based Robot Programming" - Best practices guide
    - URDF/Gazebo Integration Guide

    ### Recommended Tools
    - Gazebo Garden or Fortress
    - RViz2 for visualization
    - rqt for debugging
    - Mesh tools for 3D models

    ### Sample Models
    - Tutorials world models
    - PR2 robot simulation
    - TurtleBot3 simulation examples

    ## Next Steps

    After completing Weeks 6-7, you'll have mastered Gazebo simulation and be able to create realistic digital twins for your robots. You'll be prepared to move on to Module 3: The AI-Robot Brain (NVIDIA Isaac™) in Weeks 8-10, where you'll learn to integrate AI capabilities with your simulated robots.
  </div>
  <div className="urdu">
    # ہفتہ 6-7: Gazebo کے ساتھ روبوٹ سیمولیشن

    ## جائزہ

    Physical AI اور ہیومنائیڈ روبوٹکس کورس کے ہفتہ 6-7 میں خوش آمدید! ان ہفتوں کے دوران، آپ روبوٹکس کے لیے ایک سیمولیشن ماحول کے طور پر Gazebo کی طاقت کو دریافت کریں گے۔ Gazebo طبعی روبوٹس کے لیے ڈیجیٹل جڑواں (digital twin) کے طور پر کام کرتا ہے، جو آپ کو حقیقی ہارڈویئر پر تعینات کرنے سے پہلے اپنے الگورتھم کو محفوظ، قابل کنٹرول اور قابل تکرار ماحول میں ٹیسٹ اور تصدیق کرنے کی اجازت دیتا ہے۔

    ## سیکھنے کے مقاصد

    ہفتہ 6-7 کے اختتام تک، آپ اس قابل ہو جائیں گے:

    1. Gazebo سیمولیشن کے فن تعمیر اور صلاحیتوں کو سمجھیں۔
    2. Gazebo سیمولیشن کے لیے روبوٹ ماڈل بنائیں اور ترتیب دیں۔
    3. کسٹم سیمولیشن ماحول ڈیزائن اور تعمیر کریں۔
    4. حقیقت پسندانہ روبوٹ سیمولیشن کے لیے Gazebo کو ROS 2 کے ساتھ مربوط کریں۔
    5. درست سیمولیشن کے لیے طبیعیات کی خصوصیات اور سینسر ماڈلز کو ترتیب دیں۔
    6. کیمرے، LIDAR، اور IMUs سمیت سینسر سیمولیشن کو نافذ کریں۔
    7. حقیقی دنیا میں تعیناتی سے پہلے سیمولیشن میں روبوٹ کے رویوں کی تصدیق کریں۔
  </div>
</BilingualChapter>
