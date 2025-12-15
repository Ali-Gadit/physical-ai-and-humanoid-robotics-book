---
id: urdf-humanoids
title: "Understanding URDF for Humanoids"
sidebar_position: 4
---

import BilingualChapter from '@site/src/components/BilingualChapter';

<BilingualChapter>
  <div className="english">
    # Understanding URDF for Humanoids

    ## Introduction

    URDF (Unified Robot Description Format) is an XML-based format used in ROS to describe robot models. For humanoid robots, URDF is essential as it defines the robot's physical structure, including links (rigid bodies), joints (connections between links), and their properties. Understanding URDF is crucial for simulating, visualizing, and controlling humanoid robots in ROS 2.

    In the context of Physical AI, URDF serves as the digital blueprint that allows AI systems to understand the robot's physical structure and constraints, enabling proper motion planning, control, and interaction with the environment.

    ## URDF Basics

    ### What is URDF?

    URDF stands for Unified Robot Description Format. It's an XML-based format that describes robot models in terms of:

    - **Links**: Rigid bodies that make up the robot
    - **Joints**: Connections between links that allow relative motion
    - **Visual**: How the robot looks in simulation
    - **Collision**: How the robot interacts with the environment
    - **Inertial**: Physical properties for simulation

    ### Basic URDF Structure

    Here's a minimal URDF example:

    ```xml
    <?xml version="1.0"?>
    <robot name="simple_robot">
      <!-- Base link -->
      <link name="base_link">
        <visual>
          <geometry>
            <box size="1 1 1"/>
          </geometry>
        </visual>
        <collision>
          <geometry>
            <box size="1 1 1"/>
          </geometry>
        </collision>
        <inertial>
          <mass value="1"/>
          <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
      </link>
    </robot>
    ```

    ## URDF for Humanoid Robots

    Humanoid robots have a more complex structure than simple wheeled robots. A typical humanoid has:

    - Torso (trunk)
    - Head
    - Two arms (each with shoulder, elbow, wrist joints)
    - Two legs (each with hip, knee, ankle joints)
    - Possibly articulated hands and feet

    ### Humanoid Joint Structure

    Here's an example of a simplified humanoid leg structure in URDF:

    ```xml
    <?xml version="1.0"?>
    <robot name="humanoid_robot">
      <!-- Torso -->
      <link name="torso">
        <visual>
          <geometry>
            <box size="0.3 0.2 0.5"/>
          </geometry>
        </visual>
        <collision>
          <geometry>
            <box size="0.3 0.2 0.5"/>
          </geometry>
        </collision>
        <inertial>
          <mass value="10"/>
          <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
      </link>

      <!-- Left Hip Joint -->
      <joint name="left_hip_joint" type="revolute">
        <parent link="torso"/>
        <child link="left_thigh"/>
        <origin xyz="0 -0.1 -0.25" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
      </joint>

      <!-- Left Thigh -->
      <link name="left_thigh">
        <visual>
          <geometry>
            <cylinder length="0.4" radius="0.05"/>
          </geometry>
        </visual>
        <collision>
          <geometry>
            <cylinder length="0.4" radius="0.05"/>
          </geometry>
        </collision>
        <inertial>
          <mass value="2"/>
          <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.01"/>
        </inertial>
      </link>

      <!-- Left Knee Joint -->
      <joint name="left_knee_joint" type="revolute">
        <parent link="left_thigh"/>
        <child link="left_shin"/>
        <origin xyz="0 0 -0.2" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="0" upper="2.35" effort="100" velocity="1"/>
      </joint>

      <!-- Left Shin -->
      <link name="left_shin">
        <visual>
          <geometry>
            <cylinder length="0.4" radius="0.05"/>
          </geometry>
        </visual>
        <collision>
          <geometry>
            <cylinder length="0.4" radius="0.05"/>
          </geometry>
        </collision>
        <inertial>
          <mass value="1.5"/>
          <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.01"/>
        </inertial>
      </link>
    </robot>
    ```

    ## Key URDF Components for Humanoids

    ### 1. Links

    Links represent rigid bodies in the robot. For humanoids, typical links include:

    - `base_link` or `torso`: The main body
    - `head`: The head link
    - `left_arm_base`, `right_arm_base`: Arm base links
    - `left_upper_arm`, `right_upper_arm`: Upper arm links
    - `left_lower_arm`, `right_lower_arm`: Lower arm links
    - `left_hand`, `right_hand`: Hand links
    - `left_leg_base`, `right_leg_base`: Leg base links
    - `left_upper_leg`, `right_upper_leg`: Upper leg links
    - `left_lower_leg`, `right_lower_leg`: Lower leg links
    - `left_foot`, `right_foot`: Foot links

    ### 2. Joints

    Joints define how links connect and move relative to each other. For humanoids, common joint types include:

    - **Revolute**: Rotational joints (like human joints)
    - **Continuous**: Like revolute but unlimited rotation
    - **Prismatic**: Linear sliding joints
    - **Fixed**: No movement (for sensors or attachments)

    ### 3. Visual and Collision Elements

    These define how the robot appears and interacts:

    ```xml
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="package://humanoid_description/meshes/head.dae"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="package://humanoid_description/meshes/head_collision.dae"/>
      </geometry>
    </collision>
    ```

    ### 4. Inertial Properties

    These are crucial for physics simulation:

    ```xml
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    ```

    ## Xacro for Complex Humanoid URDFs

    For complex humanoid robots, Xacro (XML Macros) is often used to simplify URDF creation:

    ```xml
    <?xml version="1.0"?>
    <robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid">

      <!-- Define properties -->
      <xacro:property name="M_PI" value="3.1415926535897931" />

      <!-- Macro for creating a limb -->
      <xacro:macro name="arm" params="side">
        <link name="${side}_shoulder">
          <visual>
            <geometry>
              <sphere radius="0.05"/>
            </geometry>
          </visual>
        </link>

        <joint name="${side}_shoulder_joint" type="revolute">
          <parent link="torso"/>
          <child link="${side}_shoulder"/>
          <origin xyz="0 ${-0.1 if side == 'left' else 0.1} 0.3" rpy="0 0 0"/>
          <axis xyz="0 0 1"/>
          <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="1"/>
        </joint>
      </xacro:macro>

      <!-- Use the macro to create both arms -->
      <xacro:arm side="left"/>
      <xacro:arm side="right"/>

    </robot>
    ```

    ## URDF and ROS 2 Integration

    ### Robot State Publisher

    The `robot_state_publisher` node reads the URDF and publishes joint states:

    ```python
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from tf2_ros import TransformBroadcaster
    from geometry_msgs.msg import TransformStamped

    class JointStatePublisher(Node):
        def __init__(self):
            super().__init__('joint_state_publisher')
            self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
            self.tf_broadcaster = TransformBroadcaster(self)
            self.timer = self.create_timer(0.1, self.publish_joint_states)

        def publish_joint_states(self):
            msg = JointState()
            msg.name = ['joint1', 'joint2', 'joint3']
            msg.position = [0.0, 0.0, 0.0]  # Current joint positions
            msg.header.stamp = self.get_clock().now().to_msg()
            self.joint_pub.publish(msg)
    ```

    ## Best Practices for Humanoid URDF

    ### 1. Proper Naming Conventions

    Use consistent naming:
    - `left_arm_shoulder_pitch`, `left_arm_shoulder_roll`, `left_arm_shoulder_yaw`
    - `right_leg_hip_pitch`, `right_leg_hip_roll`, `right_leg_hip_yaw`

    ### 2. Accurate Inertial Properties

    For stable simulation, ensure:
    - Mass values are realistic
    - Center of mass is properly positioned
    - Inertia tensors are calculated correctly

    ### 3. Joint Limits

    Set appropriate joint limits based on the physical robot:
    - Human-like ranges for humanoid robots
    - Safety margins to prevent damage

    ### 4. Transmission Elements

    Include transmission elements for controller integration:

    ```xml
    <transmission name="left_elbow_trans">
      <type>transmission_interface/SimpleTransmission</type>
      <joint name="left_elbow_joint">
        <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      </joint>
      <actuator name="left_elbow_motor">
        <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
        <mechanicalReduction>1</mechanicalReduction>
      </actuator>
    </transmission>
    ```

    ## Tools for Working with URDF

    ### 1. rviz2
    Visualize your URDF in RViz2:
    ```bash
    ros2 run rviz2 rviz2
    ```

    ### 2. Check URDF
    Validate your URDF:
    ```bash
    check_urdf /path/to/your/robot.urdf
    ```

    ### 3. Joint State Publisher GUI
    For testing joint movements:
    ```bash
    ros2 run joint_state_publisher_gui joint_state_publisher_gui
    ```

    ## Common Issues and Troubleshooting

    ### 1. TF Tree Issues
    Ensure your URDF creates a proper tree structure (no loops).

    ### 2. Joint Direction
    Make sure joint axes are oriented correctly for positive/negative movement.

    ### 3. Collision Detection
    Ensure collision geometries are properly defined for physics simulation.

    ## Hands-on Exercise

    Create a simplified URDF for a humanoid robot that includes:
    1. A torso with head
    2. Two arms with 3 joints each
    3. Two legs with 3 joints each
    4. Proper inertial properties
    5. Visual and collision elements

    This will give you hands-on experience with creating URDF files for humanoid robots and understanding how they integrate with ROS 2 systems.
  </div>
  <div className="urdu">
    # Humanoids کے لیے URDF

    ## URDF کیا ہے؟

    URDF (Unified Robot Description Format) ایک XML فارمیٹ ہے جو روبوٹ کی حرکیاتی (kinematic) اور متحرک (dynamic) خصوصیات کو بیان کرنے کے لیے استعمال ہوتا ہے۔

    ## Humanoid کی وضاحت

    ایک humanoid robot لنکس (ہڈیوں) کا ایک درخت جیسا ڈھانچہ ہے جو جوڑوں (پٹھوں) کے ذریعے جڑا ہوتا ہے۔

    ### URDF کا نمونہ

    ```xml
    <robot name="simple_humanoid">
      <link name="base_link">
        <visual>
          <geometry>
            <box size="0.5 0.5 0.2"/>
          </geometry>
        </visual>
      </link>

      <joint name="torso_joint" type="revolute">
        <parent link="base_link"/>
        <child link="torso"/>
        <origin xyz="0 0 0.2"/>
        <axis xyz="0 0 1"/>
      </joint>
    </robot>
    ```

    ## اہم ٹیگز (Key Tags)

    *   **`<link>`**: جسم کے سخت حصے کی نمائندگی کرتا ہے۔
    *   **`<joint>`**: دو لنکس کو جوڑتا ہے اور حرکت کی حدود کی وضاحت کرتا ہے۔
    *   **`<visual>`**: روبوٹ کیسا لگتا ہے۔
    *   **`<collision>`**: فزکس انجنوں کے لیے جسمانی حد۔
  </div>
</BilingualChapter>
