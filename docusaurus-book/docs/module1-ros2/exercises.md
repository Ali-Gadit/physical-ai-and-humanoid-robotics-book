---
id: exercises
title: "Module 1 Practical Exercises"
sidebar_position: 5
---

import BilingualChapter from '@site/src/components/BilingualChapter';

<BilingualChapter>
  <div className="english">
    # Module 1 Practical Exercises

    ## Overview

    This section contains hands-on exercises to reinforce your understanding of ROS 2 fundamentals, including nodes, topics, services, Python integration, and URDF for humanoid robots. Complete these exercises to gain practical experience with the concepts covered in Module 1.

    ## Exercise 1: Basic ROS 2 Publisher and Subscriber

    ### Objective
    Create a simple publisher-subscriber pair that demonstrates ROS 2 communication.

    ### Instructions
    1. Create a new ROS 2 package called `module1_exercises`
    2. Create a publisher node that publishes a custom message containing:
       - Robot name (string)
       - Current position (x, y coordinates as floats)
       - Timestamp
    3. Create a subscriber node that receives this message and logs the information
    4. Test the communication by running both nodes simultaneously

    ### Code Template
    ```python
    # publisher_node.py
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    import time

    class RobotPublisher(Node):
        def __init__(self):
            super().__init__('robot_publisher')
            self.publisher = self.create_publisher(String, 'robot_status', 10)
            self.timer = self.create_timer(1.0, self.publish_status)
            self.counter = 0

        def publish_status(self):
            msg = String()
            msg.data = f"Robot-001, Position: ({self.counter}, {self.counter*0.5}), Time: {time.time()}"
            self.publisher.publish(msg)
            self.get_logger().info(f'Publishing: {msg.data}')
            self.counter += 1

    def main(args=None):
        rclpy.init(args=args)
        publisher = RobotPublisher()
        rclpy.spin(publisher)
        publisher.destroy_node()
        rclpy.shutdown()

    if __name__ == '__main__':
        main()
    ```

    ### Expected Output
    - Publisher node publishing messages every second
    - Subscriber node receiving and logging messages
    - Clean shutdown when interrupted

    ### Evaluation Criteria
    - Nodes communicate successfully
    - Messages are properly formatted
    - Error handling is implemented
    - Code follows ROS 2 best practices

    ## Exercise 2: Service-Based Robot Control

    ### Objective
    Implement a service-based system for controlling a simulated robot.

    ### Instructions
    1. Define a custom service called `RobotControl.srv` with:
       - Request: `string command` (e.g., "move_forward", "turn_left", "stop")
       - Response: `bool success`, `string message`
    2. Create a service server that simulates robot responses
    3. Create a client node that sends commands to the service
    4. Test with various commands and verify responses

    ### Code Template
    ```python
    # robot_control_server.py
    from module1_exercises.srv import RobotControl
    import rclpy
    from rclpy.node import Node

    class RobotControlServer(Node):
        def __init__(self):
            super().__init__('robot_control_server')
            self.srv = self.create_service(
                RobotControl,
                'robot_control',
                self.handle_robot_control
            )
            self.get_logger().info('Robot Control Service is ready')

        def handle_robot_control(self, request, response):
            self.get_logger().info(f'Received command: {request.command}')

            # Simulate robot response based on command
            if request.command in ['move_forward', 'turn_left', 'turn_right', 'stop']:
                response.success = True
                response.message = f'Command {request.command} executed successfully'
            else:
                response.success = False
                response.message = f'Unknown command: {request.command}'

            return response

        def main(args=None):
            rclpy.init(args=args)
            server = RobotControlServer()
            rclpy.spin(server)
            server.destroy_node()
            rclpy.shutdown()

        if __name__ == '__main__':
            main()
    ```

    ### Expected Output
    - Service responds appropriately to different commands
    - Invalid commands are handled gracefully
    - Success/failure responses are accurate

    ## Exercise 3: Python AI Agent Integration

    ### Objective
    Create a Python-based decision-making agent that controls a simulated robot.

    ### Instructions
    1. Create a Python node that receives sensor data (simulated as random values)
    2. Implement a simple decision-making algorithm (e.g., obstacle avoidance)
    3. Publish appropriate control commands based on the algorithm
    4. Include logging to track the agent's decisions

    ### Code Template
    ```python
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Float32MultiArray
    import random

    class DecisionMakingAgent(Node):
        def __init__(self):
            super().__init__('decision_agent')

            # Publisher for robot commands
            self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

            # Subscriber for sensor data
            self.sensor_subscriber = self.create_subscription(
                Float32MultiArray,
                'sensor_data',
                self.sensor_callback,
                10
            )

            # Timer for decision loop
            self.timer = self.create_timer(0.5, self.decision_loop)

            # Agent state
            self.sensor_data = [1.0, 1.0, 1.0]  # [front, left, right]
            self.get_logger().info('Decision Making Agent initialized')

        def sensor_callback(self, msg):
            """Update sensor data - in real scenario, this would come from actual sensors"""
            self.sensor_data = list(msg.data)
            if not self.sensor_data:
                self.sensor_data = [1.0, 1.0, 1.0]  # Default values

        def decision_loop(self):
            """Main decision-making loop"""
            cmd = Twist()

            # Simple obstacle avoidance logic
            front_dist, left_dist, right_dist = self.sensor_data

            if front_dist < 0.5:  # Obstacle in front
                if left_dist > right_dist:
                    cmd.angular.z = 0.5  # Turn left
                    self.get_logger().info('Obstacle ahead, turning left')
                else:
                    cmd.angular.z = -0.5  # Turn right
                    self.get_logger().info('Obstacle ahead, turning right')
            else:
                cmd.linear.x = 0.5  # Move forward
                self.get_logger().info('Clear path, moving forward')

            self.cmd_publisher.publish(cmd)

    def main(args=None):
        rclpy.init(args=args)
        agent = DecisionMakingAgent()

        # In a real scenario, you might want to simulate sensor data
        # For this exercise, we'll just run the agent
        rclpy.spin(agent)
        agent.destroy_node()
        rclpy.shutdown()

    if __name__ == '__main__':
        main()
    ```

    ### Expected Output
    - Agent makes decisions based on sensor input
    - Appropriate movement commands are published
    - Decision-making process is logged

    ## Exercise 4: URDF Robot Model Creation

    ### Objective
    Create a URDF model for a simple humanoid robot with basic structure.

    ### Instructions
    1. Create a URDF file for a simplified humanoid with:
       - Torso (main body)
       - Head
       - Two arms (upper and lower)
       - Two legs (upper and lower)
    2. Include proper joint definitions with realistic limits
    3. Add visual and collision elements
    4. Include inertial properties for each link
    5. Test the URDF with RViz2

    ### URDF Template
    ```xml
    <?xml version="1.0"?>
    <robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">

      <!-- Materials -->
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
      <material name="red">
        <color rgba="0.8 0 0 1"/>
      </material>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>

      <!-- Base link -->
      <link name="base_link">
        <visual>
          <geometry>
            <box size="0.3 0.2 0.5"/>
          </geometry>
          <material name="white"/>
        </visual>
        <collision>
          <geometry>
            <box size="0.3 0.2 0.5"/>
          </geometry>
        </collision>
        <inertial>
          <mass value="10"/>
          <inertia ixx="0.5" ixy="0" ixz="0" iyy="0.8" iyz="0" izz="0.6"/>
        </inertial>
      </link>

      <!-- Head -->
      <joint name="neck_joint" type="revolute">
        <parent link="base_link"/>
        <child link="head"/>
        <origin xyz="0 0 0.35" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
      </joint>

      <link name="head">
        <visual>
          <geometry>
            <sphere radius="0.1"/>
          </geometry>
          <material name="white"/>
        </visual>
        <collision>
          <geometry>
            <sphere radius="0.1"/>
          </geometry>
        </collision>
        <inertial>
          <mass value="1"/>
          <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
        </inertial>
      </link>

      <!-- Left Arm -->
      <joint name="left_shoulder_joint" type="revolute">
        <parent link="base_link"/>
        <child link="left_upper_arm"/>
        <origin xyz="0.2 0.1 0.1" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
      </joint>

      <link name="left_upper_arm">
        <visual>
          <geometry>
            <cylinder length="0.3" radius="0.05"/>
          </geometry>
          <material name="blue"/>
        </visual>
        <collision>
          <geometry>
            <cylinder length="0.3" radius="0.05"/>
          </geometry>
        </collision>
        <inertial>
          <mass value="1"/>
          <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005"/>
        </inertial>
      </link>

      <!-- Continue similarly for other joints and links -->

    </robot>
    ```

    ### Expected Output
    - Valid URDF file that loads without errors
    - Robot model displays correctly in RViz2
    - All joints move within defined limits

    ## Exercise 5: Complete Integration Challenge

    ### Objective
    Combine all concepts learned in Module 1 into a complete system.

    ### Instructions
    1. Create a system with:
       - A URDF model of a simple robot
       - Sensor simulation node (publishes fake sensor data)
       - AI decision-making node (processes sensor data)
       - Robot control node (executes commands)
    2. Use the robot state publisher to visualize the robot
    3. Create a launch file to start the entire system
    4. Test the complete system and document behavior

    ### Launch File Template
    ```xml
    <?xml version="1.0"?>
    <launch>
      <!-- Robot State Publisher -->
      <node pkg="robot_state_publisher" exec="robot_state_publisher" name="robot_state_publisher">
        <param name="robot_description" value="$(find-pkg-share module1_exercises)/urdf/simple_robot.urdf"/>
      </node>

      <!-- Sensor Simulator -->
      <node pkg="module1_exercises" exec="sensor_simulator" name="sensor_simulator"/>

      <!-- Decision Agent -->
      <node pkg="module1_exercises" exec="decision_agent" name="decision_agent"/>

      <!-- Robot Controller -->
      <node pkg="module1_exercises" exec="robot_controller" name="robot_controller"/>
    </launch>
    ```

    ### Expected Output
    - All nodes run simultaneously without errors
    - System responds to simulated sensor inputs
    - Robot model updates in RViz2
    - Complete integration works as expected

    ## Assessment Questions

    1. Explain the difference between a ROS 2 topic and a service. When would you use each?

    2. What are the key components of a URDF file? Why are inertial properties important?

    3. How does the rclpy library enable Python agents to interact with ROS 2 systems?

    4. Describe the structure of a humanoid robot in terms of links and joints.

    5. What are some best practices for designing ROS 2 nodes for humanoid robots?

    ## Submission Requirements

    For each exercise, submit:
    - Source code files
    - URDF files (if applicable)
    - Launch files (if applicable)
    - A brief report describing your implementation and any challenges encountered
    - Screenshots of successful execution (RViz2 for URDF, terminal output for nodes)

    ## Evaluation Rubric

    - **Functionality** (40%): Code works as expected and meets requirements
    - **Code Quality** (25%): Follows ROS 2 best practices and is well-structured
    - **Documentation** (20%): Clear comments and explanations
    - **Problem Solving** (15%): Creative solutions and proper error handling

    Complete all exercises to gain hands-on experience with ROS 2 fundamentals essential for Physical AI and humanoid robotics applications.
  </div>
  <div className="urdu">
    # مشقیں: Module 1

    ## مشق 1: Chatter Node

    ایک نیا ROS 2 پیکیج بنائیں اور `chatter` نامی ایک Python node لکھیں جو ہر 2 سیکنڈ میں "ROS 2 is awesome!" شائع کرے۔

    ## مشق 2: Listener Node

    ایک subscriber node لکھیں جو `chatter` ٹاپک کو سنے اور موصول ہونے والے پیغام کو کنسول پر پرنٹ کرے۔

    ## مشق 3: Custom Message

    مندرجہ ذیل فیلڈز کے ساتھ `RobotStatus.msg` نامی ایک کسٹم `.msg` فائل بنائیں:
    *   `int32 battery_level`
    *   `string status`

    اس کسٹم میسج کو بھیجنے کے لیے اپنے publisher کو اپ ڈیٹ کریں۔
  </div>
</BilingualChapter>
