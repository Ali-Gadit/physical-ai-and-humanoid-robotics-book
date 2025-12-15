---
id: weeks3-5-ros2
title: "Weeks 3-5 - ROS 2 Fundamentals"
sidebar_position: 2
---

import BilingualChapter from '@site/src/components/BilingualChapter';

<BilingualChapter>
  <div className="english">
    # Weeks 3-5: ROS 2 Fundamentals

    ## Overview

    Welcome to Weeks 3-5 of the Physical AI & Humanoid Robotics course! This module provides a comprehensive introduction to Robot Operating System 2 (ROS 2), which serves as the nervous system for robotic platforms. Just as the nervous system coordinates the human body's responses to stimuli, ROS 2 provides the communication infrastructure that allows different components of a robot to work together seamlessly.

    ROS 2 is not an operating system in the traditional sense, but rather a middleware framework that provides services such as hardware abstraction, device drivers, libraries, visualizers, message-passing, package management, and more. It's the foundation upon which all other robotic capabilities are built.

    ## Learning Objectives

    By the end of Weeks 3-5, you will be able to:

    1. Understand the architecture and core concepts of ROS 2
    2. Create and manage ROS 2 nodes for different robot components
    3. Implement communication between nodes using topics and services
    4. Use rclpy to bridge Python agents to ROS controllers
    5. Understand and work with URDF (Unified Robot Description Format) for humanoid robots
    6. Develop basic ROS 2 packages with Python
    7. Design and implement distributed robot systems using ROS 2 patterns

    ## Week 3: ROS 2 Architecture and Core Concepts

    ### Day 1: Introduction to ROS 2 Architecture

    #### What is ROS 2?

    ROS 2 is the next-generation Robot Operating System designed for building robot applications. It addresses the limitations of ROS 1 and adds features like:

    - **Real-time support**: Deterministic behavior for time-critical applications
    - **Multi-robot systems**: Native support for multiple robots coordination
    - **Security**: Built-in security features and authentication
    - **Cross-platform compatibility**: Support for Linux, Windows, and macOS
    - **Production deployment**: Designed for commercial applications

    #### Core Architecture Components

    **Nodes**: Independent processes that perform computation. Nodes are organized into packages and communicate with other nodes via messages.

    **Topics**: Named buses over which nodes exchange messages. Topics implement a publish-subscribe communication pattern.

    **Services**: Synchronous request-response communication pattern between nodes.

    **Actions**: Long-running tasks with feedback and goal management capabilities.

    **Parameters**: Configuration values that can be set at runtime and shared across nodes.

    #### Communication Patterns in ROS 2

    ```
    Node A ── Publish ──► Topic ── Subscribe ── Node B
             (Messages)                  (Messages)

    Node A ── Request ──► Service ── Response ── Node B
             (Sync)        Server     (Sync)

    Node A ── Action Goal ──► Action Server ── Feedback ── Node A
             (Async)                           (Continuous)
    ```

    ### Day 2: ROS 2 Ecosystem and Tools

    #### Essential ROS 2 Tools

    - **ros2 run**: Execute a node from a package
    - **ros2 topic**: Inspect and debug topics
    - **ros2 service**: Interact with services
    - **ros2 action**: Work with actions
    - **ros2 param**: Manage parameters
    - **rqt**: GUI tools for visualization and debugging
    - **rviz2**: 3D visualization tool for robot data

    #### Package Management

    ROS 2 uses the colcon build system for managing packages:

    - **ament_cmake**: Build system for C/C++ packages
    - **ament_python**: Build system for Python packages
    - **colcon build**: Build all packages in a workspace
    - **colcon test**: Run tests for packages

    ### Day 3: Creating Your First ROS 2 Node

    #### Python Node Structure

    ```python
    import rclpy
    from rclpy.node import Node

    class MyNode(Node):
        def __init__(self):
            super().__init__('my_node_name')
            # Initialize node components

        def some_function(self):
            # Node functionality
            pass

    def main(args=None):
        rclpy.init(args=args)
        node = MyNode()

        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()

    if __name__ == '__main__':
        main()
    ```

    #### Node Lifecycle

    1. **Initialization**: Set up node with name and parameters
    2. **Setup**: Create publishers, subscribers, services, timers
    3. **Execution**: Process callbacks and perform tasks
    4. **Shutdown**: Clean up resources and exit gracefully

    ### Day 4: Understanding ROS 2 Quality of Service (QoS)

    #### QoS Profiles

    Quality of Service profiles define how messages are delivered between nodes:

    - **Reliability**: Reliable vs. best-effort delivery
    - **Durability**: Volatile vs. transient-local durability
    - **History**: Keep-all vs. keep-last message history
    - **Deadline**: Time constraints for message delivery
    - **Liveliness**: Node availability monitoring

    #### Appropriate QoS Selection

    - **Sensors**: Best-effort, volatile, keep-last (10)
    - **Controls**: Reliable, volatile, keep-last (1)
    - **Configuration**: Reliable, transient-local, keep-all

    ### Day 5: ROS 2 Launch Systems

    #### Launch Files

    Launch files allow you to start multiple nodes with a single command:

    ```xml
    <launch>
      <node pkg="my_package" exec="my_node" name="my_node_instance">
        <param name="param_name" value="param_value"/>
        <remap from="original_topic" to="new_topic"/>
      </node>

      <group>
        <node pkg="another_pkg" exec="another_node" name="node2"/>
      </group>
    </launch>
    ```

    ## Week 4: Topics, Services, and Actions

    ### Day 6: Topics and Message Passing

    #### Publishers and Subscribers

    ```python
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class Talker(Node):
        def __init__(self):
            super().__init__('talker')
            self.publisher = self.create_publisher(String, 'chatter', 10)
            timer_period = 0.5  # seconds
            self.timer = self.create_timer(timer_period, self.timer_callback)

        def timer_callback(self):
            msg = String()
            msg.data = 'Hello World: %d' % self.get_clock().now().nanoseconds
            self.publisher.publish(msg)
            self.get_logger().info('Publishing: "%s"' % msg.data)

    class Listener(Node):
        def __init__(self):
            super().__init__('listener')
            self.subscription = self.create_subscription(
                String,
                'chatter',
                self.listener_callback,
                10)
            self.subscription  # prevent unused variable warning

        def listener_callback(self, msg):
            self.get_logger().info('I heard: "%s"' % msg.data)
    ```

    #### Message Types and Custom Messages

    Standard message types:
    - **std_msgs**: Basic data types (Bool, Int32, Float64, String)
    - **geometry_msgs**: Geometric primitives (Point, Pose, Twist)
    - **sensor_msgs**: Sensor data (Image, LaserScan, JointState)
    - **nav_msgs**: Navigation data (Odometry, Path, OccupancyGrid)

    ### Day 7: Services for Synchronous Communication

    #### Service Client-Server Pattern

    ```python
    # Service definition (add.srv)
    int64 a
    int64 b
    ---
    int64 sum

    # Service server
    from example_interfaces.srv import AddTwoInts

    class AddTwoIntsService(Node):
        def __init__(self):
            super().__init__('add_two_ints_service')
            self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_callback)

        def add_callback(self, request, response):
            response.sum = request.a + request.b
            self.get_logger().info(f'Returning {response.sum}')
            return response

    # Service client
    class AddTwoIntsClient(Node):
        def __init__(self):
            super().__init__('add_two_ints_client')
            self.cli = self.create_client(AddTwoInts, 'add_two_ints')

        def send_request(self, a, b):
            while not self.cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Service not available, waiting again...')

            request = AddTwoInts.Request()
            request.a = a
            request.b = b
            self.future = self.cli.call_async(request)
            rclpy.spin_until_future_complete(self, self.future)
            return self.future.result()
    ```

    ### Day 8: Actions for Long-Running Tasks

    #### Action Structure

    Actions combine the benefits of services and topics for long-running operations:

    - **Goal**: Request for the action to start
    - **Feedback**: Continuous updates during execution
    - **Result**: Final outcome when completed

    ```python
    import rclpy
    from rclpy.action import ActionServer
    from rclpy.node import Node
    from example_interfaces.action import Fibonacci

    class FibonacciActionServer(Node):
        def __init__(self):
            super().__init__('fibonacci_action_server')
            self._action_server = ActionServer(
                self,
                Fibonacci,
                'fibonacci',
                self.execute_callback)

        def execute_callback(self, goal_handle):
            self.get_logger().info('Executing goal...')

            feedback_msg = Fibonacci.Feedback()
            feedback_msg.sequence = [0, 1]

            for i in range(1, goal_handle.request.order):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    self.get_logger().info('Goal canceled')
                    return Fibonacci.Result()

                feedback_msg.sequence.append(
                    feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

                goal_handle.publish_feedback(feedback_msg)

            goal_handle.succeed()
            result = Fibonacci.Result()
            result.sequence = feedback_msg.sequence
            return result
    ```

    ### Day 9: Advanced Communication Patterns

    #### Publisher-Subscriber with Transformers

    Using tf2 for coordinate transformations:

    ```python
    import rclpy
    from rclpy.node import Node
    from tf2_ros import TransformBroadcaster
    from geometry_msgs.msg import TransformStamped

    class FramePublisher(Node):
        def __init__(self):
            super().__init__('frame_publisher')
            self.tf_broadcaster = TransformBroadcaster(self)
            self.timer = self.create_timer(0.1, self.broadcast_transform)

        def broadcast_transform(self):
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'turtle1'
            t.child_frame_id = 'carrot1'
            t.transform.translation.x = 4.0
            t.transform.translation.y = 2.0
            t.transform.translation.z = 0.0
            # Set rotation
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            self.tf_broadcaster.sendTransform(t)
    ```

    #### Parameter Server Integration

    ```python
    class ParameterNode(Node):
        def __init__(self):
            super().__init__('parameter_node')

            # Declare parameters with defaults
            self.declare_parameter('robot_name', 'turtlebot')
            self.declare_parameter('max_velocity', 0.5)

            # Get parameter values
            self.robot_name = self.get_parameter('robot_name').value
            self.max_velocity = self.get_parameter('max_velocity').value

            # Parameter callback
            self.add_on_set_parameters_callback(self.parameter_callback)

        def parameter_callback(self, params):
            for param in params:
                if param.name == 'max_velocity' and param.value > 1.0:
                    return SetParametersResult(successful=False, reason="Max velocity too high")
            return SetParametersResult(successful=True)
    ```

    ### Day 10: Designing Robust Communication Systems

    #### Error Handling and Recovery

    Best practices for resilient ROS 2 communication:
    - Implement timeouts for service calls
    - Use appropriate QoS settings for different data types
    - Handle connection and disconnection events
    - Implement retry logic for critical communications
    - Monitor network health and performance

    ## Week 5: Python Integration and URDF

    ### Day 11: Python Agents to ROS Controllers

    #### Bridging Python AI Agents to ROS

    ```python
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import LaserScan
    import numpy as np

    class AIAgentNode(Node):
        def __init__(self):
            super().__init__('ai_agent_node')

            # Publishers for robot control
            self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

            # Subscribers for sensor data
            self.scan_sub = self.create_subscription(
                LaserScan, '/scan', self.scan_callback, 10)

            # AI agent initialization
            self.ai_model = self.initialize_ai_model()
            self.latest_scan = None

        def scan_callback(self, msg):
            self.latest_scan = msg
            if self.should_act():
                self.take_action()

        def initialize_ai_model(self):
            # Initialize your AI model here
            return None

        def should_act(self):
            # Determine if the AI agent should take action
            return self.latest_scan is not None

        def take_action(self):
            # Process sensor data through AI model
            action = self.ai_model.predict(self.latest_scan)

            # Convert to ROS message
            cmd_msg = Twist()
            cmd_msg.linear.x = action.linear_velocity
            cmd_msg.angular.z = action.angular_velocity

            self.cmd_vel_pub.publish(cmd_msg)
    ```

    #### Real-time Performance Considerations

    - Minimize Python object allocation in critical loops
    - Use efficient data structures for sensor processing
    - Consider using Cython or C++ for performance-critical components
    - Profile and optimize communication patterns
    - Implement proper threading for blocking operations

    ### Day 12: Understanding URDF for Humanoids

    #### Unified Robot Description Format (URDF)

    URDF is an XML format for representing robot models, including:
    - **Links**: Rigid parts of the robot
    - **Joints**: Connections between links
    - **Visual**: Geometry for visualization
    - **Collision**: Collision geometry for physics simulation
    - **Inertial**: Mass and inertia properties

    #### Basic URDF Structure

    ```xml
    <?xml version="1.0"?>
    <robot name="simple_humanoid">
      <!-- Base Link -->
      <link name="base_link">
        <visual>
          <geometry>
            <cylinder length="0.6" radius="0.2"/>
          </geometry>
          <material name="blue">
            <color rgba="0 0 0.8 1"/>
          </material>
        </visual>
        <collision>
          <geometry>
            <cylinder length="0.6" radius="0.2"/>
          </geometry>
        </collision>
        <inertial>
          <mass value="10"/>
          <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
        </inertial>
      </link>

      <!-- Head Link -->
      <link name="head">
        <visual>
          <geometry>
            <sphere radius="0.1"/>
          </geometry>
          <material name="white">
            <color rgba="1 1 1 1"/>
          </material>
        </visual>
      </link>

      <!-- Joint connecting base to head -->
      <joint name="neck_joint" type="revolute">
        <parent link="base_link"/>
        <child link="head"/>
        <origin xyz="0 0 0.4" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-0.5" upper="0.5" effort="100" velocity="1"/>
      </joint>
    </robot>
    ```

    ### Day 13: Advanced URDF for Humanoid Robots

    #### Humanoid Kinematic Chains

    Humanoid robots typically have complex kinematic chains:

    - **Torso**: Main body with head and hip connections
    - **Legs**: Hip, knee, ankle joints forming leg chains
    - **Arms**: Shoulder, elbow, wrist joints for manipulation
    - **Hands**: Multiple joints for dexterous manipulation

    #### Transmission Elements

    ```xml
    <transmission name="wheel_trans">
      <type>transmission_interface/SimpleTransmission</type>
      <joint name="wheel_joint">
        <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
      </joint>
      <actuator name="wheel_motor">
        <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
        <mechanicalReduction>1</mechanicalReduction>
      </actuator>
    </transmission>
    ```

    ### Day 14: URDF Best Practices for Physical AI

    #### Physical Accuracy

    For Physical AI applications, URDF models must accurately represent:
    - **Mass distribution**: Proper inertial tensors for realistic physics
    - **Friction coefficients**: Accurate surface properties
    - **Collision geometry**: Detailed meshes for precise collision detection
    - **Joint limits**: Realistic range of motion constraints
    - **Center of gravity**: Critical for balance and stability

    #### Simulation vs. Real Robot

    Considerations for sim-to-real transfer:
    - Use identical URDF for both simulation and real robot
    - Include realistic sensor noise models
    - Account for actuator dynamics and delays
    - Model flexible components and compliance
    - Include environmental factors (friction, damping)

    ### Day 15: Integration and Testing

    #### Testing ROS 2 Nodes

    ```python
    import unittest
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import String

    class TestROSNode(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            rclpy.init()

        @classmethod
        def tearDownClass(cls):
            rclpy.shutdown()

        def test_publisher_subscriber(self):
            # Create publisher and subscriber nodes
            publisher_node = TestPublisher()
            subscriber_node = TestSubscriber()

            executor = SingleThreadedExecutor()
            executor.add_node(publisher_node)
            executor.add_node(subscriber_node)

            # Send a message
            publisher_node.publish_message('test message')

            # Spin to process messages
            executor.spin_once(timeout_sec=1.0)

            # Check received message
            self.assertEqual(subscriber_node.received_message, 'test message')
    ```

    #### Performance Testing

    Evaluate ROS 2 system performance:
    - **Message latency**: Time from publication to receipt
    - **Bandwidth utilization**: Data throughput under load
    - **CPU usage**: Resource consumption of nodes
    - **Memory footprint**: Memory usage over time
    - **Real-time performance**: Deadline meeting consistency

    ## Hands-On Projects

    ### Week 3 Project: Basic ROS 2 Node Development

    1. Create a publisher node that broadcasts robot status
    2. Create a subscriber node that processes sensor data
    3. Implement a service server for basic robot commands
    4. Use rqt to visualize and debug your nodes
    5. Write a launch file to start all nodes together

    ### Week 4 Project: Advanced Communication Systems

    1. Implement an action server for navigation tasks
    2. Create a client that monitors action progress
    3. Build a parameter server for robot configuration
    4. Design a fault-tolerant communication pattern
    5. Test with simulated network conditions

    ### Week 5 Project: Humanoid Robot Integration

    1. Create a URDF model for a simple humanoid robot
    2. Implement Python nodes for sensor processing
    3. Bridge an AI model to control the robot
    4. Integrate with Gazebo simulation
    5. Test basic locomotion patterns

    ## Assessment

    ### Week 3 Assessment
    - **Quiz**: ROS 2 architecture and core concepts
    - **Lab Exercise**: Create basic publisher-subscriber nodes
    - **Code Review**: Evaluate node structure and design patterns

    ### Week 4 Assessment
    - **Implementation**: Build service and action-based systems
    - **Debugging Challenge**: Fix communication problems in provided code
    - **Design Exercise**: Plan communication architecture for a robot system

    ### Week 5 Assessment
    - **URDF Creation**: Design humanoid robot model
    - **Integration Project**: Connect AI agent to ROS control
    - **Performance Analysis**: Measure and optimize communication systems

    ## Resources

    ### Required Reading
    - "Programming Robots with ROS" by Morgan Quigley
    - ROS 2 Documentation: Core Concepts
    - URDF Tutorials and Best Practices

    ### Recommended Tools
    - RViz2 for visualization
    - rqt for debugging
    - Gazebo for simulation
    - URDF validators and parsers

    ### Sample Code Repositories
    - ROS 2 Tutorials
    - MoveIt! for manipulation
    - Navigation2 for navigation

    ## Next Steps

    After completing Weeks 3-5, you'll have a solid understanding of ROS 2 fundamentals and be able to design and implement distributed robot systems. You'll be prepared to move on to Module 2: The Digital Twin (Gazebo & Unity) in Weeks 6-7, where you'll learn to simulate your ROS-controlled robots in realistic environments.
  </div>
  <div className="urdu">
    # ہفتہ 3-5: ROS 2 کے بنیادی اصول

    ## جائزہ

    Physical AI اور ہیومنائیڈ روبوٹکس کورس کے ہفتہ 3-5 میں خوش آمدید! یہ ماڈیول Robot Operating System 2 (ROS 2) کا ایک جامع تعارف فراہم کرتا ہے، جو روبوٹک پلیٹ فارمز کے لیے اعصابی نظام (nervous system) کے طور پر کام کرتا ہے۔ جس طرح اعصابی نظام محرکات کے بارے میں انسانی جسم کے ردعمل کو مربوط کرتا ہے، ROS 2 مواصلاتی بنیادی ڈھانچہ فراہم کرتا ہے جو روبوٹ کے مختلف اجزاء کو بغیر کسی رکاوٹ کے ایک ساتھ کام کرنے کی اجازت دیتا ہے۔

    ## سیکھنے کے مقاصد

    ہفتہ 3-5 کے اختتام تک، آپ اس قابل ہو جائیں گے:

    1. ROS 2 کے فن تعمیر اور بنیادی تصورات کو سمجھیں۔
    2. مختلف روبوٹ اجزاء کے لیے ROS 2 نوڈز بنائیں اور ان کا نظم کریں۔
    3. topics اور services کا استعمال کرتے ہوئے نوڈز کے درمیان مواصلات کو نافذ کریں۔
    4. Python ایجنٹس کو ROS کنٹرولرز سے جوڑنے کے لیے rclpy استعمال کریں۔
    5. ہیومنائیڈ روبوٹس کے لیے URDF کو سمجھیں اور اس کے ساتھ کام کریں۔
    6. Python کے ساتھ بنیادی ROS 2 پیکجز تیار کریں۔
    7. ROS 2 پیٹرنز کا استعمال کرتے ہوئے تقسیم شدہ روبوٹ سسٹمز کو ڈیزائن اور نافذ کریں۔
  </div>
</BilingualChapter>
