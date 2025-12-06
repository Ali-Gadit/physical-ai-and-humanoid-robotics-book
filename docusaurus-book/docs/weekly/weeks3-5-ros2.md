---
id: weeks3-5-ros2
title: "Weeks 3-5 - ROS 2 Fundamentals"
sidebar_position: 2
---

# Weeks 3-5: ROS 2 Fundamentals

## Overview

During these three weeks, you'll master the Robot Operating System 2 (ROS 2), which serves as the nervous system for robotic platforms. ROS 2 provides the communication infrastructure that allows different components of a robot to work together seamlessly. This is foundational for all subsequent modules, as it provides the middleware framework that enables communication between all the different systems in your humanoid robot.

These weeks will take you from basic ROS 2 concepts to advanced topics including building packages with Python, launch files, and parameter management. You'll learn how ROS 2 handles the complex task of coordinating sensors, actuators, and AI components in a humanoid robot system.

## Learning Objectives

By the end of Weeks 3-5, you will be able to:

1. Understand ROS 2 architecture and core concepts
2. Create and manage ROS 2 nodes for different robot components
3. Implement communication between nodes using topics and services
4. Build ROS 2 packages with Python
5. Use launch files and parameter management effectively
6. Understand how ROS 2 enables the "nervous system" of humanoid robots
7. Bridge Python agents to ROS controllers using rclpy
8. Understand URDF (Unified Robot Description Format) for humanoids

## Week 3: ROS 2 Architecture and Core Concepts

### Day 11: ROS 2 Architecture and Core Concepts

#### What is ROS 2?

ROS 2 (Robot Operating System 2) is not an operating system but rather a middleware framework that provides services such as hardware abstraction, device drivers, libraries, visualizers, message-passing, package management, and more. It's the foundation upon which all other robotic capabilities are built.

**Key Components of ROS 2:**
- **Nodes**: Executables that use ROS 2 client library
- **Topics**: Asynchronous, many-to-many communication mechanism
- **Services**: Synchronous, request-response communication
- **Actions**: Goal-oriented communication with feedback
- **Parameters**: Configuration values stored in nodes
- **Lifecycle**: Node state management for complex systems

#### ROS 2 vs. Traditional Middleware

ROS 2 provides several advantages for robotics:
- **Distributed Computing**: Nodes can run on different machines
- **Language Independence**: Support for multiple programming languages
- **Real-time Support**: Improved real-time capabilities
- **Security**: Built-in security features
- **Quality of Service**: Configurable communication guarantees

#### ROS 2 Client Libraries

ROS 2 supports multiple client libraries:
- **rclcpp**: C++ client library
- **rclpy**: Python client library (primary focus for this course)
- **rclc**: C client library for embedded systems
- **rclrs**: Rust client library
- **rclnodejs**: JavaScript/Node.js client library

### Day 12: Nodes, Topics, and Services

#### ROS 2 Nodes

A node is an executable that uses the ROS 2 client library to communicate with other nodes. Nodes are the basic computational elements of a ROS 2 program.

**Creating a Simple Node:**

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Topics and Publishers/Subscribers

Topics provide asynchronous, many-to-many communication. Publishers send messages to topics, and subscribers receive messages from topics.

**Subscriber Example:**

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Services

Services provide synchronous, request-response communication between nodes.

**Service Example:**

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Day 13: Actions and Parameters

#### Actions

Actions provide goal-oriented communication with feedback, suitable for long-running tasks.

**Action Client Example:**

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(self, Fibonacci, 'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info('Received feedback: {0}'.format(feedback.partial_sequence))

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result: {0}'.format(result.sequence))

def main(args=None):
    rclpy.init(args=args)
    action_client = FibonacciActionClient()
    action_client.send_goal(10)
    rclpy.spin(action_client)

if __name__ == '__main__':
    main()
```

#### Parameters

Parameters allow nodes to be configured dynamically.

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'humanoid_robot')
        self.declare_parameter('max_speed', 0.5)
        self.declare_parameter('safety_distance', 1.0)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_speed = self.get_parameter('max_speed').value
        self.safety_distance = self.get_parameter('safety_distance').value

        self.get_logger().info(f'Robot: {self.robot_name}, Max Speed: {self.max_speed}')

def main(args=None):
    rclpy.init(args=args)
    node = ParameterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Day 14: Quality of Service (QoS) and Communication Patterns

#### Quality of Service

QoS settings control how messages are delivered between nodes.

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String

class QoSPublisher(Node):
    def __init__(self):
        super().__init__('qos_publisher')

        # Create a QoS profile with specific settings
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.publisher_ = self.create_publisher(String, 'qos_topic', qos_profile)

def main(args=None):
    rclpy.init(args=args)
    node = QoSPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Communication Patterns in Humanoid Robots

For humanoid robots, specific communication patterns emerge:
- **Sensor Data Flow**: Sensors publish to topics consumed by perception nodes
- **Control Commands**: Planning nodes publish to actuator command topics
- **State Updates**: Joint state publisher provides robot state to all nodes
- **Behavior Coordination**: Behavior trees coordinate complex actions

### Day 15: ROS 2 Workspace and Package Management

#### Creating a ROS 2 Package

```bash
# Create a new package
ros2 pkg create --build-type ament_python my_robot_package

# Package structure
my_robot_package/
├── my_robot_package/
│   ├── __init__.py
│   └── my_node.py
├── test/
├── setup.py
├── setup.cfg
├── package.xml
└── resource/
```

#### Package.xml Configuration

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Package for my humanoid robot</description>
  <maintainer email="user@example.com">User Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>sensor_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Week 4: Building ROS 2 Packages with Python

### Day 16: Python Node Development

#### Advanced Node Structure

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import math

class HumanoidControllerNode(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Parameters
        self.declare_parameter('control_rate', 50)  # Hz
        self.control_rate = self.get_parameter('control_rate').value

        # Timers
        self.control_timer = self.create_timer(1.0/self.control_rate, self.control_loop)

        # Internal state
        self.joint_positions = {}
        self.desired_positions = {}

        self.get_logger().info('Humanoid Controller initialized')

    def joint_state_callback(self, msg):
        """Update joint position state"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]

    def control_loop(self):
        """Main control loop"""
        # Implement control logic here
        pass

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidControllerNode()

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

### Day 17: Message Types and Custom Messages

#### Creating Custom Messages

To create a custom message for humanoid joint commands:

1. Create a `msg` directory in your package
2. Define the message structure in a `.msg` file

**msg/HumanoidJointCommand.msg:**
```
string[] joint_names
float64[] positions
float64[] velocities
float64[] efforts
uint8[] modes
```

#### Using Custom Messages

```python
import rclpy
from rclpy.node import Node
from my_robot_package.msg import HumanoidJointCommand

class JointCommandPublisher(Node):
    def __init__(self):
        super().__init__('joint_command_publisher')
        self.publisher = self.create_publisher(HumanoidJointCommand, '/humanoid_joint_commands', 10)
        self.timer = self.create_timer(0.1, self.publish_command)

    def publish_command(self):
        msg = HumanoidJointCommand()
        msg.joint_names = ['left_hip', 'left_knee', 'right_hip', 'right_knee']
        msg.positions = [0.0, 0.0, 0.0, 0.0]
        msg.velocities = [0.0, 0.0, 0.0, 0.0]
        msg.efforts = [0.0, 0.0, 0.0, 0.0]
        msg.modes = [0, 0, 0, 0]  # Position control mode

        self.publisher.publish(msg)
```

### Day 18: Launch Files and Parameter Management

#### Creating Launch Files

**launch/humanoid_system.launch.py:**
```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='humanoid',
        description='Namespace for the robot'
    )

    # Get launch configuration
    namespace = LaunchConfiguration('namespace')

    # Create nodes
    controller_node = Node(
        package='my_robot_package',
        executable='humanoid_controller',
        name='controller',
        namespace=namespace,
        parameters=[
            {'control_rate': 100},
            {'max_speed': 0.5}
        ],
        remappings=[
            ('/cmd_vel', 'cmd_vel'),
            ('/joint_commands', 'joint_commands')
        ]
    )

    return LaunchDescription([
        namespace_arg,
        controller_node
    ])
```

#### Advanced Parameter Management

```python
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

class AdvancedParameterNode(Node):
    def __init__(self):
        super().__init__('advanced_parameter_node')

        # Declare parameters with descriptors
        self.declare_parameter(
            'robot_config.body_mass',
            50.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Mass of the robot body in kg',
                floating_point_range=[{'from_value': 1.0, 'to_value': 100.0}]
            )
        )

        self.declare_parameter(
            'robot_config.joint_limits',
            [1.57, 1.57, 1.57],  # hip, knee, ankle
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description='Joint limits for leg joints in radians'
            )
        )

def main(args=None):
    rclpy.init(args=args)
    node = AdvancedParameterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Day 19: Testing and Debugging ROS 2 Nodes

#### Writing Tests

**test/test_my_node.py:**
```python
import unittest
import rclpy
from my_robot_package.my_node import HumanoidControllerNode

class TestHumanoidController(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = HumanoidControllerNode()

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_node_initialization(self):
        """Test that node initializes correctly"""
        self.assertIsNotNone(self.node)
        self.assertEqual(self.node.get_name(), 'humanoid_controller')

if __name__ == '__main__':
    unittest.main()
```

#### Debugging Tools

Common ROS 2 debugging commands:
```bash
# Check node status
ros2 node list
ros2 node info <node_name>

# Check topic status
ros2 topic list
ros2 topic echo <topic_name>
ros2 topic info <topic_name>

# Check service status
ros2 service list
ros2 service call <service_name> <service_type> <request_data>

# Check parameter status
ros2 param list
ros2 param get <node_name> <param_name>
```

### Day 20: ROS 2 for Humanoid Robot Control

#### Humanoid-Specific Communication Patterns

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Float64MultiArray
import numpy as np

class HumanoidCommunicationNode(Node):
    def __init__(self):
        super().__init__('humanoid_communication')

        # Humanoid-specific publishers
        self.left_leg_cmd_pub = self.create_publisher(
            Float64MultiArray, '/left_leg_controller/commands', 10
        )
        self.right_leg_cmd_pub = self.create_publisher(
            Float64MultiArray, '/right_leg_controller/commands', 10
        )
        self.left_arm_cmd_pub = self.create_publisher(
            Float64MultiArray, '/left_arm_controller/commands', 10
        )
        self.right_arm_cmd_pub = self.create_publisher(
            Float64MultiArray, '/right_arm_controller/commands', 10
        )

        # Balance control publisher
        self.balance_cmd_pub = self.create_publisher(
            Float64MultiArray, '/balance_controller/commands', 10
        )

        # Subscribers for humanoid state
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Timer for balance control
        self.balance_timer = self.create_timer(0.01, self.balance_control_loop)  # 100Hz

    def joint_state_callback(self, msg):
        """Process joint state information"""
        # Update internal state for balance control
        pass

    def balance_control_loop(self):
        """Implement balance control logic"""
        # Calculate balance corrections based on joint states
        balance_corrections = self.calculate_balance_corrections()

        # Publish balance commands
        cmd_msg = Float64MultiArray()
        cmd_msg.data = balance_corrections
        self.balance_cmd_pub.publish(cmd_msg)

    def calculate_balance_corrections(self):
        """Calculate balance corrections based on current state"""
        # Implement balance control algorithm
        # This would use IMU data, joint positions, and center of mass
        return [0.0, 0.0, 0.0, 0.0]  # Placeholder values

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidCommunicationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Week 5: Advanced ROS 2 Concepts for Humanoid Robots

### Day 21: rclpy for Python-Robot Integration

#### Advanced rclpy Usage

```python
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from threading import Thread
import asyncio

class AdvancedHumanoidNode(Node):
    def __init__(self):
        super().__init__('advanced_humanoid_node')

        # Create callback groups for different threads
        self.perception_cb_group = MutuallyExclusiveCallbackGroup()
        self.control_cb_group = MutuallyExclusiveCallbackGroup()

        # Publishers and subscribers with callback groups
        self.cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', 10, callback_group=self.control_cb_group
        )

        # Timers with different callback groups
        self.perception_timer = self.create_timer(
            0.033,  # ~30Hz for perception
            self.perception_callback,
            callback_group=self.perception_cb_group
        )

        self.control_timer = self.create_timer(
            0.01,   # 100Hz for control
            self.control_callback,
            callback_group=self.control_cb_group
        )

    def perception_callback(self):
        """High-level perception processing"""
        # Process sensor data for navigation, object detection, etc.
        pass

    def control_callback(self):
        """Low-level control processing"""
        # Send commands to actuators
        pass

def main(args=None):
    rclpy.init(args=args)

    # Use multi-threaded executor to handle different callback groups
    node = AdvancedHumanoidNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Day 22: URDF Integration with ROS 2

#### URDF for Humanoid Robots

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math

class URDFIntegrationNode(Node):
    def __init__(self):
        super().__init__('urdf_integration')

        # Joint state subscriber
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # TF broadcaster for robot transforms
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer for broadcasting transforms
        self.tf_timer = self.create_timer(0.05, self.broadcast_transforms)

        # Store joint positions
        self.joint_positions = {}

    def joint_state_callback(self, msg):
        """Update joint positions from joint state messages"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]

    def broadcast_transforms(self):
        """Broadcast robot transforms based on joint positions"""
        # Example: Broadcast a simple leg transform
        if 'left_hip_joint' in self.joint_positions:
            t = TransformStamped()

            # Header
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'torso'
            t.child_frame_id = 'left_upper_leg'

            # Translation (simplified)
            t.transform.translation.x = 0.0
            t.transform.translation.y = -0.1  # Hip offset
            t.transform.translation.z = 0.0

            # Rotation based on joint position
            hip_angle = self.joint_positions['left_hip_joint']
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = math.sin(hip_angle / 2.0)
            t.transform.rotation.w = math.cos(hip_angle / 2.0)

            self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = URDFIntegrationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Day 23: Advanced Communication Patterns

#### Behavior Trees in ROS 2

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose
import time

class BehaviorTreeNode(Node):
    def __init__(self):
        super().__init__('behavior_tree_node')

        # Publishers for behavior status
        self.behavior_status_pub = self.create_publisher(Bool, '/behavior_active', 10)

        # Timer for behavior tree execution
        self.bt_timer = self.create_timer(0.1, self.execute_behavior_tree)

    def execute_behavior_tree(self):
        """Execute behavior tree logic"""
        # Simple behavior tree example
        if self.check_battery_level() < 0.2:
            self.return_to_charging_station()
        elif self.detect_obstacle():
            self.avoid_obstacle()
        elif self.receive_navigation_goal():
            self.execute_navigation()
        else:
            self.standby_mode()

    def check_battery_level(self):
        """Check battery level"""
        # In real implementation, subscribe to battery status
        return 0.8  # Placeholder

    def return_to_charging_station(self):
        """Return to charging station behavior"""
        self.get_logger().info('Returning to charging station')
        # Implementation for returning to charge

    def detect_obstacle(self):
        """Detect obstacles"""
        # In real implementation, use sensor data
        return False  # Placeholder

    def avoid_obstacle(self):
        """Avoid obstacle behavior"""
        self.get_logger().info('Avoiding obstacle')
        # Implementation for obstacle avoidance

    def receive_navigation_goal(self):
        """Check for navigation goals"""
        # In real implementation, check for navigation goals
        return False  # Placeholder

    def execute_navigation(self):
        """Execute navigation behavior"""
        self.get_logger().info('Executing navigation')
        # Implementation for navigation

    def standby_mode(self):
        """Standby mode behavior"""
        self.get_logger().info('Standing by')
        # Implementation for standby behavior

def main(args=None):
    rclpy.init(args=args)
    node = BehaviorTreeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Day 24: Performance Optimization

#### Optimizing ROS 2 Communications

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
import numpy as np
from collections import deque
import time

class OptimizedCommunicationNode(Node):
    def __init__(self):
        super().__init__('optimized_communication')

        # Optimize publisher settings
        from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy

        # For high-frequency sensor data
        sensor_qos = QoSProfile(
            depth=1,  # Only keep latest message
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT  # For sensor data
        )

        # For critical control commands
        control_qos = QoSProfile(
            depth=10,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE  # For commands
        )

        # Publishers with optimized QoS
        self.optimized_pub = self.create_publisher(
            Image, '/optimized_image', sensor_qos
        )

        # Message throttling
        self.message_counter = 0
        self.throttle_rate = 3  # Send every 3rd message

        # Buffer for processing optimization
        self.processing_buffer = deque(maxlen=10)

        # Timer for optimized processing
        self.optimized_timer = self.create_timer(0.005, self.optimized_callback)  # 200Hz

    def optimized_callback(self):
        """Optimized callback with reduced processing overhead"""
        self.message_counter += 1

        if self.message_counter % self.throttle_rate == 0:
            # Only process every nth message
            self.process_optimized_data()

    def process_optimized_data(self):
        """Optimized data processing"""
        # Minimize memory allocations
        # Use numpy arrays for numerical computations
        # Avoid unnecessary string operations
        pass

def main(args=None):
    rclpy.init(args=args)
    node = OptimizedCommunicationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Day 25: Integration with Humanoid Systems

#### Complete Humanoid ROS 2 System

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String, Float64MultiArray
from tf2_ros import TransformBroadcaster
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np

class CompleteHumanoidSystem(Node):
    def __init__(self):
        super().__init__('complete_humanoid_system')

        # QoS profiles for different data types
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        control_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        # Publishers for different subsystems
        self.left_leg_pub = self.create_publisher(
            Float64MultiArray, '/left_leg_controller/commands', control_qos
        )
        self.right_leg_pub = self.create_publisher(
            Float64MultiArray, '/right_leg_controller/commands', control_qos
        )
        self.left_arm_pub = self.create_publisher(
            Float64MultiArray, '/left_arm_controller/commands', control_qos
        )
        self.right_arm_pub = self.create_publisher(
            Float64MultiArray, '/right_arm_controller/commands', control_qos
        )
        self.base_cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', control_qos
        )

        # Subscribers for sensor data
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, sensor_qos
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, sensor_qos
        )
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, sensor_qos
        )

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # System timers
        self.balance_timer = self.create_timer(0.01, self.balance_control)  # 100Hz
        self.locomotion_timer = self.create_timer(0.02, self.locomotion_control)  # 50Hz
        self.perception_timer = self.create_timer(0.033, self.perception_processing)  # ~30Hz

        # Internal state
        self.joint_states = {}
        self.imu_data = None
        self.laser_data = None
        self.balance_ok = True
        self.current_pose = Pose()

        self.get_logger().info('Complete Humanoid System initialized')

    def joint_state_callback(self, msg):
        """Process joint state data"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_states[name] = {
                    'position': msg.position[i],
                    'velocity': msg.velocity[i] if i < len(msg.velocity) else 0.0,
                    'effort': msg.effort[i] if i < len(msg.effort) else 0.0
                }

    def imu_callback(self, msg):
        """Process IMU data for balance"""
        self.imu_data = {
            'orientation': (msg.orientation.x, msg.orientation.y,
                           msg.orientation.z, msg.orientation.w),
            'angular_velocity': (msg.angular_velocity.x, msg.angular_velocity.y,
                               msg.angular_velocity.z),
            'linear_acceleration': (msg.linear_acceleration.x, msg.linear_acceleration.y,
                                  msg.linear_acceleration.z)
        }

        # Update balance status
        self.update_balance_status()

    def laser_callback(self, msg):
        """Process laser scan data for navigation"""
        self.laser_data = {
            'ranges': np.array(msg.ranges),
            'intensities': np.array(msg.intensities),
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment
        }

    def update_balance_status(self):
        """Update robot balance status based on IMU data"""
        if self.imu_data:
            # Calculate tilt angles from orientation
            x, y, z, w = self.imu_data['orientation']

            # Calculate roll and pitch
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)

            sinp = 2 * (w * y - z * x)
            pitch = np.arcsin(sinp) if abs(sinp) < 1 else np.sign(sinp) * np.pi/2

            # Check if within balance limits
            max_tilt = np.radians(15)  # 15 degrees
            self.balance_ok = abs(roll) < max_tilt and abs(pitch) < max_tilt

    def balance_control(self):
        """Execute balance control"""
        if not self.balance_ok:
            self.get_logger().warn('Balance compromised!')
            # Emergency balance recovery
            self.emergency_balance_recovery()
        else:
            # Normal balance maintenance
            self.maintain_balance()

    def locomotion_control(self):
        """Execute locomotion control"""
        if self.balance_ok:
            # Execute planned movements
            self.execute_locomotion_plan()
        else:
            # Stop movement if balance is compromised
            self.stop_movement()

    def perception_processing(self):
        """Process perception data"""
        if self.laser_data is not None:
            # Process obstacle detection
            obstacles = self.detect_obstacles()
            if obstacles:
                # Adjust navigation plan based on obstacles
                self.adjust_navigation(obstacles)

    def detect_obstacles(self):
        """Detect obstacles from laser data"""
        if self.laser_data is None:
            return []

        # Find valid ranges (not infinite or NaN)
        valid_ranges = self.laser_data['ranges'][np.isfinite(self.laser_data['ranges'])]

        # Define threshold for obstacle detection
        obstacle_threshold = 1.0  # meter

        # Find obstacles
        obstacles = valid_ranges[valid_ranges < obstacle_threshold]

        return obstacles if len(obstacles) > 0 else []

    def emergency_balance_recovery(self):
        """Execute emergency balance recovery"""
        # Stop all movement
        self.stop_movement()

        # Adjust joint positions to recover balance
        # This would involve complex balance algorithms
        self.get_logger().info('Executing emergency balance recovery')

    def maintain_balance(self):
        """Maintain balance during normal operation"""
        # Adjust joint positions slightly to maintain balance
        pass

    def execute_locomotion_plan(self):
        """Execute planned locomotion"""
        # Send commands to leg controllers based on planned movement
        pass

    def stop_movement(self):
        """Stop all movement"""
        # Send zero commands to all controllers
        zero_cmd = Float64MultiArray()
        zero_cmd.data = [0.0] * 6  # Assuming 6 joints per leg
        self.left_leg_pub.publish(zero_cmd)
        self.right_leg_pub.publish(zero_cmd)

        stop_cmd = Twist()
        self.base_cmd_pub.publish(stop_cmd)

    def adjust_navigation(self, obstacles):
        """Adjust navigation based on obstacle detection"""
        self.get_logger().info(f'Detected {len(obstacles)} obstacles, adjusting navigation')

def main(args=None):
    rclpy.init(args=args)
    node = CompleteHumanoidSystem()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Complete Humanoid System')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Hands-On Activities

### Week 3 Activities

1. **ROS 2 Installation and Setup**
   - Install ROS 2 Humble Hawksbill
   - Set up workspace and environment variables
   - Verify installation with basic tutorials

2. **Node Creation Exercise**
   - Create a publisher node that publishes robot status
   - Create a subscriber node that processes the status
   - Test communication between nodes

3. **Service Implementation**
   - Create a service that accepts robot commands
   - Implement service server and client
   - Test synchronous communication

### Week 4 Activities

1. **Package Development**
   - Create a complete ROS 2 package for humanoid control
   - Implement multiple nodes with different functionalities
   - Create custom message types for humanoid joints

2. **Launch File Creation**
   - Create launch files for different robot configurations
   - Implement parameter management
   - Test launching multiple nodes simultaneously

3. **Parameter Management Exercise**
   - Create a node that manages robot configuration parameters
   - Implement dynamic parameter updates
   - Test parameter changes during runtime

### Week 5 Activities

1. **Humanoid Communication System**
   - Create a complete communication system for humanoid robot
   - Implement joint control interfaces
   - Integrate sensor data processing

2. **Performance Optimization**
   - Optimize node communications for real-time performance
   - Implement message throttling and filtering
   - Measure and improve system performance

3. **Integration Project**
   - Combine all learned concepts into a working humanoid system
   - Implement basic locomotion control
   - Test with simulated humanoid robot

## Assessment

### Week 3 Assessment
- **Lab Exercise**: Create basic ROS 2 publisher/subscriber system
- **Quiz**: ROS 2 architecture and core concepts
- **Discussion**: Compare different communication patterns

### Week 4 Assessment
- **Project**: Develop a complete ROS 2 package with custom messages
- **Demonstration**: Launch file functionality and parameter management
- **Code Review**: Quality and structure of ROS 2 code

### Week 5 Assessment
- **Integration Project**: Complete humanoid communication system
- **Performance Test**: Measure and optimize system performance
- **Presentation**: Demonstrate working humanoid control system

## Resources

### Required Reading
- "Programming Robots with ROS" by Morgan Quigley
- "Effective Robotics Programming with ROS" by Anil Mahtani
- ROS 2 Documentation: Core Concepts and Client Libraries

### Tutorials
- ROS 2 Beginner: Client Libraries
- ROS 2 Beginner: Launch Files
- ROS 2 Intermediate: Parameters
- ROS 2 Advanced: Composition

### Tools and Libraries
- rclpy: Python client library
- rviz2: Visualization tool
- rqt: GUI tools for ROS 2
- Gazebo: Simulation environment
- URDF: Robot description format

## Next Steps

After completing Weeks 3-5, you'll have mastered ROS 2 fundamentals and be ready to move on to Weeks 6-7: Robot Simulation with Gazebo. You'll apply your ROS 2 knowledge to create simulation environments for your humanoid robot, learning about physics simulation, sensor simulation, and robot description formats (URDF/SDF).

The ROS 2 foundation you've built will be essential for all subsequent modules, as it provides the communication backbone for all robot systems.