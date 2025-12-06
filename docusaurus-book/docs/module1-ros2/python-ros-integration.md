---
id: python-ros-integration
title: "Python Agents to ROS Controllers Integration"
sidebar_position: 3
---

# Python Agents to ROS Controllers Integration

## Introduction

One of the most powerful aspects of ROS 2 is its ability to bridge different programming languages and systems. In this section, we'll explore how to connect Python-based AI agents to ROS controllers using the `rclpy` library. This integration is crucial for Physical AI applications where Python-based machine learning models need to control physical robots.

The integration allows you to:
- Connect Python AI/ML models to robot hardware
- Create sophisticated control systems using Python's rich ecosystem
- Implement cognitive planning and decision-making in Python
- Bridge high-level AI reasoning with low-level robot control

## Understanding rclpy

`rclpy` is the Python client library for ROS 2. It provides Python bindings for the ROS 2 client library (rcl) and allows Python programs to interact with ROS 2 in the same way that C++ programs do with `rclcpp`.

### Key Features of rclpy:
- Node creation and management
- Publisher and subscriber functionality
- Service and action clients/servers
- Parameter management
- Time and duration utilities

## Basic Integration Pattern

The most common pattern for integrating Python agents with ROS controllers involves:

1. **Python Agent Node**: Runs the AI/ML logic
2. **ROS Controllers**: Handle low-level robot control
3. **Message Passing**: Communication via topics, services, or actions

Here's a basic example of a Python agent that controls a robot:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np

class RobotAgent(Node):
    def __init__(self):
        super().__init__('robot_agent')

        # Publisher for robot commands
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber for sensor data
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        # Agent state
        self.laser_data = None
        self.get_logger().info('Robot Agent initialized')

    def scan_callback(self, msg):
        """Process laser scan data"""
        self.laser_data = np.array(msg.ranges)
        # Filter out invalid readings
        self.laser_data = self.laser_data[np.isfinite(self.laser_data)]

    def control_loop(self):
        """Main control loop for the AI agent"""
        if self.laser_data is not None:
            # Simple obstacle avoidance algorithm
            cmd = self.simple_navigation()
            self.cmd_vel_publisher.publish(cmd)

    def simple_navigation(self):
        """Simple AI navigation algorithm"""
        msg = Twist()

        # Check for obstacles in front
        if len(self.laser_data) > 0:
            front_scan = self.laser_data[len(self.laser_data)//2 - 50 : len(self.laser_data)//2 + 50]
            min_distance = np.min(front_scan) if len(front_scan) > 0 else float('inf')

            if min_distance < 1.0:  # Obstacle too close
                msg.linear.x = 0.0
                msg.angular.z = 0.5  # Turn right
            else:
                msg.linear.x = 0.5  # Move forward
                msg.angular.z = 0.0

        return msg

def main(args=None):
    rclpy.init(args=args)
    agent = RobotAgent()

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Integration Patterns

### 1. Behavior Trees Integration

Behavior trees are a popular method for organizing complex robot behaviors:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus
import py_trees

class BehaviorTreeAgent(Node):
    def __init__(self):
        super().__init__('behavior_tree_agent')

        # Create behavior tree
        self.behaviour_tree = self.setup_behavior_tree()

        # Timer for behavior tree execution
        self.timer = self.create_timer(0.1, self.tick_tree)

    def setup_behavior_tree(self):
        """Setup the behavior tree structure"""
        root = py_trees.composites.Sequence(name="Navigation Sequence")

        # Add behaviors to the tree
        check_battery = py_trees.behaviours.CheckBattery(name="Check Battery")
        navigate_to_goal = py_trees.actions.ActionClient(
            name="Navigate to Goal",
            action_type="nav2_msgs/action/NavigateToPose",
            action_name="navigate_to_pose"
        )

        root.add_children([check_battery, navigate_to_goal])
        return py_trees.trees.BehaviourTree(root)

    def tick_tree(self):
        """Execute one tick of the behavior tree"""
        self.behaviour_tree.tick_once()
```

### 2. Machine Learning Model Integration

Integrating trained ML models with ROS controllers:

```python
import rclpy
from rclpy.node import Node
import torch
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class MLControlAgent(Node):
    def __init__(self):
        super().__init__('ml_control_agent')

        # Load pre-trained model
        self.model = self.load_model()
        self.cv_bridge = CvBridge()

        # Set up ROS interfaces
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def load_model(self):
        """Load a pre-trained PyTorch model"""
        # Example: Load a trained navigation model
        model = torch.load('navigation_model.pth')
        model.eval()
        return model

    def image_callback(self, msg):
        """Process camera image and generate control command"""
        # Convert ROS image to OpenCV format
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Preprocess image for the model
        input_tensor = self.preprocess_image(cv_image)

        # Get model prediction
        with torch.no_grad():
            control_output = self.model(input_tensor)

        # Convert prediction to ROS command
        cmd = self.convert_to_twist(control_output)
        self.cmd_pub.publish(cmd)

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Resize, normalize, convert to tensor
        processed = cv2.resize(image, (224, 224))
        processed = processed.astype(np.float32) / 255.0
        processed = np.transpose(processed, (2, 0, 1))
        return torch.from_numpy(processed).unsqueeze(0)

    def convert_to_twist(self, control_output):
        """Convert model output to Twist message"""
        cmd = Twist()
        cmd.linear.x = float(control_output[0])
        cmd.angular.z = float(control_output[1])
        return cmd
```

## Integration with Humanoid Robot Controllers

For humanoid robots specifically, the integration often involves:

### Joint Trajectory Control
```python
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class HumanoidAgent(Node):
    def __init__(self):
        super().__init__('humanoid_agent')

        # Publisher for joint trajectories
        self.joint_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

    def move_arm(self, joint_positions):
        """Send joint positions to move the humanoid's arm"""
        msg = JointTrajectory()
        msg.joint_names = ['shoulder_joint', 'elbow_joint', 'wrist_joint']

        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start = Duration(sec=2)  # 2 seconds to reach position

        msg.points = [point]
        self.joint_pub.publish(msg)
```

## Best Practices for Python-ROS Integration

### 1. Error Handling
Always implement robust error handling for network communication:

```python
def safe_publish(self, publisher, msg):
    """Safely publish a message with error handling"""
    try:
        publisher.publish(msg)
    except Exception as e:
        self.get_logger().error(f'Failed to publish message: {e}')
```

### 2. Threading Considerations
Be aware of threading issues when using Python agents with ROS:

```python
import threading

class ThreadedAgent(Node):
    def __init__(self):
        super().__init__('threaded_agent')

        # Lock for thread-safe operations
        self.data_lock = threading.Lock()
        self.sensor_data = None

    def sensor_callback(self, msg):
        with self.data_lock:
            self.sensor_data = msg
```

### 3. Performance Optimization
For real-time applications, optimize for performance:

- Use efficient data structures
- Minimize message copying
- Consider using Fast DDS for low-latency communication
- Profile your Python code for bottlenecks

## Troubleshooting Common Issues

### Message Synchronization
Ensure proper synchronization between AI processing and control:

```python
from rclpy.time import Time

class SynchronizedAgent(Node):
    def __init__(self):
        super().__init__('sync_agent')
        self.last_sensor_time = Time()

    def sensor_callback(self, msg):
        self.last_sensor_time = self.get_clock().now()
        # Process sensor data...
```

### Memory Management
Monitor memory usage with Python agents:

```python
import psutil
import gc

def check_memory(self):
    """Monitor and manage memory usage"""
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > 80:
        gc.collect()  # Force garbage collection
```

## Hands-on Exercise

Create a Python agent that:
1. Subscribes to sensor data (e.g., laser scan)
2. Implements a simple decision-making algorithm
3. Publishes control commands to move a robot
4. Includes proper error handling and logging

This exercise will solidify your understanding of connecting Python-based logic to ROS-based robot control systems.