---
id: nodes-topics-services
title: "ROS 2 Nodes, Topics, and Services"
sidebar_position: 2
---

# ROS 2 Nodes, Topics, and Services

## Introduction

In this section, we'll explore the fundamental communication mechanisms in ROS 2: Nodes, Topics, and Services. These components form the backbone of robot communication, enabling different parts of a robot to exchange information and coordinate their actions.

Think of a humanoid robot as a human body: nodes are like organs (brain, heart, muscles), topics are like the nervous system carrying sensory information (touch, sight, sound), and services are like specific requests (like raising your hand when asked).

## Nodes

### What is a Node?

A node is an executable that uses ROS 2 client library to communicate with other nodes. Nodes are the basic computational elements of a ROS 2 program. Each node runs a specific task and communicates with other nodes through topics, services, or actions.

In a humanoid robot, you might have nodes for:
- Joint controllers
- Sensor data processing
- Path planning
- Vision processing
- Speech recognition

### Creating a Node

Here's a basic example of a ROS 2 node in Python:

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

## Topics

### What is a Topic?

Topics are ROS 2's asynchronous, many-to-many communication mechanism. They allow nodes to publish messages that other nodes can subscribe to. This is a fire-and-forget system where publishers don't know who is listening, and subscribers don't know who is publishing.

Topics are ideal for:
- Sensor data streams (camera images, LiDAR scans)
- Robot state information (joint positions, battery levels)
- Continuous data that multiple nodes might need

### Publishers and Subscribers

In ROS 2, communication happens through a publish-subscribe model:

- **Publisher**: A node that sends messages to a topic
- **Subscriber**: A node that receives messages from a topic

Here's an example of a subscriber:

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

## Services

### What is a Service?

Services provide synchronous, request-response communication between nodes. Unlike topics, services are one-to-one and block until a response is received. This makes them ideal for operations that require a guaranteed response.

Services are ideal for:
- Configuration changes
- Calibration procedures
- Error reporting
- Any operation where you need to confirm success/failure

### Service Structure

A service consists of:
- **Request**: The data sent to the service
- **Response**: The data returned by the service
- **Service Definition**: Defines the format of request and response

Here's an example service definition (in `.srv` file format):
```
string name
int32 age
---
bool success
string message
```

And here's a service server implementation:

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

## Communication Patterns in Humanoid Robots

### Sensor Data Pipeline

In a humanoid robot, sensors typically publish data to topics:

```
Camera Node → /camera/image_raw (sensor_msgs/Image)
IMU Node → /imu/data (sensor_msgs/Imu)
Lidar Node → /scan (sensor_msgs/LaserScan)
```

Multiple nodes can subscribe to these topics for different purposes:
- Perception node processes camera images
- Balance controller uses IMU data
- Navigation system uses LiDAR data

### Control Commands

Control commands often use services or actions:
- Service: "Move head to position X,Y,Z"
- Action: "Walk to destination" (with feedback during execution)

## Best Practices

1. **Topic Naming**: Use descriptive names that indicate the data type and source (e.g., `/robot1/joint_states`)

2. **QoS Settings**: Configure Quality of Service settings appropriately for your data type:
   - Sensors: Use reliable delivery for critical data
   - Images: Use best-effort for high-frequency data

3. **Message Types**: Use standard message types when possible to ensure compatibility with existing tools

4. **Node Organization**: Design nodes with single responsibilities to improve maintainability

## Hands-on Exercise

Create a simple ROS 2 package with:
1. A publisher node that publishes a custom message with robot status
2. A subscriber node that receives and displays the status
3. A service server that responds to status queries

This will give you hands-on experience with the core communication mechanisms in ROS 2.