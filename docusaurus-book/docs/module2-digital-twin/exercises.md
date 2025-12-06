---
id: exercises
title: "Module 2 Practical Exercises"
sidebar_position: 6
---

# Module 2 Practical Exercises

## Overview

This section contains hands-on exercises to reinforce your understanding of digital twin simulation using Gazebo and Unity. These exercises will help you gain practical experience with physics simulation, sensor simulation, and creating realistic environments for humanoid robots.

## Exercise 1: Basic Gazebo Environment Setup

### Objective
Create a basic Gazebo environment with a humanoid robot model and test basic physics simulation.

### Instructions
1. Create a new ROS 2 package called `module2_exercises`
2. Set up a basic world file with ground plane and lighting
3. Add a simple humanoid robot model (you can use a basic stick figure model)
4. Configure physics parameters for realistic humanoid simulation
5. Launch the simulation and verify the robot falls due to gravity

### Required Components
- World file with physics configuration
- Basic humanoid robot URDF/SDF model
- Launch file to start the simulation
- Verification that the robot responds to gravity and collisions

### Expected Output
- Gazebo launches with a humanoid robot
- Robot falls and stabilizes on the ground
- Physics simulation runs at reasonable speed
- No errors in the console

### Code Template
```xml
<!-- world file: basic_humanoid.world -->
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="basic_humanoid">
    <!-- Physics -->
    <physics name="humanoid_physics" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
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
          </material>
        </visual>
      </link>
    </model>

    <!-- Include your humanoid robot -->
    <include>
      <uri>model://simple_humanoid</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### Evaluation Criteria
- World loads without errors
- Robot falls and stabilizes properly
- Physics parameters are appropriately set
- Simulation runs smoothly

## Exercise 2: Physics Tuning for Humanoid Stability

### Objective
Tune physics parameters to achieve stable humanoid simulation with realistic movement.

### Instructions
1. Create a more complex humanoid model with multiple joints
2. Implement different physics parameter sets for different scenarios:
   - Stable walking simulation
   - Fast movement simulation
   - High-precision manipulation simulation
3. Test each configuration and document the differences
4. Implement a simple walking controller to test stability

### Physics Parameter Sets
```xml
<!-- Stable Walking Configuration -->
<physics name="stable_walking" type="ode">
  <max_step_size>0.0005</max_step_size>
  <real_time_update_rate>2000.0</real_time_update_rate>
  <ode>
    <solver>
      <type>quick</type>
      <iters>100</iters>
    </solver>
    <constraints>
      <cfm>1e-6</cfm>
      <erp>0.05</erp>
    </constraints>
  </ode>
</physics>

<!-- Fast Simulation Configuration -->
<physics name="fast_sim" type="ode">
  <max_step_size>0.002</max_step_size>
  <real_time_update_rate>500.0</real_time_update_rate>
  <ode>
    <solver>
      <type>quick</type>
      <iters>20</iters>
    </solver>
    <constraints>
      <cfm>1e-4</cfm>
      <erp>0.2</erp>
    </constraints>
  </ode>
</physics>
```

### Walking Controller Template
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from builtin_interfaces.msg import Duration

class SimpleWalker(Node):
    def __init__(self):
        super().__init__('simple_walker')

        # Publishers for joint commands
        self.left_leg_pub = self.create_publisher(Float64MultiArray,
                                                '/left_leg_controller/commands', 10)
        self.right_leg_pub = self.create_publisher(Float64MultiArray,
                                                 '/right_leg_controller/commands', 10)

        # Timer for walking pattern
        self.timer = self.create_timer(0.1, self.step_callback)
        self.step_phase = 0.0

    def step_callback(self):
        # Generate simple walking pattern
        left_angles = [0.1 * (1 - abs(self.step_phase)), 0.2 * self.step_phase, -0.1 * self.step_phase]
        right_angles = [-0.1 * (1 - abs(self.step_phase)), -0.2 * self.step_phase, 0.1 * self.step_phase]

        # Publish commands
        left_msg = Float64MultiArray()
        left_msg.data = left_angles
        self.left_leg_pub.publish(left_msg)

        right_msg = Float64MultiArray()
        right_msg.data = right_angles
        self.right_leg_pub.publish(right_msg)

        # Update phase
        self.step_phase = (self.step_phase + 0.1) % (2 * 3.14159)

def main(args=None):
    rclpy.init(args=args)
    walker = SimpleWalker()
    rclpy.spin(walker)
    walker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Expected Output
- Different physics configurations produce different behaviors
- Stable walking pattern is achieved
- Documentation of parameter effects

## Exercise 3: Sensor Simulation and Integration

### Objective
Implement and test various sensor simulations in Gazebo and integrate with ROS 2.

### Instructions
1. Add multiple sensor types to your humanoid robot:
   - RGB camera
   - Depth camera
   - 2D LiDAR
   - IMU
   - Force/Torque sensors
2. Configure realistic noise models for each sensor
3. Create ROS 2 nodes to subscribe and process sensor data
4. Validate sensor outputs match expected ranges and characteristics

### Sensor Configuration Template
```xml
<!-- Camera sensor -->
<gazebo reference="head_camera_link">
  <sensor name="head_camera" type="camera">
    <update_rate>30.0</update_rate>
    <camera name="head">
      <horizontal_fov>1.089</horizontal_fov>
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
        <stddev>0.01</stddev>
      </noise>
    </camera>
    <always_on>true</always_on>
    <visualize>true</visualize>
  </sensor>
</gazebo>

<!-- LiDAR sensor -->
<gazebo reference="lidar_link">
  <sensor name="lidar" type="ray">
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.10</min>
        <max>10.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <always_on>true</always_on>
    <visualize>true</visualize>
  </sensor>
</gazebo>

<!-- IMU sensor -->
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
            <stddev>0.01</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.01</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.01</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

### Sensor Processing Node
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from cv_bridge import CvBridge
import numpy as np

class SensorValidator(Node):
    def __init__(self):
        super().__init__('sensor_validator')

        self.cv_bridge = CvBridge()

        # Subscribe to all sensor topics
        self.camera_sub = self.create_subscription(
            Image, '/head_camera/image_raw', self.camera_callback, 10
        )
        self.lidar_sub = self.create_subscription(
            LaserScan, '/lidar/scan', self.lidar_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu_sensor/data', self.imu_callback, 10
        )

        self.get_logger().info('Sensor validator initialized')

    def camera_callback(self, msg):
        """Validate camera data"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            height, width, channels = cv_image.shape

            # Validate image properties
            if width == 640 and height == 480:
                self.get_logger().info(f'Valid camera image: {width}x{height}')
            else:
                self.get_logger().warn(f'Unexpected image size: {width}x{height}')

        except Exception as e:
            self.get_logger().error(f'Camera callback error: {e}')

    def lidar_callback(self, msg):
        """Validate LiDAR data"""
        ranges = np.array(msg.ranges)
        valid_ranges = ranges[np.isfinite(ranges)]

        if len(valid_ranges) > 0:
            min_range = np.min(valid_ranges)
            max_range = np.max(valid_ranges)
            self.get_logger().info(f'LiDAR: min={min_range:.2f}, max={max_range:.2f}')

    def imu_callback(self, msg):
        """Validate IMU data"""
        orientation = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        angular_vel = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        linear_acc = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]

        self.get_logger().info(f'IMU orientation: {orientation}')

def main(args=None):
    rclpy.init(args=args)
    validator = SensorValidator()
    rclpy.spin(validator)
    validator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Expected Output
- All sensors publish data to ROS 2 topics
- Sensor validator node receives and processes data
- Data ranges match expected values
- Noise is present in sensor data

## Exercise 4: Unity High-Fidelity Environment

### Objective
Create a high-fidelity environment in Unity with realistic rendering and human-robot interaction.

### Instructions
1. Set up a Unity project with ROS-TCP-Connector
2. Create a humanoid robot model in Unity
3. Implement realistic lighting and materials
4. Add basic human-robot interaction scenario
5. Connect to ROS 2 for sensor simulation

### Unity Setup Steps
1. Create new Unity 3D project
2. Import ROS-TCP-Connector package
3. Add ROSConnection prefab to scene
4. Create humanoid robot with proper joint hierarchy
5. Add lighting and environment objects
6. Implement basic interaction script

### Interaction Script Template
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;

public class HumanRobotInteraction : MonoBehaviour
{
    public GameObject robot;
    public GameObject human;
    public float interactionDistance = 3.0f;
    public float personalSpace = 1.0f;

    private ROSConnection ros;
    private float lastInteractionTime = 0f;
    private bool isInteracting = false;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize("127.0.0.1", 10000);
    }

    void Update()
    {
        float distance = Vector3.Distance(robot.transform.position, human.transform.position);

        if (distance <= interactionDistance && !isInteracting)
        {
            StartInteraction();
        }
        else if (distance > interactionDistance && isInteracting)
        {
            EndInteraction();
        }

        if (isInteracting)
        {
            HandleInteraction();
        }
    }

    void StartInteraction()
    {
        isInteracting = true;
        lastInteractionTime = Time.time;

        // Publish interaction start message
        var interactionMsg = new std_msgs.msg.String();
        interactionMsg.data = "Interaction started";
        ros.Publish("/interaction_status", interactionMsg);
    }

    void HandleInteraction()
    {
        // Rotate robot to face human
        Vector3 direction = human.transform.position - robot.transform.position;
        direction.y = 0; // Keep rotation in XZ plane
        robot.transform.rotation = Quaternion.LookRotation(direction);
    }

    void EndInteraction()
    {
        isInteracting = false;

        // Publish interaction end message
        var interactionMsg = new std_msgs.msg.String();
        interactionMsg.data = "Interaction ended";
        ros.Publish("/interaction_status", interactionMsg);
    }
}
```

### Expected Output
- Unity scene with humanoid robot and human avatar
- Robot responds to human proximity
- Interaction messages published to ROS
- Realistic rendering and lighting

## Exercise 5: Complete Digital Twin Integration

### Objective
Integrate all concepts from Module 2 into a complete digital twin system.

### Instructions
1. Create a complete simulation environment that includes:
   - Realistic humanoid robot with multiple sensors
   - Physics simulation tuned for humanoid dynamics
   - Unity visualization for high-fidelity rendering
   - ROS 2 integration for all components
2. Implement a simple task (e.g., navigating to a goal while avoiding obstacles)
3. Test the system with both Gazebo and Unity components
4. Document performance and accuracy differences

### Launch File Template
```xml
<?xml version="1.0"?>
<launch>
  <!-- Gazebo simulation -->
  <include file="$(find-pkg-share gazebo_ros)/launch/gazebo.launch.py">
    <arg name="world" value="$(find-pkg-share module2_exercises)/worlds/digital_twin.world"/>
  </include>

  <!-- Robot state publisher -->
  <node pkg="robot_state_publisher" exec="robot_state_publisher" name="robot_state_publisher">
    <param name="robot_description" value="$(find-pkg-share module2_exercises)/urdf/humanoid.urdf"/>
  </node>

  <!-- Sensor processing nodes -->
  <node pkg="module2_exercises" exec="sensor_validator" name="sensor_validator"/>

  <!-- Navigation nodes -->
  <node pkg="module2_exercises" exec="simple_navigator" name="simple_navigator"/>
</launch>
```

### Navigation Node Template
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np

class SimpleNavigator(Node):
    def __init__(self):
        super().__init__('simple_navigator')

        # Publishers and subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/lidar/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Navigation parameters
        self.target_x = 5.0
        self.target_y = 5.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.safe_distance = 0.5

        # Timer for navigation loop
        self.timer = self.create_timer(0.1, self.navigation_loop)

    def odom_callback(self, msg):
        """Update current position"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def scan_callback(self, msg):
        """Process LiDAR data for obstacle avoidance"""
        self.ranges = np.array(msg.ranges)

    def navigation_loop(self):
        """Main navigation logic"""
        cmd = Twist()

        # Calculate distance to target
        dist_to_target = np.sqrt(
            (self.target_x - self.current_x)**2 +
            (self.target_y - self.current_y)**2
        )

        if dist_to_target < 0.5:  # Reached target
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            # Check for obstacles
            if hasattr(self, 'ranges'):
                min_range = np.min(self.ranges[np.isfinite(self.ranges)])

                if min_range < self.safe_distance:
                    # Obstacle detected - turn away
                    cmd.angular.z = 0.5
                    cmd.linear.x = 0.0
                else:
                    # Move toward target
                    cmd.linear.x = 0.5
                    cmd.angular.z = 0.1  # Small turn toward target

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    navigator = SimpleNavigator()
    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Expected Output
- Complete integrated system running
- Robot navigates to goal while avoiding obstacles
- All sensors functioning properly
- Unity and Gazebo components working together

## Assessment Questions

1. How do physics parameters affect humanoid robot simulation stability?

2. What are the key differences between Gazebo and Unity for robotics simulation?

3. How do you configure realistic sensor noise models in Gazebo?

4. What are the advantages of using digital twins in Physical AI development?

5. How would you validate that your simulation accurately represents real-world conditions?

## Submission Requirements

For each exercise, submit:
- Configuration files (URDF, SDF, world files)
- Source code for any custom nodes or scripts
- Launch files
- Screenshots of successful execution
- A brief report documenting your implementation and findings

## Evaluation Rubric

- **Functionality** (40%): Systems work as expected and meet requirements
- **Realism** (25%): Simulation parameters match realistic values
- **Integration** (20%): Components work together seamlessly
- **Documentation** (15%): Clear explanations and proper documentation

Complete all exercises to gain comprehensive experience with digital twin simulation for Physical AI and humanoid robotics applications.