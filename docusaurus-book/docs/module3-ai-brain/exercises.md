---
id: exercises
title: "Module 3 Practical Exercises"
sidebar_position: 5
---

import BilingualChapter from '@site/src/components/BilingualChapter';

<BilingualChapter>
  <div className="english">
    # Module 3 Practical Exercises

    ## Overview

    This section contains hands-on exercises to reinforce your understanding of NVIDIA Isaac, including Isaac Sim for photorealistic simulation, Isaac ROS for hardware-accelerated perception and navigation, and Nav2 for path planning in humanoid robots. These exercises will help you gain practical experience with the AI-Robot Brain components essential for Physical AI applications.

    ## Exercise 1: Isaac Sim Installation and Basic Setup

    ### Objective
    Install and configure NVIDIA Isaac Sim, then create a basic simulation environment with a humanoid robot.

    ### Instructions
    1. Install Isaac Sim following the hardware requirements (RTX 4070 Ti or higher)
    2. Launch Isaac Sim and familiarize yourself with the interface
    3. Create a simple environment with a ground plane and basic lighting
    4. Add a humanoid robot model to the scene
    5. Configure the robot with appropriate physics properties
    6. Verify that the robot responds correctly to gravity and collisions

    ### Required Components
    - Isaac Sim installation with proper hardware
    - Basic scene with ground plane and lighting
    - Humanoid robot model with physics configuration
    - Verification of physics simulation

    ### Expected Output
    - Isaac Sim launches without errors
    - Robot falls and stabilizes on the ground
    - Physics simulation runs smoothly
    - Basic scene is properly configured

    ### Configuration Template
    ```python
    # basic_isaac_sim_setup.py
    import omni
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.prims import create_prim
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    import numpy as np

    def setup_basic_environment():
        """Set up a basic Isaac Sim environment"""
        # Initialize the world
        world = World(stage_units_in_meters=1.0)

        # Create ground plane
        create_prim(
            prim_path="/World/ground_plane",
            prim_type="Plane",
            position=np.array([0, 0, 0]),
            scale=np.array([10, 10, 1])
        )

        # Add lighting
        create_prim(
            prim_path="/World/Room/Light",
            prim_type="DistantLight",
            position=np.array([0, 0, 10]),
            attributes={"color": np.array([0.8, 0.8, 0.8]), "intensity": 3000}
        )

        # Add a simple humanoid robot (using a basic model as example)
        assets_root_path = get_assets_root_path()
        if assets_root_path is not None:
            # Add a simple robot model (replace with humanoid model)
            add_reference_to_stage(
                usd_path=assets_root_path + "/Isaac/Robots/TurtleBot/turtlebot3_differential.usd",
                prim_path="/World/Robot"
            )

        # Reset the world to apply changes
        world.reset()

        return world

    def main():
        """Main function to run the basic setup"""
        world = setup_basic_environment()

        # Run the simulation for a few steps to verify setup
        for i in range(100):
            world.step(render=True)

        print("Basic Isaac Sim environment created successfully!")

    if __name__ == "__main__":
        main()
    ```

    ### Evaluation Criteria
    - Isaac Sim installed and running
    - Basic environment created successfully
    - Robot model added and responding to physics
    - No errors during simulation

    ## Exercise 2: Isaac ROS Visual SLAM Implementation

    ### Objective
    Implement and test Isaac ROS Visual SLAM with a humanoid robot in simulation.

    ### Instructions
    1. Install Isaac ROS packages on your system
    2. Configure a stereo camera setup for the humanoid robot
    3. Set up Isaac ROS Visual SLAM node with proper parameters
    4. Test SLAM performance in a simulated environment
    5. Evaluate the quality of the generated map and localization

    ### Required Components
    - Isaac ROS Visual SLAM packages installed
    - Stereo camera configuration for the robot
    - Proper parameter configuration
    - SLAM testing and evaluation

    ### Configuration Template
    ```yaml
    # visual_slam_config.yaml
    /**:
      ros__parameters:
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
    ```

    ### ROS Node Template
    ```python
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, Imu
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped
    from tf2_ros import TransformBroadcaster
    import numpy as np

    class IsaacSLAMTestNode(Node):
        def __init__(self):
            super().__init__('isaac_slam_test')

            # Publishers and subscribers
            self.odom_pub = self.create_publisher(Odometry, '/visual_slam/odometry', 10)
            self.pose_pub = self.create_publisher(PoseStamped, '/visual_slam/pose', 10)

            # TF broadcaster
            self.tf_broadcaster = TransformBroadcaster(self)

            # SLAM state
            self.position = np.array([0.0, 0.0, 0.0])
            self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion
            self.last_update_time = self.get_clock().now()

            # Timer for periodic updates
            self.timer = self.create_timer(0.1, self.publish_slam_data)

        def publish_slam_data(self):
            """Simulate SLAM data publishing"""
            current_time = self.get_clock().now()

            # Simulate robot movement (in real scenario, this comes from SLAM)
            dt = (current_time.nanoseconds - self.last_update_time.nanoseconds) / 1e9
            self.position[0] += 0.1 * dt  # Move forward at 0.1 m/s
            self.position[1] += 0.05 * dt  # Slight lateral movement

            # Create and publish odometry message
            odom_msg = Odometry()
            odom_msg.header.stamp = current_time.to_msg()
            odom_msg.header.frame_id = 'map'
            odom_msg.child_frame_id = 'base_link'

            odom_msg.pose.pose.position.x = float(self.position[0])
            odom_msg.pose.pose.position.y = float(self.position[1])
            odom_msg.pose.pose.position.z = float(self.position[2])
            odom_msg.pose.pose.orientation.x = float(self.orientation[0])
            odom_msg.pose.pose.orientation.y = float(self.orientation[1])
            odom_msg.pose.pose.orientation.z = float(self.orientation[2])
            odom_msg.pose.pose.orientation.w = float(self.orientation[3])

            self.odom_pub.publish(odom_msg)

            # Publish TF transform
            self.broadcast_transform(current_time)

            self.last_update_time = current_time

        def broadcast_transform(self, timestamp):
            """Broadcast transform from map to base_link"""
            t = TransformStamped()
            t.header.stamp = timestamp.to_msg()
            t.header.frame_id = 'map'
            t.child_frame_id = 'base_link'

            t.transform.translation.x = float(self.position[0])
            t.transform.translation.y = float(self.position[1])
            t.transform.translation.z = float(self.position[2])
            t.transform.rotation.x = float(self.orientation[0])
            t.transform.rotation.y = float(self.orientation[1])
            t.transform.rotation.z = float(self.orientation[2])
            t.transform.rotation.w = float(self.orientation[3])

            self.tf_broadcaster.sendTransform(t)

    def main(args=None):
        rclpy.init(args=args)
        node = IsaacSLAMTestNode()

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

    ### Expected Output
    - Isaac ROS Visual SLAM node running
    - Odometry and pose data published
    - TF transforms broadcasting correctly
    - SLAM performance metrics recorded

    ## Exercise 3: Isaac ROS Stereo DNN Object Detection

    ### Objective
    Implement object detection using Isaac ROS Stereo DNN and integrate with humanoid navigation.

    ### Instructions
    1. Configure Isaac ROS Stereo DNN with appropriate neural network
    2. Set up camera topics and parameters for object detection
    3. Process detection results and filter for humanoid-relevant objects
    4. Integrate detection results with navigation system
    5. Test object detection performance in various scenarios

    ### Configuration Template
    ```yaml
    # stereo_dnn_config.yaml
    /**:
      ros__parameters:
        # Input topics
        left_image_topic: "/camera/left/image_rect_color"
        right_image_topic: "/camera/right/image_rect_color"
        left_camera_info_topic: "/camera/left/camera_info"
        right_camera_info_topic: "/camera/right/camera_info"

        # Neural network parameters
        model_type: "detectnet"
        model_name: "resnet18_detector"
        confidence_threshold: 0.5
        max_objects: 100

        # Performance parameters
        input_width: 960
        input_height: 544
        batch_size: 1
    ```

    ### Object Detection Node
    ```python
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from vision_msgs.msg import Detection2DArray
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker, MarkerArray
    from std_msgs.msg import ColorRGBA
    import numpy as np

    class IsaacStereoDNNNode(Node):
        def __init__(self):
            super().__init__('isaac_stereo_dnn')

            # Publisher for filtered detections
            self.detection_pub = self.create_publisher(
                Detection2DArray,
                '/humanoid/detections',
                10
            )

            # Publisher for visualization markers
            self.marker_pub = self.create_publisher(
                MarkerArray,
                '/detection_markers',
                10
            )

            # Objects of interest for humanoid navigation
            self.target_objects = [
                'person', 'chair', 'table', 'door', 'bottle',
                'cup', 'bowl', 'couch', 'potted plant'
            ]

            self.get_logger().info('Isaac Stereo DNN node initialized')

        def process_detections(self, detections_msg):
            """Process incoming detections and filter for humanoid-relevant objects"""
            filtered_detections = Detection2DArray()
            filtered_detections.header = detections_msg.header

            markers = MarkerArray()

            for i, detection in enumerate(detections_msg.detections):
                # Check if detection is of interest
                if detection.results[0].hypothesis.class_id in self.target_objects:
                    # Add to filtered list
                    filtered_detections.detections.append(detection)

                    # Create visualization marker
                    marker = Marker()
                    marker.header = detections_msg.header
                    marker.ns = "detections"
                    marker.id = i
                    marker.type = Marker.TEXT_VIEW_FACING
                    marker.action = Marker.ADD

                    # Position marker at center of bounding box
                    marker.pose.position.x = detection.bbox.center.x
                    marker.pose.position.y = detection.bbox.center.y
                    marker.pose.position.z = 1.0  # Above the objects

                    marker.text = detection.results[0].hypothesis.class_id
                    marker.scale.z = 0.2  # Text scale
                    marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)

                    markers.markers.append(marker)

            # Publish results
            if len(filtered_detections.detections) > 0:
                self.detection_pub.publish(filtered_detections)

            if len(markers.markers) > 0:
                self.marker_pub.publish(markers)

    def main(args=None):
        rclpy.init(args=args)
        node = IsaacStereoDNNNode()

        # In a real implementation, you would subscribe to Isaac ROS detection topics
        # For this exercise, we'll simulate the detection processing

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

    ### Expected Output
    - Isaac ROS Stereo DNN processing images
    - Detection results published to humanoid-specific topic
    - Visualization markers showing detected objects
    - Performance metrics recorded

    ## Exercise 4: Nav2 Path Planning for Humanoid Robots

    ### Objective
    Configure and test Nav2 for humanoid robot navigation with specialized parameters.

    ### Instructions
    1. Install and configure Nav2 with humanoid-specific parameters
    2. Set up costmaps with appropriate inflation and robot radius
    3. Configure local and global planners for humanoid constraints
    4. Test navigation in simulation with various obstacles
    5. Evaluate path quality and navigation performance

    ### Nav2 Configuration Template
    ```yaml
    # humanoid_nav2_config.yaml
    amcl:
      ros__parameters:
        use_sim_time: True
        alpha1: 0.2
        alpha2: 0.2
        alpha3: 0.2
        alpha4: 0.2
        alpha5: 0.2
        base_frame_id: "base_footprint"
        beam_skip_distance: 0.5
        beam_skip_error_threshold: 0.9
        beam_skip_threshold: 0.3
        do_beamskip: false
        global_frame_id: "map"
        lambda_short: 0.1
        laser_likelihood_max_dist: 2.0
        laser_max_range: 10.0
        laser_min_range: -1.0
        laser_model_type: "likelihood_field"
        max_beams: 60
        max_particles: 2000
        min_particles: 500
        odom_frame_id: "odom"
        pf_err: 0.05
        pf_z: 0.99
        recovery_alpha_fast: 0.0
        recovery_alpha_slow: 0.0
        resample_interval: 1
        robot_model_type: "nav2_amcl::DifferentialMotionModel"
        save_pose_rate: 0.5
        sigma_hit: 0.2
        tf_broadcast: true
        transform_tolerance: 1.0
        update_min_a: 0.2
        update_min_d: 0.2
        z_hit: 0.5
        z_max: 0.05
        z_rand: 0.5
        z_short: 0.05

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
          xy_goal_tolerance: 0.3
          yaw_goal_tolerance: 0.3
          stateful: True
          progress_checker: "progress_checker"
          goal_checker: "goal_checker"
          costmap_converter_plugin: "costmap_converter"
          costmap_converter_spin_thread: True
          costmap_converter_frequency: 5
          # Humanoid-specific parameters
          step_size: 0.3  # Maximum step size for bipedal locomotion
          balance_constraint: 0.8  # Balance stability factor

    local_costmap:
      local_costmap:
        ros__parameters:
          update_frequency: 5.0
          publish_frequency: 2.0
          global_frame: "odom"
          robot_base_frame: "base_footprint"
          use_sim_time: True
          rolling_window: true
          width: 6
          height: 6
          resolution: 0.05
          # Humanoid-specific parameters
          robot_radius: 0.4  # Larger radius for humanoid safety
          inflation_radius: 0.6
          cost_scaling_factor: 5.0

    global_costmap:
      global_costmap:
        ros__parameters:
          update_frequency: 1.0
          publish_frequency: 1.0
          global_frame: "map"
          robot_base_frame: "base_footprint"
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
    ```

    ### Navigation Test Node
    ```python
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionClient
    import time

    class HumanoidNavigationTestNode(Node):
        def __init__(self):
            super().__init__('humanoid_navigation_test')

            # Action client for Nav2
            self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

            # Timer to send navigation goals
            self.timer = self.create_timer(10.0, self.send_navigation_goal)
            self.goal_count = 0

            # Define test waypoints
            self.waypoints = [
                (2.0, 2.0, 0.0),    # x, y, theta
                (4.0, 1.0, 1.57),   # Turn around
                (3.0, 4.0, 3.14),   # Another position
                (1.0, 3.0, -1.57)   # Return near start
            ]

        def send_navigation_goal(self):
            """Send navigation goal to Nav2"""
            if not self.nav_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error('Nav2 server not available')
                return

            # Get next waypoint
            if self.goal_count >= len(self.waypoints):
                self.get_logger().info('Completed all navigation goals')
                self.timer.cancel()
                return

            x, y, theta = self.waypoints[self.goal_count]

            goal_msg = NavigateToPose.Goal()
            goal_msg.pose.header.frame_id = 'map'
            goal_msg.pose.pose.position.x = float(x)
            goal_msg.pose.pose.position.y = float(y)
            goal_msg.pose.pose.position.z = 0.0

            # Convert theta to quaternion
            import math
            goal_msg.pose.pose.orientation.z = math.sin(theta / 2)
            goal_msg.pose.pose.orientation.w = math.cos(theta / 2)

            self.get_logger().info(f'Sending navigation goal {self.goal_count + 1}: ({x}, {y}, {theta})')

            future = self.nav_client.send_goal_async(goal_msg)
            future.add_done_callback(self.navigation_result_callback)

            self.goal_count += 1

        def navigation_result_callback(self, future):
            """Handle navigation result"""
            try:
                goal_handle = future.result()
                if goal_handle.accepted:
                    self.get_logger().info('Navigation goal accepted')
                    # In a real implementation, you'd wait for result completion
                else:
                    self.get_logger().error('Navigation goal rejected')
            except Exception as e:
                self.get_logger().error(f'Navigation failed: {e}')

    def main(args=None):
        rclpy.init(args=args)
        node = HumanoidNavigationTestNode()

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

    ### Expected Output
    - Nav2 configured with humanoid-specific parameters
    - Navigation goals sent and processed
    - Path planning and execution working
    - Performance metrics recorded

    ## Exercise 5: Isaac ROS and Nav2 Integration

    ### Objective
    Integrate Isaac ROS perception with Nav2 navigation for complete AI-Robot Brain functionality.

    ### Instructions
    1. Set up Isaac ROS Visual SLAM for localization
    2. Configure Isaac ROS Stereo DNN for object detection
    3. Integrate perception data with Nav2 costmaps
    4. Implement dynamic obstacle avoidance
    5. Test complete system in simulation

    ### Integration Node Template
    ```python
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped, Twist
    from sensor_msgs.msg import LaserScan, Imu
    from vision_msgs.msg import Detection2DArray
    from tf2_ros import TransformListener, Buffer
    from geometry_msgs.msg import TransformStamped
    from std_msgs.msg import Bool
    import numpy as np

    class IsaacROSNav2Integrator(Node):
        def __init__(self):
            super().__init__('isaac_ros_nav2_integrator')

            # Subscribers for Isaac ROS data
            self.pose_sub = self.create_subscription(
                PoseStamped,
                '/visual_slam/pose',
                self.pose_callback,
                10
            )
            self.detection_sub = self.create_subscription(
                Detection2DArray,
                '/humanoid/detections',
                self.detection_callback,
                10
            )
            self.imu_sub = self.create_subscription(
                Imu,
                '/imu/data',
                self.imu_callback,
                10
            )

            # Publishers for navigation commands
            self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
            self.emergency_stop_pub = self.create_publisher(Bool, '/emergency_stop', 10)

            # TF listener
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

            # Robot state
            self.current_pose = None
            self.detections = []
            self.balance_ok = True
            self.obstacle_detected = False
            self.navigation_enabled = True

            # Safety parameters
            self.balance_threshold = 0.3  # radians
            self.obstacle_distance_threshold = 0.8  # meters

            # Timer for safety checks
            self.safety_timer = self.create_timer(0.1, self.safety_check)

        def pose_callback(self, msg):
            """Update current pose from Isaac ROS"""
            self.current_pose = msg

        def detection_callback(self, msg):
            """Process object detections"""
            self.detections = msg.detections

            # Check for obstacles in path
            self.check_obstacle_detections()

        def imu_callback(self, msg):
            """Check balance from IMU data"""
            # Extract roll and pitch from orientation
            import math
            quat = msg.orientation
            sinr_cosp = 2 * (quat.w * quat.x + quat.y * quat.z)
            cosr_cosp = 1 - 2 * (quat.x * quat.x + quat.y * quat.y)
            roll = math.atan2(sinr_cosp, cosr_cosp)

            sinp = 2 * (quat.w * quat.y - quat.z * quat.x)
            pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi/2, sinp)

            self.balance_ok = (abs(roll) < self.balance_threshold and
                              abs(pitch) < self.balance_threshold)

        def check_obstacle_detections(self):
            """Check if detections represent obstacles in navigation path"""
            # In a real implementation, you'd transform detection coordinates
            # to robot frame and check if they're in the navigation path
            self.obstacle_detected = len(self.detections) > 0

        def safety_check(self):
            """Perform safety checks and emergency stops if needed"""
            emergency_stop = False
            reason = ""

            if not self.balance_ok:
                emergency_stop = True
                reason = "Balance threshold exceeded"
            elif self.obstacle_detected:
                emergency_stop = True
                reason = "Obstacle detected in path"

            if emergency_stop and self.navigation_enabled:
                self.get_logger().warn(f'Emergency stop triggered: {reason}')

                # Stop robot
                stop_msg = Twist()
                self.cmd_vel_pub.publish(stop_msg)

                # Publish emergency stop signal
                emergency_msg = Bool()
                emergency_msg.data = True
                self.emergency_stop_pub.publish(emergency_msg)

                self.navigation_enabled = False

        def enable_navigation(self):
            """Re-enable navigation after safety conditions are resolved"""
            if self.balance_ok and not self.obstacle_detected:
                self.navigation_enabled = True
                self.get_logger().info('Navigation re-enabled')

    def main(args=None):
        rclpy.init(args=args)
        node = IsaacROSNav2Integrator()

        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            # Stop robot before shutting down
            stop_msg = Twist()
            node.cmd_vel_pub.publish(stop_msg)
            node.get_logger().info('Stopped robot for shutdown')
        finally:
            node.destroy_node()
            rclpy.shutdown()

    if __name__ == '__main__':
        main()
    ```

    ### Launch File Template
    ```xml
    <?xml version="1.0"?>
    <launch>
      <!-- Isaac ROS Visual SLAM -->
      <node pkg="isaac_ros_visual_slam" exec="visual_slam_node" name="visual_slam" output="screen">
        <param from="$(find-pkg-share my_robot_config)/config/visual_slam_config.yaml"/>
      </node>

      <!-- Isaac ROS Stereo DNN -->
      <node pkg="isaac_ros_stereo_dnn" exec="detectnet_node" name="detectnet" output="screen">
        <param from="$(find-pkg-share my_robot_config)/config/stereo_dnn_config.yaml"/>
      </node>

      <!-- Nav2 Stack -->
      <include file="$(find-pkg-share nav2_bringup)/launch/navigation_launch.py">
        <arg name="params_file" value="$(find-pkg-share my_robot_config)/config/humanoid_nav2_config.yaml"/>
      </include>

      <!-- Integration Node -->
      <node pkg="my_robot_perception" exec="isaac_ros_nav2_integrator" name="isaac_ros_nav2_integrator" output="screen"/>

      <!-- RViz for visualization -->
      <node pkg="rviz2" exec="rviz2" name="rviz2" args="-d $(find-pkg-share my_robot_config)/rviz/integration.rviz"/>
    </launch>
    ```

    ### Expected Output
    - Complete integration of Isaac ROS and Nav2
    - Real-time perception and navigation
    - Safety checks and emergency responses
    - Coordinated system behavior

    ## Assessment Questions

    1. How does Isaac Sim's photorealistic rendering benefit Physical AI development?

    2. What are the key differences between CPU-based and GPU-accelerated ROS packages?

    3. How do you configure Nav2 parameters specifically for humanoid robot constraints?

    4. What safety considerations are unique to humanoid robot navigation?

    5. How does the integration of perception and navigation systems improve robot autonomy?

    ## Submission Requirements

    For each exercise, submit:
    - Configuration files and launch files
    - Source code for custom nodes
    - Simulation results and performance metrics
    - Screenshots of successful execution
    - A brief report documenting your implementation and findings

    ## Evaluation Rubric

    - **Functionality** (40%): Systems work as expected and meet requirements
    - **Integration** (25%): Components work together seamlessly
    - **Safety Considerations** (20%): Proper safety checks and emergency procedures
    - **Documentation** (15%): Clear explanations and proper documentation

    Complete all exercises to gain comprehensive experience with NVIDIA Isaac's AI-Robot Brain capabilities for Physical AI and humanoid robotics applications.
  </div>
  <div className="urdu">
    # ماڈیول 3 عملی مشقیں

    ## جائزہ

    یہ سیکشن NVIDIA Isaac کی آپ کی سمجھ کو مضبوط کرنے کے لیے عملی مشقوں پر مشتمل ہے، جس میں فوٹو ریئلسٹک سیمولیشن کے لیے Isaac Sim، ہارڈویئر ایکسلریٹڈ پرسیپشن اور نیویگیشن کے لیے Isaac ROS، اور ہیومنائیڈ روبوٹس میں راستے کی منصوبہ بندی کے لیے Nav2 شامل ہیں۔ یہ مشقیں آپ کو Physical AI ایپلی کیشنز کے لیے ضروری AI-Robot Brain اجزاء کے ساتھ عملی تجربہ حاصل کرنے میں مدد کریں گی۔

    ## مشق 1: Isaac Sim انسٹالیشن اور بنیادی سیٹ اپ

    ### مقصد
    NVIDIA Isaac Sim کو انسٹال اور کنفیگر کریں، پھر ہیومنائیڈ روبوٹ کے ساتھ ایک بنیادی سیمولیشن ماحول بنائیں۔

    ### ہدایات
    1.  ہارڈویئر کی ضروریات (RTX 4070 Ti یا اس سے زیادہ) کے بعد Isaac Sim انسٹال کریں۔
    2.  Isaac Sim لانچ کریں اور انٹرفیس سے واقف ہوں۔
    3.  ایک سادہ ماحول بنائیں جس میں زمینی ہوائی جہاز اور بنیادی روشنی ہو۔
    4.  منظر میں ایک ہیومنائیڈ روبوٹ ماڈل شامل کریں۔
    5.  روبوٹ کو مناسب فزکس خصوصیات کے ساتھ کنفیگر کریں۔
    6.  اس بات کی تصدیق کریں کہ روبوٹ کشش ثقل اور ٹکراؤ پر صحیح ردعمل ظاہر کرتا ہے۔

    ### ضروری اجزاء
    *   مناسب ہارڈ ویئر کے ساتھ Isaac Sim کی تنصیب
    *   زمینی ہوائی جہاز اور روشنی کے ساتھ بنیادی منظر
    *   فزکس کنفیگریشن کے ساتھ ہیومنائیڈ روبوٹ ماڈل
    *   فزکس سیمولیشن کی تصدیق

    ### متوقع آؤٹ پٹ
    *   Isaac Sim بغیر کسی غلطی کے لانچ ہوتا ہے۔
    *   روبوٹ گرتا ہے اور زمین پر مستحکم ہوتا ہے۔
    *   فزکس سیمولیشن آسانی سے چلتی ہے۔
    *   بنیادی منظر مناسب طریقے سے ترتیب دیا گیا ہے۔

    ## مشق 2: Isaac ROS Visual SLAM کا نفاذ

    ### مقصد
    سیمولیشن میں ہیومنائیڈ روبوٹ کے ساتھ Isaac ROS Visual SLAM کو نافذ اور ٹیسٹ کریں۔

    ### ہدایات
    1.  اپنے سسٹم پر Isaac ROS پیکجز انسٹال کریں۔
    2.  ہیومنائیڈ روبوٹ کے لیے سٹیریو کیمرہ سیٹ اپ کنفیگر کریں۔
    3.  مناسب پیرامیٹرز کے ساتھ Isaac ROS Visual SLAM نوڈ سیٹ اپ کریں۔
    4.  سیمولیٹڈ ماحول میں SLAM کی کارکردگی کی جانچ کریں۔
    5.  تیار کردہ نقشے اور لوکلائزیشن کے معیار کا اندازہ لگائیں۔

    ### ضروری اجزاء
    *   Isaac ROS Visual SLAM پیکجز انسٹال ہیں۔
    *   روبوٹ کے لیے سٹیریو کیمرہ کنفیگریشن۔
    *   مناسب پیرامیٹر کنفیگریشن۔
    *   SLAM ٹیسٹنگ اور تشخیص۔

    ## مشق 3: Isaac ROS Stereo DNN آبجیکٹ ڈیٹیکشن

    ### مقصد
    Isaac ROS Stereo DNN کا استعمال کرتے ہوئے آبجیکٹ ڈیٹیکشن کو نافذ کریں اور ہیومنائیڈ نیویگیشن کے ساتھ ضم کریں۔

    ### ہدایات
    1.  مناسب نیورل نیٹ ورک کے ساتھ Isaac ROS Stereo DNN کو کنفیگر کریں۔
    2.  آبجیکٹ ڈیٹیکشن کے لیے کیمرہ ٹاپکس اور پیرامیٹرز سیٹ اپ کریں۔
    3.  ڈیٹیکشن کے نتائج پر کارروائی کریں اور ہیومنائیڈ سے متعلقہ اشیاء کے لیے فلٹر کریں۔
    4.  نیویگیشن سسٹم کے ساتھ ڈیٹیکشن کے نتائج کو ضم کریں۔
    5.  مختلف منظرناموں میں آبجیکٹ ڈیٹیکشن کی کارکردگی کی جانچ کریں۔

    ## مشق 4: ہیومنائیڈ روبوٹس کے لیے Nav2 پاتھ پلاننگ

    ### مقصد
    خصوصی پیرامیٹرز کے ساتھ ہیومنائیڈ روبوٹ نیویگیشن کے لیے Nav2 کو کنفیگر اور ٹیسٹ کریں۔

    ### ہدایات
    1.  ہیومنائیڈ کے لیے مخصوص پیرامیٹرز کے ساتھ Nav2 کو انسٹال اور کنفیگر کریں۔
    2.  مناسب افراط زر اور روبوٹ کے رداس کے ساتھ کاسٹ میپس (costmaps) ترتیب دیں۔
    3.  ہیومنائیڈ رکاوٹوں کے لیے مقامی اور عالمی منصوبہ سازوں کو کنفیگر کریں۔
    4.  مختلف رکاوٹوں کے ساتھ سیمولیشن میں نیویگیشن کی جانچ کریں۔
    5.  راستے کے معیار اور نیویگیشن کی کارکردگی کا اندازہ لگائیں۔

    ## مشق 5: Isaac ROS اور Nav2 انٹیگریشن

    ### مقصد
    مکمل AI-Robot Brain فعالیت کے لیے Isaac ROS پرسیپشن کو Nav2 نیویگیشن کے ساتھ ضم کریں۔

    ### ہدایات
    1.  لوکلائزیشن کے لیے Isaac ROS Visual SLAM سیٹ اپ کریں۔
    2.  آبجیکٹ ڈیٹیکشن کے لیے Isaac ROS Stereo DNN کنفیگر کریں۔
    3.  Nav2 کاسٹ میپس کے ساتھ پرسیپشن ڈیٹا کو ضم کریں۔
    4.  متحرک رکاوٹ سے بچاؤ کو نافذ کریں۔
    5.  سیمولیشن میں مکمل سسٹم کی جانچ کریں۔

    ### متوقع آؤٹ پٹ
    *   Isaac ROS اور Nav2 کا مکمل انضمام۔
    *   ریئل ٹائم پرسیپشن اور نیویگیشن۔
    *   حفاظتی چیک اور ہنگامی ردعمل۔
    *   مربوط نظام کا رویہ۔

    ## تشخیصی سوالات

    1.  Isaac Sim کی فوٹو ریئلسٹک رینڈرنگ Physical AI کی ترقی کو کیسے فائدہ دیتی ہے؟
    2.  CPU پر مبنی اور GPU ایکسلریٹڈ ROS پیکجز کے درمیان اہم فرق کیا ہیں؟
    3.  آپ خاص طور پر ہیومنائیڈ روبوٹ کی رکاوٹوں کے لیے Nav2 پیرامیٹرز کو کیسے کنفیگر کرتے ہیں؟
    4.  ہیومنائیڈ روبوٹ نیویگیشن کے لیے کون سے حفاظتی تحفظات منفرد ہیں؟
    5.  پرسیپشن اور نیویگیشن سسٹمز کا انضمام روبوٹ کی خود مختاری کو کیسے بہتر بناتا ہے؟

    ## جمع کرانے کے تقاضے

    ہر مشق کے لیے، جمع کرائیں:
    *   کنفیگریشن فائلیں اور لانچ فائلیں۔
    *   کسٹم نوڈز کے لیے سورس کوڈ۔
    *   سیمولیشن کے نتائج اور کارکردگی کے میٹرکس۔
    *   کامیاب عملدرآمد کے اسکرین شاٹس۔
    *   آپ کے نفاذ اور نتائج کو دستاویزی شکل دینے والی ایک مختصر رپورٹ۔

    ## تشخیصی روبرک (Evaluation Rubric)

    *   **فعالیت** (40%): سسٹمز توقع کے مطابق کام کرتے ہیں اور ضروریات کو پورا کرتے ہیں۔
    *   **انٹیگریشن** (25%): اجزاء بغیر کسی رکاوٹ کے مل کر کام کرتے ہیں۔
    *   **حفاظتی تحفظات** (20%): مناسب حفاظتی چیک اور ہنگامی طریقہ کار۔
    *   **دستاویزی** (15%): واضح وضاحتیں اور مناسب دستاویزات۔

    Physical AI اور ہیومنائیڈ روبوٹکس ایپلی کیشنز کے لیے NVIDIA Isaac کی AI-Robot Brain صلاحیتوں کے ساتھ جامع تجربہ حاصل کرنے کے لیے تمام مشقیں مکمل کریں۔
  </div>
</BilingualChapter>