---
id: isaac-ros
title: "Isaac ROS - Hardware-Accelerated VSLAM and Navigation"
sidebar_position: 3
---

import BilingualChapter from '@site/src/components/BilingualChapter';

<BilingualChapter>
  <div className="english">
    # Isaac ROS - Hardware-Accelerated VSLAM and Navigation

    ## Introduction

    Isaac ROS is a collection of hardware-accelerated perception and navigation packages that leverage NVIDIA's GPU computing capabilities to provide real-time performance for robotics applications. Unlike traditional CPU-based ROS packages, Isaac ROS packages are optimized to run on NVIDIA GPUs, delivering 10-100x performance improvements for computationally intensive tasks like Visual SLAM (VSLAM), computer vision, and sensor processing.

    For humanoid robots operating in dynamic environments, Isaac ROS enables real-time perception and navigation capabilities that would be impossible with CPU-only processing, making it essential for Physical AI applications.

    ## Isaac ROS Architecture

    ### Core Components

    Isaac ROS provides several key hardware-accelerated packages:

    1. **Isaac ROS Visual SLAM**: GPU-accelerated Simultaneous Localization and Mapping
    2. **Isaac ROS Stereo DNN**: Accelerated deep neural network processing for stereo vision
    3. **Isaac ROS Apriltag**: GPU-accelerated AprilTag detection
    4. **Isaac ROS NITROS**: Network Interface for Time-based, Resolved, and Ordered communication
    5. **Isaac ROS Image Pipeline**: Optimized image processing pipeline
    6. **Isaac ROS Point Cloud**: Accelerated point cloud processing

    ### Hardware Acceleration Stack

    ```
    Application Layer (ROS 2 Nodes)
            ↓
    Isaac ROS Packages (GPU Accelerated)
            ↓
    CUDA/ cuDNN/ TensorRT Libraries
            ↓
    NVIDIA GPU Hardware
    ```

    ## Installing Isaac ROS

    ### System Requirements

    - **GPU**: NVIDIA RTX 4070 Ti or higher (RTX 3090/4090 recommended)
    - **Driver**: NVIDIA driver 535 or later
    - **CUDA**: CUDA 12.0 or later
    - **OS**: Ubuntu 22.04 LTS with ROS 2 Humble

    ### Installation Process

    1. **Install NVIDIA Container Toolkit**:
       ```bash
       # Add NVIDIA package repository
       curl -sL https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
       distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
       curl -sL https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
         sudo tee /etc/apt/sources.list.d/nvidia-docker.list

       # Install nvidia-container-toolkit
       sudo apt-get update
       sudo apt-get install -y nvidia-container-toolkit
       sudo systemctl restart docker
       ```

    2. **Install Isaac ROS Packages**:
       ```bash
       # Install Isaac ROS packages
       sudo apt update
       sudo apt install ros-humble-isaac-ros-common
       sudo apt install ros-humble-isaac-ros-visual-slam
       sudo apt install ros-humble-isaac-ros-stereo-dnn
       sudo apt install ros-humble-isaac-ros-apriltag
       sudo apt install ros-humble-isaac-ros-point-cloud
       ```

    3. **Verify Installation**:
       ```bash
       # Check if Isaac ROS packages are available
       ros2 pkg list | grep isaac_ros
       ```

    ## Isaac ROS Visual SLAM

    ### Overview

    Isaac ROS Visual SLAM provides GPU-accelerated Visual SLAM capabilities, enabling robots to simultaneously localize themselves and build maps of their environment using only camera sensors. This is particularly valuable for humanoid robots that need to navigate in unknown environments.

    ### Key Features

    - **Real-time Performance**: Up to 30 FPS on supported hardware
    - **Multi-camera Support**: Stereo and RGB-D camera configurations
    - **GPU-accelerated Tracking**: Feature detection and matching on GPU
    - **Loop Closure**: Detection and correction of mapping loops
    - **IMU Integration**: Fuses IMU data for more stable tracking

    ### Configuration and Usage

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

    ### Launch File Example

    ```xml
    <?xml version="1.0"?>
    <launch>
      <!-- Isaac ROS Visual SLAM -->
      <node pkg="isaac_ros_visual_slam" exec="isaac_ros_visual_slam_node" name="visual_slam" output="screen">
        <param from="$(find-pkg-share my_robot_config)/config/visual_slam_config.yaml"/>
      </node>

      <!-- Optional: Rviz2 for visualization -->
      <node pkg="rviz2" exec="rviz2" name="rviz2" args="-d $(find-pkg-share my_robot_config)/rviz/visual_slam.rviz"/>
    </launch>
    ```

    ### Integration with Humanoid Robots

    For humanoid robots, Visual SLAM provides critical capabilities:

    ```python
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image, Imu
    from tf2_ros import TransformBroadcaster
    import numpy as np

    class HumanoidVisualSlamNode(Node):
        def __init__(self):
            super().__init__('humanoid_visual_slam')

            # Publishers and subscribers
            self.left_image_sub = self.create_subscription(
                Image, '/camera/left/image_rect_color', self.left_image_callback, 10
            )
            self.right_image_sub = self.create_subscription(
                Image, '/camera/right/image_rect_color', self.right_image_callback, 10
            )
            self.imu_sub = self.create_subscription(
                Imu, '/imu/data', self.imu_callback, 10
            )

            # SLAM result publisher
            self.odom_pub = self.create_publisher(Odometry, '/visual_slam/odometry', 10)
            self.pose_pub = self.create_publisher(PoseStamped, '/visual_slam/pose', 10)

            # TF broadcaster
            self.tf_broadcaster = TransformBroadcaster(self)

            # SLAM state
            self.current_pose = np.eye(4)  # 4x4 transformation matrix
            self.has_initialized = False

            self.get_logger().info('Humanoid Visual SLAM node initialized')

        def left_image_callback(self, msg):
            """Process left camera image for stereo SLAM"""
            if not self.has_initialized:
                self.initialize_slam(msg)
                self.has_initialized = True

            # Process image with Isaac ROS Visual SLAM (in practice, this would interface
            # with the actual Isaac ROS node)
            self.process_stereo_image_pair(msg, self.last_right_image)

        def right_image_callback(self, msg):
            """Process right camera image for stereo SLAM"""
            self.last_right_image = msg

        def imu_callback(self, msg):
            """Process IMU data for sensor fusion"""
            # In practice, IMU data would be fused with visual data
            # in the Isaac ROS Visual SLAM node
            pass

        def process_stereo_image_pair(self, left_img, right_img):
            """Process stereo image pair and update pose estimate"""
            # This is a simplified example - in practice, this would interface
            # with the Isaac ROS Visual SLAM node
            if left_img and right_img:
                # Simulate pose update from SLAM
                delta_pose = self.estimate_motion(left_img, right_img)
                self.current_pose = self.current_pose @ delta_pose

                # Publish odometry
                odom_msg = Odometry()
                odom_msg.header.stamp = self.get_clock().now().to_msg()
                odom_msg.header.frame_id = 'map'
                odom_msg.child_frame_id = 'base_link'

                # Convert transformation matrix to pose
                position = self.current_pose[:3, 3]
                orientation = self.matrix_to_quaternion(self.current_pose[:3, :3])

                odom_msg.pose.pose.position.x = float(position[0])
                odom_msg.pose.pose.position.y = float(position[1])
                odom_msg.pose.pose.position.z = float(position[2])
                odom_msg.pose.pose.orientation.x = float(orientation[0])
                odom_msg.pose.pose.orientation.y = float(orientation[1])
                odom_msg.pose.pose.orientation.z = float(orientation[2])
                odom_msg.pose.pose.orientation.w = float(orientation[3])

                self.odom_pub.publish(odom_msg)

                # Broadcast TF
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = 'map'
                t.child_frame_id = 'base_link'
                t.transform.translation.x = float(position[0])
                t.transform.translation.y = float(position[1])
                t.transform.translation.z = float(position[2])
                t.transform.rotation.x = float(orientation[0])
                t.transform.rotation.y = float(orientation[1])
                t.transform.rotation.z = float(orientation[2])
                t.transform.rotation.w = float(orientation[3])

                self.tf_broadcaster.sendTransform(t)

        def estimate_motion(self, left_img, right_img):
            """Estimate motion between consecutive stereo pairs"""
            # Simplified motion estimation - in practice, Isaac ROS handles this
            # with GPU-accelerated feature matching and pose estimation
            dt = 0.1  # Assume 10Hz processing
            linear_vel = 0.1  # 0.1 m/s forward
            angular_vel = 0.0  # No rotation

            # Create incremental transformation
            delta_x = linear_vel * dt
            delta_theta = angular_vel * dt

            delta_pose = np.array([
                [np.cos(delta_theta), -np.sin(delta_theta), 0, delta_x * np.cos(delta_theta)],
                [np.sin(delta_theta), np.cos(delta_theta), 0, delta_x * np.sin(delta_theta)],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

            return delta_pose

        def matrix_to_quaternion(self, rotation_matrix):
            """Convert 3x3 rotation matrix to quaternion"""
            # Simplified conversion - in practice, use tf2 or scipy
            trace = np.trace(rotation_matrix)
            if trace > 0:
                s = np.sqrt(trace + 1.0) * 2
                qw = 0.25 * s
                qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
                qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
                qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
            else:
                # Handle other cases...
                qw, qx, qy, qz = 1, 0, 0, 0

            return np.array([qx, qy, qz, qw])

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidVisualSlamNode()

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

    ## Isaac ROS Stereo DNN

    ### Overview

    Isaac ROS Stereo DNN provides GPU-accelerated deep learning inference for stereo vision applications, enabling real-time object detection, segmentation, and classification using neural networks.

    ### Configuration

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
        model_type: "detectnet"  # Options: detectnet, segnet, classify
        model_name: "resnet18_detector"
        confidence_threshold: 0.5
        max_objects: 100

        # Performance parameters
        input_width: 960
        input_height: 544
        batch_size: 1
    ```

    ### Object Detection Example

    ```python
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
    from isaac_ros_detectnet_interfaces.msg import Detection2DArray as IsaacDetectionArray
    import numpy as np

    class HumanoidObjectDetectionNode(Node):
        def __init__(self):
            super().__init__('humanoid_object_detection')

            # Subscribe to processed images from Isaac ROS Stereo DNN
            self.detection_sub = self.create_subscription(
                IsaacDetectionArray,
                '/detectnet/detections',
                self.detection_callback,
                10
            )

            # Publisher for filtered detections relevant to humanoid navigation
            self.filtered_pub = self.create_publisher(
                Detection2DArray,
                '/humanoid/detections',
                10
            )

            # Objects of interest for humanoid robots
            self.target_objects = ['person', 'chair', 'table', 'door']

        def detection_callback(self, msg):
            """Process object detections from Isaac ROS Stereo DNN"""
            filtered_detections = Detection2DArray()
            filtered_detections.header = msg.header

            for detection in msg.detections:
                # Filter for objects relevant to humanoid navigation
                if detection.results[0].class_name in self.target_objects:
                    # Convert Isaac format to standard vision_msgs format
                    std_detection = self.convert_detection_format(detection)
                    filtered_detections.detections.append(std_detection)

            # Publish filtered detections
            if len(filtered_detections.detections) > 0:
                self.filtered_pub.publish(filtered_detections)

        def convert_detection_format(self, isaac_detection):
            """Convert Isaac ROS detection format to standard format"""
            std_detection = Detection2D()

            # Convert bounding box
            std_detection.bbox.center.x = float(isaac_detection.bbox.center.x)
            std_detection.bbox.center.y = float(isaac_detection.bbox.center.y)
            std_detection.bbox.size_x = float(isaac_detection.bbox.size_x)
            std_detection.bbox.size_y = float(isaac_detection.bbox.size_y)

            # Convert classification result
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = isaac_detection.results[0].class_name
            hypothesis.hypothesis.score = float(isaac_detection.results[0].confidence)

            std_detection.results.append(hypothesis)

            return std_detection

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidObjectDetectionNode()

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

    ## Isaac ROS Point Cloud Processing

    ### Overview

    Isaac ROS provides GPU-accelerated point cloud processing capabilities, essential for 3D perception in humanoid robots.

    ### Point Cloud Fusion Example

    ```python
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    import numpy as np
    import sensor_msgs_py.point_cloud2 as pc2

    class HumanoidPointCloudFusionNode(Node):
        def __init__(self):
            super().__init__('humanoid_pointcloud_fusion')

            # Subscribers for different point cloud sources
            self.depth_sub = self.create_subscription(
                PointCloud2,
                '/camera/depth/color/points',
                self.depth_cloud_callback,
                10
            )

            self.lidar_sub = self.create_subscription(
                PointCloud2,
                '/velodyne_points',
                self.lidar_cloud_callback,
                10
            )

            # Publisher for fused point cloud
            self.fused_pub = self.create_publisher(
                PointCloud2,
                '/fused_pointcloud',
                10
            )

            # Store point clouds for fusion
            self.depth_cloud = None
            self.lidar_cloud = None

        def depth_cloud_callback(self, msg):
            """Process depth camera point cloud"""
            self.depth_cloud = msg
            self.fuse_pointclouds()

        def lidar_cloud_callback(self, msg):
            """Process LiDAR point cloud"""
            self.lidar_cloud = msg
            self.fuse_pointclouds()

        def fuse_pointclouds(self):
            """Fuse depth and LiDAR point clouds"""
            if self.depth_cloud is None or self.lidar_cloud is None:
                return

            # Convert to numpy arrays for processing
            depth_points = np.array(list(pc2.read_points(
                self.depth_cloud,
                field_names=("x", "y", "z"),
                skip_nans=True
            )))

            lidar_points = np.array(list(pc2.read_points(
                self.lidar_cloud,
                field_names=("x", "y", "z"),
                skip_nans=True
            )))

            # Transform LiDAR points to camera frame if needed
            # (simplified - in practice, use tf2 for transformations)
            fused_points = np.vstack([depth_points, lidar_points])

            # Create fused point cloud message
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = self.depth_cloud.header.frame_id

            # Create PointCloud2 message
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]

            fused_cloud_msg = pc2.create_cloud(header, fields, fused_points)
            self.fused_pub.publish(fused_cloud_msg)

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidPointCloudFusionNode()

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

    ## Isaac ROS NITROS (Network Interface for Time-based, Resolved, and Ordered communication)

    ### Overview

    NITROS is a key technology in Isaac ROS that optimizes data transmission between nodes by preserving temporal relationships and reducing CPU overhead.

    ### NITROS Configuration

    ```yaml
    # nitros_config.yaml
    /**:
      ros__parameters:
        # Enable NITROS for specific topics
        use_nitros: true
        nitros_subscribers:
          - topic_name: "/camera/left/image_rect_color"
            type: "nitros_image"
            qos:
              history: "keep_last"
              depth: 1
              reliability: "reliable"
              durability: "volatile"

        nitros_publishers:
          - topic_name: "/visual_slam/tracking/feature"
            type: "nitros_feature_array"
            qos:
              history: "keep_last"
              depth: 1
              reliability: "reliable"
              durability: "volatile"
    ```

    ## Performance Optimization with Isaac ROS

    ### GPU Memory Management

    ```python
    import rclpy
    from rclpy.node import Node
    import pycuda.driver as cuda
    import pycuda.autoinit

    class IsaacROSMemoryManager(Node):
        def __init__(self):
            super().__init__('isaac_ros_memory_manager')

            # Monitor GPU memory usage
            self.timer = self.create_timer(1.0, self.monitor_gpu_memory)

        def monitor_gpu_memory(self):
            """Monitor GPU memory usage and log warnings"""
            try:
                # Get GPU memory info (simplified - in practice, use pynvml)
                free_mem, total_mem = cuda.mem_get_info()
                used_mem = total_mem - free_mem
                usage_percent = (used_mem / total_mem) * 100

                if usage_percent > 90:
                    self.get_logger().warn(
                        f'GPU memory usage is high: {usage_percent:.1f}%'
                    )
                elif usage_percent > 75:
                    self.get_logger().info(
                        f'GPU memory usage: {usage_percent:.1f}%'
                    )

            except Exception as e:
                self.get_logger().error(f'GPU memory monitoring error: {e}')
    ```

    ### Pipeline Optimization

    For humanoid robots, optimizing the perception pipeline is crucial:

    ```python
    class OptimizedPerceptionPipeline(Node):
        def __init__(self):
            super().__init__('optimized_perception_pipeline')

            # Use Isaac ROS image pipeline for optimized processing
            self.image_sub = self.create_subscription(
                Image,
                '/camera/image_raw',
                self.optimized_image_callback,
                5  # Reduced queue size for lower latency
            )

            # Throttle processing based on robot state
            self.processing_enabled = True
            self.processing_rate = 10.0  # Hz
            self.last_process_time = 0.0

        def optimized_image_callback(self, msg):
            """Optimized image processing with throttling"""
            current_time = self.get_clock().now().nanoseconds / 1e9

            if (current_time - self.last_process_time) >= (1.0 / self.processing_rate):
                if self.processing_enabled:
                    self.process_image_optimized(msg)
                self.last_process_time = current_time

        def process_image_optimized(self, image_msg):
            """Optimized image processing using Isaac ROS"""
            # This would interface with Isaac ROS nodes in practice
            pass
    ```

    ## Integration with Navigation Systems

    ### Nav2 Integration

    Isaac ROS works seamlessly with Nav2 for advanced navigation:

    ```python
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionClient

    class IsaacROSNav2Integrator(Node):
        def __init__(self):
            super().__init__('isaac_ros_nav2_integrator')

            # Action client for Nav2
            self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

            # Subscribe to Isaac ROS pose estimates
            self.pose_sub = self.create_subscription(
                PoseStamped,
                '/visual_slam/pose',
                self.pose_callback,
                10
            )

            # Timer for navigation decisions
            self.nav_timer = self.create_timer(5.0, self.make_navigation_decision)

        def pose_callback(self, msg):
            """Update current pose from Isaac ROS"""
            self.current_pose = msg

        def make_navigation_decision(self):
            """Make navigation decisions based on current pose and goals"""
            # Example: Navigate to a predefined goal
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.pose.position.x = 5.0
            goal_pose.pose.position.y = 5.0
            goal_pose.pose.orientation.w = 1.0

            self.send_navigation_goal(goal_pose)

        def send_navigation_goal(self, pose):
            """Send navigation goal to Nav2"""
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = pose

            self.nav_client.wait_for_server()
            future = self.nav_client.send_goal_async(goal_msg)
            future.add_done_callback(self.navigation_result_callback)

        def navigation_result_callback(self, future):
            """Handle navigation result"""
            result = future.result()
            if result:
                self.get_logger().info('Navigation completed successfully')
            else:
                self.get_logger().error('Navigation failed')

def main(args=None):
    rclpy.init(args=args)
    node = IsaacROSNav2Integrator()

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

    ## Best Practices for Isaac ROS

    ### 1. Hardware Utilization

    - Monitor GPU utilization and memory usage
    - Use appropriate batch sizes for neural networks
    - Consider multi-GPU setups for complex applications

    ### 2. Data Pipeline Optimization

    - Use NITROS for optimized data transmission
    - Implement proper QoS settings for real-time performance
    - Minimize data copying between CPU and GPU

    ### 3. Algorithm Selection

    - Choose algorithms appropriate for GPU acceleration
    - Consider trade-offs between accuracy and speed
    - Validate results against CPU-based implementations

    ### 4. Error Handling

    - Implement fallback mechanisms for GPU failures
    - Monitor for CUDA errors and handle gracefully
    - Log performance metrics for optimization

    ## Troubleshooting Common Issues

    ### GPU Memory Issues
    - Reduce batch sizes for neural networks
    - Use lower resolution inputs
    - Implement memory pooling

    ### Performance Problems
    - Check GPU utilization and temperature
    - Verify CUDA installation and driver compatibility
    - Optimize data transfer between CPU and GPU

    ### Integration Issues
    - Verify ROS 2 message type compatibility
    - Check frame ID consistency across nodes
    - Validate timing and synchronization

    ## Advanced Features

    ### Custom Isaac ROS Extensions

    For specialized humanoid applications, you can create custom Isaac ROS extensions:

    ```cpp
    // Example custom Isaac ROS component
    #include <rclcpp/rclcpp.hpp>
    #include <isaac_ros_nitros/nitros_node.hpp>
    #include <sensor_msgs/msg/image.hpp>

    class HumanoidPerceptionNode : public rclcpp::Node
    {
    public:
      explicit HumanoidPerceptionNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : rclcpp::Node("humanoid_perception", options)
      {
        // Initialize Isaac ROS Nitros publisher/subscriber
        // with custom data types for humanoid-specific processing
      }

    private:
      // Custom processing methods
    };
    ```

    ## Hands-on Exercise

    Create a complete Isaac ROS perception pipeline that includes:

    1. Isaac ROS Visual SLAM node for localization and mapping
    2. Isaac ROS Stereo DNN for object detection
    3. Isaac ROS Point Cloud processing for 3D perception
    4. NITROS configuration for optimized data flow
    5. Integration with Nav2 for navigation
    6. Performance monitoring and optimization

    This exercise will give you hands-on experience with Isaac ROS hardware-accelerated perception and navigation capabilities for humanoid robots.
  </div>
  <div className="urdu">
    # Isaac ROS - ہارڈویئر ایکسلریٹڈ VSLAM اور نیویگیشن

    ## تعارف

    Isaac ROS ہارڈویئر ایکسلریٹڈ پرسیپشن (perception) اور نیویگیشن پیکجز کا ایک مجموعہ ہے جو روبوٹکس ایپلی کیشنز کے لیے ریئل ٹائم کارکردگی فراہم کرنے کے لیے NVIDIA کی GPU کمپیوٹنگ صلاحیتوں کا فائدہ اٹھاتا ہے۔ روایتی CPU پر مبنی ROS پیکجز کے برعکس، Isaac ROS پیکجز کو NVIDIA GPUs پر چلانے کے لیے بہتر بنایا گیا ہے، جو Visual SLAM (VSLAM)، کمپیوٹر ویژن، اور سینسر پروسیسنگ جیسے کمپیوٹیشنل طور پر بھاری کاموں کے لیے 10-100 گنا کارکردگی میں بہتری لاتے ہیں.

    متحرک ماحول میں کام کرنے والے ہیومنائیڈ روبوٹس کے لیے، Isaac ROS ریئل ٹائم پرسیپشن اور نیویگیشن کی صلاحیتوں کو قابل بناتا ہے جو صرف CPU پروسیسنگ کے ساتھ ناممکن ہوں گی، جس سے یہ Physical AI ایپلی کیشنز کے لیے ضروری بن جاتا ہے.

    ## Isaac ROS آرکیٹیکچر

    ### بنیادی اجزاء

    Isaac ROS کئی اہم ہارڈویئر ایکسلریٹڈ پیکجز فراہم کرتا ہے:

    1.  **Isaac ROS Visual SLAM**: GPU سے تیز رفتار لوکلائزیشن اور میپنگ (Simultaneous Localization and Mapping)۔
    2.  **Isaac ROS Stereo DNN**: سٹیریو ویژن کے لیے تیز رفتار ڈیپ نیورل نیٹ ورک پروسیسنگ۔
    3.  **Isaac ROS Apriltag**: GPU سے تیز رفتار AprilTag ڈیٹیکشن۔
    4.  **Isaac ROS NITROS**: وقت پر مبنی، حل شدہ، اور ترتیب وار مواصلات کے لیے نیٹ ورک انٹرفیس۔
    5.  **Isaac ROS Image Pipeline**: آپٹمائزڈ امیج پروسیسنگ پائپ لائن۔
    6.  **Isaac ROS Point Cloud**: تیز رفتار پوائنٹ کلاؤڈ پروسیسنگ۔

    ### ہارڈویئر ایکسلریشن اسٹیک

    ```
    ایپلیکیشن لیئر (ROS 2 نوڈز)
            ↓
    Isaac ROS پیکجز (GPU ایکسلریٹڈ)
            ↓
    CUDA/ cuDNN/ TensorRT لائبریریز
            ↓
    NVIDIA GPU ہارڈویئر
    ```

    ## Isaac ROS کی تنصیب

    ### سسٹم کے تقاضے

    *   **GPU**: NVIDIA RTX 4070 Ti یا اس سے زیادہ (RTX 3090/4090 تجویز کردہ)
    *   **ڈرائیور**: NVIDIA ڈرائیور 535 یا بعد کا
    *   **CUDA**: CUDA 12.0 یا بعد کا
    *   **OS**: Ubuntu 22.04 LTS جس میں ROS 2 Humble ہو

    ### انسٹالیشن کا عمل

    1.  **NVIDIA کنٹینر ٹول کٹ انسٹال کریں**:
        ```bash
        # NVIDIA پیکیج ریپوزٹری شامل کریں
        curl -sL https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -sL https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
          sudo tee /etc/apt/sources.list.d/nvidia-docker.list

        # nvidia-container-toolkit انسٹال کریں
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        sudo systemctl restart docker
        ```

    2.  **Isaac ROS پیکجز انسٹال کریں**:
        ```bash
        # Isaac ROS پیکجز انسٹال کریں
        sudo apt update
        sudo apt install ros-humble-isaac-ros-common
        sudo apt install ros-humble-isaac-ros-visual-slam
        sudo apt install ros-humble-isaac-ros-stereo-dnn
        sudo apt install ros-humble-isaac-ros-apriltag
        sudo apt install ros-humble-isaac-ros-point-cloud
        ```

    3.  **تنصیب کی تصدیق کریں**:
        ```bash
        # چیک کریں کہ آیا Isaac ROS پیکجز دستیاب ہیں
        ros2 pkg list | grep isaac_ros
        ```

    ## Isaac ROS Visual SLAM

    ### جائزہ

    Isaac ROS Visual SLAM GPU سے تیز رفتار Visual SLAM صلاحیتیں فراہم کرتا ہے، جو روبوٹس کو بیک وقت خود کو لوکلائز کرنے اور صرف کیمرہ سینسرز کا استعمال کرتے ہوئے اپنے ماحول کا نقشہ بنانے کے قابل بناتا ہے۔ یہ ہیومنائیڈ روبوٹس کے لیے خاص طور پر قیمتی ہے جنہیں نامعلوم ماحول میں نیویگیٹ کرنے کی ضرورت ہوتی ہے.

    ### اہم خصوصیات

    *   **ریئل ٹائم کارکردگی**: معاون ہارڈویئر پر 30 FPS تک۔
    *   **ملٹی کیمرہ سپورٹ**: سٹیریو اور RGB-D کیمرہ کنفیگریشنز۔
    *   **GPU سے تیز رفتار ٹریکنگ**: GPU پر فیچر ڈیٹیکشن اور میچنگ۔
    *   **لوپ کلوزر**: میپنگ لوپس کا پتہ لگانا اور تصحیح کرنا۔
    *   **IMU انٹیگریشن**: زیادہ مستحکم ٹریکنگ کے لیے IMU ڈیٹا کو ضم کرتا ہے۔

    ### کنفیگریشن اور استعمال

    ```yaml
    # visual_slam_config.yaml
    /**:
      ros__parameters:
        # ان پٹ ٹاپکس
        camera_topic_left: "/camera/left/image_rect_color"
        camera_info_topic_left: "/camera/left/camera_info"
        camera_topic_right: "/camera/right/image_rect_color"
        camera_info_topic_right: "/camera/right/camera_info"
        imu_topic: "/imu/data"

        # پروسیسنگ پیرامیٹرز
        enable_debug_mode: false
        enable_mapping: true
        enable_localization: true
        enable_point_cloud_output: true

        # کارکردگی کے پیرامیٹرز
        max_num_points: 100000
        map_publish_period: 1.0
        tracking_rate: 30.0
    ```

    ### لانچ فائل کی مثال

    ```xml
    <?xml version="1.0"?>
    <launch>
      <!-- Isaac ROS Visual SLAM -->
      <node pkg="isaac_ros_visual_slam" exec="isaac_ros_visual_slam_node" name="visual_slam" output="screen">
        <param from="$(find-pkg-share my_robot_config)/config/visual_slam_config.yaml"/>
      </node>

      <!-- اختیاری: ویژولائزیشن کے لیے Rviz2 -->
      <node pkg="rviz2" exec="rviz2" name="rviz2" args="-d $(find-pkg-share my_robot_config)/rviz/visual_slam.rviz"/>
    </launch>
    ```

    ### ہیومنائیڈ روبوٹس کے ساتھ انٹیگریشن

    ہیومنائیڈ روبوٹس کے لیے، Visual SLAM اہم صلاحیتیں فراہم کرتا ہے:

    ```python
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image, Imu
    from tf2_ros import TransformBroadcaster
    import numpy as np

    class HumanoidVisualSlamNode(Node):
        def __init__(self):
            super().__init__('humanoid_visual_slam')

            # پبلشرز اور سبسکرائبرز
            self.left_image_sub = self.create_subscription(
                Image, '/camera/left/image_rect_color', self.left_image_callback, 10
            )
            self.right_image_sub = self.create_subscription(
                Image, '/camera/right/image_rect_color', self.right_image_callback, 10
            )
            self.imu_sub = self.create_subscription(
                Imu, '/imu/data', self.imu_callback, 10
            )

            # SLAM رزلٹ پبلشر
            self.odom_pub = self.create_publisher(Odometry, '/visual_slam/odometry', 10)
            self.pose_pub = self.create_publisher(PoseStamped, '/visual_slam/pose', 10)

            # TF براڈکاسٹر
            self.tf_broadcaster = TransformBroadcaster(self)

            # SLAM اسٹیٹ
            self.current_pose = np.eye(4)  # 4x4 ٹرانسفارمیشن میٹرکس
            self.has_initialized = False

            self.get_logger().info('Humanoid Visual SLAM node initialized')

        def left_image_callback(self, msg):
            """سٹیریو SLAM کے لیے بائیں کیمرے کی تصویر پر کارروائی کریں"""
            if not self.has_initialized:
                self.initialize_slam(msg)
                self.has_initialized = True

            # Isaac ROS Visual SLAM کے ساتھ تصویر پر کارروائی کریں (عملی طور پر، یہ
            # اصل Isaac ROS نوڈ کے ساتھ انٹرفیس کرے گا)
            self.process_stereo_image_pair(msg, self.last_right_image)

        def right_image_callback(self, msg):
            """سٹیریو SLAM کے لیے دائیں کیمرے کی تصویر پر کارروائی کریں"""
            self.last_right_image = msg

        def imu_callback(self, msg):
            """سینسر فیوژن کے لیے IMU ڈیٹا پر کارروائی کریں"""
            # عملی طور پر، IMU ڈیٹا کو بصری ڈیٹا کے ساتھ Isaac ROS Visual SLAM نوڈ میں ضم کیا جائے گا
            pass

        def process_stereo_image_pair(self, left_img, right_img):
            """سٹیریو امیج کے جوڑے پر کارروائی کریں اور پوز کا اندازہ اپ ڈیٹ کریں"""
            # یہ ایک آسان مثال ہے - عملی طور پر، یہ Isaac ROS Visual SLAM نوڈ کے ساتھ انٹرفیس کرے گا
            if left_img and right_img:
                # SLAM سے پوز اپ ڈیٹ کو سیمولیٹ کریں
                delta_pose = self.estimate_motion(left_img, right_img)
                self.current_pose = self.current_pose @ delta_pose

                # اوڈومیٹری شائع کریں
                odom_msg = Odometry()
                odom_msg.header.stamp = self.get_clock().now().to_msg()
                odom_msg.header.frame_id = 'map'
                odom_msg.child_frame_id = 'base_link'

                # ٹرانسفارمیشن میٹرکس کو پوز میں تبدیل کریں
                position = self.current_pose[:3, 3]
                orientation = self.matrix_to_quaternion(self.current_pose[:3, :3])

                odom_msg.pose.pose.position.x = float(position[0])
                odom_msg.pose.pose.position.y = float(position[1])
                odom_msg.pose.pose.position.z = float(position[2])
                odom_msg.pose.pose.orientation.x = float(orientation[0])
                odom_msg.pose.pose.orientation.y = float(orientation[1])
                odom_msg.pose.pose.orientation.z = float(orientation[2])
                odom_msg.pose.pose.orientation.w = float(orientation[3])

                self.odom_pub.publish(odom_msg)

                # TF براڈکاسٹ کریں
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = 'map'
                t.child_frame_id = 'base_link'
                t.transform.translation.x = float(position[0])
                t.transform.translation.y = float(position[1])
                t.transform.translation.z = float(position[2])
                t.transform.rotation.x = float(orientation[0])
                t.transform.rotation.y = float(orientation[1])
                t.transform.rotation.z = float(orientation[2])
                t.transform.rotation.w = float(orientation[3])

                self.tf_broadcaster.sendTransform(t)

        def estimate_motion(self, left_img, right_img):
            """مسلسل سٹیریو جوڑوں کے درمیان حرکت کا اندازہ لگائیں"""
            # آسان حرکت کا اندازہ - عملی طور پر، Isaac ROS اسے
            # GPU ایکسلریٹڈ فیچر میچنگ اور پوز کے اندازے کے ساتھ سنبھالتا ہے
            dt = 0.1  # فرض کریں 10Hz پروسیسنگ
            linear_vel = 0.1  # 0.1 میٹر/سیکنڈ آگے
            angular_vel = 0.0  # کوئی گردش نہیں

            # انکریمنٹل ٹرانسفارمیشن بنائیں
            delta_x = linear_vel * dt
            delta_theta = angular_vel * dt

            delta_pose = np.array([
                [np.cos(delta_theta), -np.sin(delta_theta), 0, delta_x * np.cos(delta_theta)],
                [np.sin(delta_theta), np.cos(delta_theta), 0, delta_x * np.sin(delta_theta)],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

            return delta_pose

        def matrix_to_quaternion(self, rotation_matrix):
            """3x3 روٹیشن میٹرکس کو کواٹرنین میں تبدیل کریں"""
            # آسان تبدیلی - عملی طور پر، tf2 یا scipy استعمال کریں
            trace = np.trace(rotation_matrix)
            if trace > 0:
                s = np.sqrt(trace + 1.0) * 2
                qw = 0.25 * s
                qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
                qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
                qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
            else:
                # دوسرے کیسز کو ہینڈل کریں...
                qw, qx, qy, qz = 1, 0, 0, 0

            return np.array([qx, qy, qz, qw])

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidVisualSlamNode()

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

    ## Isaac ROS Stereo DNN

    ### جائزہ

    Isaac ROS Stereo DNN سٹیریو ویژن ایپلی کیشنز کے لیے GPU سے تیز رفتار ڈیپ لرننگ انفرنس (inference) فراہم کرتا ہے، جو نیورل نیٹ ورکس کا استعمال کرتے ہوئے ریئل ٹائم آبجیکٹ ڈیٹیکشن، سیگمنٹیشن، اور درجہ بندی کو قابل بناتا ہے۔

    ### کنفیگریشن

    ```yaml
    # stereo_dnn_config.yaml
    /**:
      ros__parameters:
        # ان پٹ ٹاپکس
        left_image_topic: "/camera/left/image_rect_color"
        right_image_topic: "/camera/right/image_rect_color"
        left_camera_info_topic: "/camera/left/camera_info"
        right_camera_info_topic: "/camera/right/camera_info"

        # نیورل نیٹ ورک پیرامیٹرز
        model_type: "detectnet"  # اختیارات: detectnet, segnet, classify
        model_name: "resnet18_detector"
        confidence_threshold: 0.5
        max_objects: 100

        # کارکردگی کے پیرامیٹرز
        input_width: 960
        input_height: 544
        batch_size: 1
    ```

    ### آبجیکٹ ڈیٹیکشن کی مثال

    ```python
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
    from isaac_ros_detectnet_interfaces.msg import Detection2DArray as IsaacDetectionArray
    import numpy as np

    class HumanoidObjectDetectionNode(Node):
        def __init__(self):
            super().__init__('humanoid_object_detection')

            # Isaac ROS Stereo DNN سے پروسیس شدہ تصاویر کو سبسکرائب کریں
            self.detection_sub = self.create_subscription(
                IsaacDetectionArray,
                '/detectnet/detections',
                self.detection_callback,
                10
            )

            # ہیومنائیڈ نیویگیشن سے متعلق فلٹر شدہ ڈیٹیکشنز کے لیے پبلشر
            self.filtered_pub = self.create_publisher(
                Detection2DArray,
                '/humanoid/detections',
                10
            )

            # ہیومنائیڈ روبوٹس کے لیے دلچسپی کی اشیاء
            self.target_objects = ['person', 'chair', 'table', 'door']

        def detection_callback(self, msg):
            """Isaac ROS Stereo DNN سے آبجیکٹ ڈیٹیکشنز پر کارروائی کریں"""
            filtered_detections = Detection2DArray()
            filtered_detections.header = msg.header

            for detection in msg.detections:
                # ہیومنائیڈ نیویگیشن سے متعلق اشیاء کے لیے فلٹر کریں
                if detection.results[0].class_name in self.target_objects:
                    # Isaac فارمیٹ کو معیاری vision_msgs فارمیٹ میں تبدیل کریں
                    std_detection = self.convert_detection_format(detection)
                    filtered_detections.detections.append(std_detection)

            # فلٹر شدہ ڈیٹیکشنز شائع کریں
            if len(filtered_detections.detections) > 0:
                self.filtered_pub.publish(filtered_detections)

        def convert_detection_format(self, isaac_detection):
            """Isaac ROS ڈیٹیکشن فارمیٹ کو معیاری فارمیٹ میں تبدیل کریں"""
            std_detection = Detection2D()

            # باؤنڈنگ باکس تبدیل کریں
            std_detection.bbox.center.x = float(isaac_detection.bbox.center.x)
            std_detection.bbox.center.y = float(isaac_detection.bbox.center.y)
            std_detection.bbox.size_x = float(isaac_detection.bbox.size_x)
            std_detection.bbox.size_y = float(isaac_detection.bbox.size_y)

            # درجہ بندی کے نتیجے کو تبدیل کریں
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = isaac_detection.results[0].class_name
            hypothesis.hypothesis.score = float(isaac_detection.results[0].confidence)

            std_detection.results.append(hypothesis)

            return std_detection

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidObjectDetectionNode()

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

    ## Isaac ROS Point Cloud پروسیسنگ

    ### جائزہ

    Isaac ROS GPU سے تیز رفتار پوائنٹ کلاؤڈ پروسیسنگ کی صلاحیتیں فراہم کرتا ہے، جو ہیومنائیڈ روبوٹس میں 3D پرسیپشن کے لیے ضروری ہے۔

    ### پوائنٹ کلاؤڈ فیوژن کی مثال

    ```python
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    import numpy as np
    import sensor_msgs_py.point_cloud2 as pc2

    class HumanoidPointCloudFusionNode(Node):
        def __init__(self):
            super().__init__('humanoid_pointcloud_fusion')

            # مختلف پوائنٹ کلاؤڈ ذرائع کے لیے سبسکرائبرز
            self.depth_sub = self.create_subscription(
                PointCloud2,
                '/camera/depth/color/points',
                self.depth_cloud_callback,
                10
            )

            self.lidar_sub = self.create_subscription(
                PointCloud2,
                '/velodyne_points',
                self.lidar_cloud_callback,
                10
            )

            # فیوزڈ پوائنٹ کلاؤڈ کے لیے پبلشر
            self.fused_pub = self.create_publisher(
                PointCloud2,
                '/fused_pointcloud',
                10
            )

            # فیوژن کے لیے پوائنٹ کلاؤڈز اسٹور کریں
            self.depth_cloud = None
            self.lidar_cloud = None

        def depth_cloud_callback(self, msg):
            """گہرائی والے کیمرے کے پوائنٹ کلاؤڈ پر کارروائی کریں"""
            self.depth_cloud = msg
            self.fuse_pointclouds()

        def lidar_cloud_callback(self, msg):
            """LiDAR پوائنٹ کلاؤڈ پر کارروائی کریں"""
            self.lidar_cloud = msg
            self.fuse_pointclouds()

        def fuse_pointclouds(self):
            """ڈیپتھ اور LiDAR پوائنٹ کلاؤڈز کو فیوز کریں"""
            if self.depth_cloud is None or self.lidar_cloud is None:
                return

            # پروسیسنگ کے لیے numpy arrays میں تبدیل کریں
            depth_points = np.array(list(pc2.read_points(
                self.depth_cloud,
                field_names=("x", "y", "z"),
                skip_nans=True
            )))

            lidar_points = np.array(list(pc2.read_points(
                self.lidar_cloud,
                field_names=("x", "y", "z"),
                skip_nans=True
            )))

            # LiDAR پوائنٹس کو کیمرہ فریم میں تبدیل کریں اگر ضرورت ہو
            # (آسان - عملی طور پر، تبدیلیوں کے لیے tf2 کا استعمال کریں)
            fused_points = np.vstack([depth_points, lidar_points])

            # فیوزڈ پوائنٹ کلاؤڈ میسج بنائیں
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = self.depth_cloud.header.frame_id

            # PointCloud2 میسج بنائیں
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]

            fused_cloud_msg = pc2.create_cloud(header, fields, fused_points)
            self.fused_pub.publish(fused_cloud_msg)

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidPointCloudFusionNode()

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

    ## Isaac ROSNITROS (Network Interface for Time-based, Resolved, and Ordered communication)

    ### جائزہ

    NITROS Isaac ROS میں ایک کلیدی ٹیکنالوجی ہے جو نوڈز کے درمیان ڈیٹا کی منتقلی کو وقتی تعلقات کو محفوظ رکھ کر اور CPU اوور ہیڈ کو کم کر کے بہتر بناتی ہے۔

    ### NITROS کنفیگریشن

    ```yaml
    # nitros_config.yaml
    /**:
      ros__parameters:
        # مخصوص ٹاپکس کے لیے NITROS کو فعال کریں
        use_nitros: true
        nitros_subscribers:
          - topic_name: "/camera/left/image_rect_color"
            type: "nitros_image"
            qos:
              history: "keep_last"
              depth: 1
              reliability: "reliable"
              durability: "volatile"

        nitros_publishers:
          - topic_name: "/visual_slam/tracking/feature"
            type: "nitros_feature_array"
            qos:
              history: "keep_last"
              depth: 1
              reliability: "reliable"
              durability: "volatile"
    ```

    ## Isaac ROS کے ساتھ کارکردگی کی اصلاح

    ### GPU میموری کا انتظام

    ```python
    import rclpy
    from rclpy.node import Node
    import pycuda.driver as cuda
    import pycuda.autoinit

    class IsaacROSMemoryManager(Node):
        def __init__(self):
            super().__init__('isaac_ros_memory_manager')

            # GPU میموری کے استعمال کی نگرانی کریں
            self.timer = self.create_timer(1.0, self.monitor_gpu_memory)

        def monitor_gpu_memory(self):
            """GPU میموری کے استعمال کی نگرانی کریں اور انتباہات لاگ کریں"""
            try:
                # GPU میموری کی معلومات حاصل کریں (آسان - عملی طور پر، pynvml استعمال کریں)
                free_mem, total_mem = cuda.mem_get_info()
                used_mem = total_mem - free_mem
                usage_percent = (used_mem / total_mem) * 100

                if usage_percent > 90:
                    self.get_logger().warn(
                        f'GPU memory usage is high: {usage_percent:.1f}%'
                    )
                elif usage_percent > 75:
                    self.get_logger().info(
                        f'GPU memory usage: {usage_percent:.1f}%'
                    )

            except Exception as e:
                self.get_logger().error(f'GPU memory monitoring error: {e}')
    ```

    ### پائپ لائن کی اصلاح

    ہیومنائیڈ روبوٹس کے لیے، پرسیپشن پائپ لائن کو بہتر بنانا بہت ضروری ہے:

    ```python
    class OptimizedPerceptionPipeline(Node):
        def __init__(self):
            super().__init__('optimized_perception_pipeline')

            # بہتر پروسیسنگ کے لیے Isaac ROS امیج پائپ لائن کا استعمال کریں
            self.image_sub = self.create_subscription(
                Image,
                '/camera/image_raw',
                self.optimized_image_callback,
                5  # کم تاخیر کے لیے قطار کا سائز کم کریں
            )

            # روبوٹ کی حالت کی بنیاد پر پروسیسنگ کو تھروٹل کریں
            self.processing_enabled = True
            self.processing_rate = 10.0  # Hz
            self.last_process_time = 0.0

        def optimized_image_callback(self, msg):
            """تھروٹلنگ کے ساتھ آپٹمائزڈ امیج پروسیسنگ"""
            current_time = self.get_clock().now().nanoseconds / 1e9

            if (current_time - self.last_process_time) >= (1.0 / self.processing_rate):
                if self.processing_enabled:
                    self.process_image_optimized(msg)
                self.last_process_time = current_time

        def process_image_optimized(self, image_msg):
            """Isaac ROS کا استعمال کرتے ہوئے آپٹمائزڈ امیج پروسیسنگ"""
            # یہ عملی طور پر Isaac ROS نوڈز کے ساتھ انٹرفیس کرے گا
            pass
    ```

    ## نیویگیشن سسٹمز کے ساتھ انٹیگریشن

    ### Nav2 انٹیگریشن

    Isaac ROS جدید نیویگیشن کے لیے Nav2 کے ساتھ بغیر کسی رکاوٹ کے کام کرتا ہے:

    ```python
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionClient

    class IsaacROSNav2Integrator(Node):
        def __init__(self):
            super().__init__('isaac_ros_nav2_integrator')

            # Nav2 کے لیے ایکشن کلائنٹ
            self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

            # Isaac ROS پوز کے تخمینے کو سبسکرائب کریں
            self.pose_sub = self.create_subscription(
                PoseStamped,
                '/visual_slam/pose',
                self.pose_callback,
                10
            )

            # نیویگیشن فیصلوں کے لیے ٹائمر
            self.nav_timer = self.create_timer(5.0, self.make_navigation_decision)

        def pose_callback(self, msg):
            """Isaac ROS سے موجودہ پوز کو اپ ڈیٹ کریں"""
            self.current_pose = msg

        def make_navigation_decision(self):
            """موجودہ پوز اور اہداف کی بنیاد پر نیویگیشن کے فیصلے کریں"""
            # مثال: پہلے سے طے شدہ ہدف پر جائیں
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.pose.position.x = 5.0
            goal_pose.pose.position.y = 5.0
            goal_pose.pose.orientation.w = 1.0

            self.send_navigation_goal(goal_pose)

        def send_navigation_goal(self, pose):
            """Nav2 کو نیویگیشن گول بھیجیں"""
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = pose

            self.nav_client.wait_for_server()
            future = self.nav_client.send_goal_async(goal_msg)
            future.add_done_callback(self.navigation_result_callback)

        def navigation_result_callback(self, future):
            """نیویگیشن نتیجہ کو ہینڈل کریں"""
            result = future.result()
            if result:
                self.get_logger().info('Navigation completed successfully')
            else:
                self.get_logger().error('Navigation failed')

def main(args=None):
    rclpy.init(args=args)
    node = IsaacROSNav2Integrator()

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

    ## Isaac ROS کے لیے بہترین طریقے

    ### 1. ہارڈویئر کا استعمال

    *   GPU کے استعمال اور میموری کے استعمال کی نگرانی کریں۔
    *   نیورل نیٹ ورکس کے لیے مناسب بیچ سائز استعمال کریں۔
    *   پیچیدہ ایپلی کیشنز کے لیے ملٹی GPU سیٹ اپ پر غور کریں۔

    ### 2. ڈیٹا پائپ لائن کی اصلاح

    *   تیز رفتار ڈیٹا کی منتقلی کے لیے NITROS کا استعمال کریں۔
    *   ریئل ٹائم کارکردگی کے لیے مناسب QoS سیٹنگز کو نافذ کریں۔
    *   CPU اور GPU کے درمیان ڈیٹا کاپی کرنے کو کم سے کم کریں۔

    ### 3. الگورتھم کا انتخاب

    *   GPU ایکسلریشن کے لیے موزوں الگورتھم منتخب کریں۔
    *   درستگی اور رفتار کے درمیان ٹریڈ آف پر غور کریں۔
    *   CPU پر مبنی نفاذ کے خلاف نتائج کی توثیق کریں۔

    ### 4. غلطی کو سنبھالنا (Error Handling)

    *   GPU کی ناکامیوں کے لیے فال بیک میکانزم نافذ کریں۔
    *   CUDA کی غلطیوں کی نگرانی کریں اور انہیں خوبصورتی سے ہینڈل کریں۔
    *   اصلاح کے لیے کارکردگی کے میٹرکس کو لاگ کریں۔

    ## عام مسائل کا حل (Troubleshooting)

    ### GPU میموری کے مسائل
    *   نیورل نیٹ ورکس کے لیے بیچ کا سائز کم کریں۔
    *   کم ریزولیوشن ان پٹس استعمال کریں۔
    *   میموری پولنگ کو نافذ کریں۔

    ### کارکردگی کے مسائل
    *   GPU کے استعمال اور درجہ حرارت کو چیک کریں۔
    *   CUDA کی تنصیب اور ڈرائیور کی مطابقت کی تصدیق کریں۔
    *   CPU اور GPU کے درمیان ڈیٹا کی منتقلی کو بہتر بنائیں۔

    ### انٹیگریشن کے مسائل
    *   ROS 2 پیغام کی قسم کی مطابقت کی تصدیق کریں۔
    *   نوڈز میں فریم ID کی مستقل مزاجی کو چیک کریں۔
    *   وقت اور ہم آہنگی (synchronization) کی توثیق کریں۔

    ## جدید خصوصیات

    ### کسٹم Isaac ROS ایکسٹینشنز

    خصوصی ہیومنائیڈ ایپلی کیشنز کے لیے، آپ کسٹم Isaac ROS ایکسٹینشن بنا سکتے ہیں:

    ```cpp
    // مثال کسٹم Isaac ROS جزو
    #include <rclcpp/rclcpp.hpp>
    #include <isaac_ros_nitros/nitros_node.hpp>
    #include <sensor_msgs/msg/image.hpp>

    class HumanoidPerceptionNode : public rclcpp::Node
    {
    public:
      explicit HumanoidPerceptionNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : rclcpp::Node("humanoid_perception", options)
      {
        // Isaac ROS Nitros پبلشر/سبسکرائبر کو شروع کریں
        // ہیومنائیڈ کے لیے مخصوص پروسیسنگ کے لیے کسٹم ڈیٹا کی اقسام کے ساتھ
      }

    private:
      // کسٹم پروسیسنگ کے طریقے
    };
    ```

    ## ہینڈس آن مشق

    ایک مکمل Isaac ROS پرسیپشن پائپ لائن بنائیں جس میں شامل ہوں:

    1.  لوکلائزیشن اور میپنگ کے لیے Isaac ROS Visual SLAM نوڈ۔
    2.  آبجیکٹ ڈیٹیکشن کے لیے Isaac ROS Stereo DNN۔
    3.  3D پرسیپشن کے لیے Isaac ROS Point Cloud پروسیسنگ۔
    4.  بہتر ڈیٹا فلو کے لیے NITROS کنفیگریشن۔
    5.  نیویگیشن کے لیے Nav2 کے ساتھ انٹیگریشن۔
    6.  کارکردگی کی نگرانی اور اصلاح۔

    یہ مشق آپ کو ہیومنائیڈ روبوٹس کے لیے Isaac ROS ہارڈویئر ایکسلریٹڈ پرسیپشن اور نیویگیشن کی صلاحیتوں کے ساتھ عملی تجربہ فراہم کرے گی۔
  </div>
</BilingualChapter>