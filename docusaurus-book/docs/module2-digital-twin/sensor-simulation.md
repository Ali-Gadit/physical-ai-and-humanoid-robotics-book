---
id: sensor-simulation
title: "Sensor Simulation in Gazebo"
sidebar_position: 5
---

import BilingualChapter from '@site/src/components/BilingualChapter';

<BilingualChapter>
  <div className="english">
    # Sensor Simulation in Gazebo

    ## Introduction

    Sensor simulation is critical for developing robust Physical AI systems. In Gazebo, you can simulate various types of sensors including cameras, LiDAR, IMUs, force/torque sensors, and more. For humanoid robots, accurate sensor simulation enables the development of perception algorithms, navigation systems, and human-robot interaction capabilities without requiring expensive physical hardware.

    This section covers the setup and configuration of various sensors in Gazebo, including realistic noise models and integration with ROS 2.

    ## Types of Sensors in Gazebo

    ### 1. Camera Sensors

    Camera sensors simulate RGB, depth, and stereo cameras. They're essential for computer vision applications in humanoid robotics.

    #### Basic Camera Configuration

    ```xml
    <gazebo reference="camera_link">
      <sensor name="camera" type="camera">
        <update_rate>30.0</update_rate>
        <camera name="head">
          <horizontal_fov>1.089</horizontal_fov>  <!-- 62.4 degrees -->
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
            <stddev>0.007</stddev>
          </noise>
        </camera>
        <always_on>true</always_on>
        <visualize>true</visualize>
      </sensor>
    </gazebo>
    ```

    #### Depth Camera Configuration

    ```xml
    <gazebo reference="depth_camera_link">
      <sensor name="depth_camera" type="depth">
        <update_rate>30.0</update_rate>
        <camera name="depth_head">
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
        </camera>
        <always_on>true</always_on>
        <visualize>true</visualize>
      </sensor>
    </gazebo>
    ```

    ### 2. LiDAR Sensors

    LiDAR sensors provide 2D or 3D distance measurements, crucial for navigation and obstacle detection.

    #### 2D LiDAR Configuration

    ```xml
    <gazebo reference="laser_link">
      <sensor name="laser" type="ray">
        <update_rate>10</update_rate>
        <ray>
          <scan>
            <horizontal>
              <samples>720</samples>
              <resolution>1</resolution>
              <min_angle>-1.570796</min_angle>  <!-- -90 degrees -->
              <max_angle>1.570796</max_angle>    <!-- 90 degrees -->
            </horizontal>
          </scan>
          <range>
            <min>0.10</min>
            <max>30.0</max>
            <resolution>0.01</resolution>
          </range>
        </ray>
        <always_on>true</always_on>
        <visualize>true</visualize>
      </sensor>
    </gazebo>
    ```

    #### 3D LiDAR Configuration (Velodyne-style)

    ```xml
    <gazebo reference="velodyne_link">
      <sensor name="velodyne" type="ray">
        <update_rate>10</update_rate>
        <ray>
          <scan>
            <horizontal>
              <samples>1800</samples>
              <resolution>1</resolution>
              <min_angle>-3.14159</min_angle>
              <max_angle>3.14159</max_angle>
            </horizontal>
            <vertical>
              <samples>16</samples>
              <resolution>1</resolution>
              <min_angle>-0.261799</min_angle>  <!-- -15 degrees -->
              <max_angle>0.261799</max_angle>    <!-- 15 degrees -->
            </vertical>
          </scan>
          <range>
            <min>0.1</min>
            <max>100.0</max>
            <resolution>0.01</resolution>
          </range>
        </ray>
        <always_on>true</always_on>
        <visualize>true</visualize>
      </sensor>
    </gazebo>
    ```

    ### 3. IMU Sensors

    IMU (Inertial Measurement Unit) sensors provide orientation, angular velocity, and linear acceleration data, essential for balance and navigation in humanoid robots.

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
                <stddev>0.0017</stddev>  <!-- ~0.1 deg/s (1-sigma) -->
                <bias_mean>0.0004</bias_mean>
                <bias_stddev>0.0000008</bias_stddev>
              </noise>
            </x>
            <y>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.0017</stddev>
                <bias_mean>0.0004</bias_mean>
                <bias_stddev>0.0000008</bias_stddev>
              </noise>
            </y>
            <z>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.0017</stddev>
                <bias_mean>0.0004</bias_mean>
                <bias_stddev>0.0000008</bias_stddev>
              </noise>
            </z>
          </angular_velocity>
          <linear_acceleration>
            <x>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.017</stddev>  <!-- 1-sigma: 0.017 m/s^2 -->
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.0017</bias_stddev>
              </noise>
            </x>
            <y>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.017</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.0017</bias_stddev>
              </noise>
            </y>
            <z>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.017</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.0017</bias_stddev>
              </noise>
            </z>
          </linear_acceleration>
        </imu>
      </sensor>
    </gazebo>
    ```

    ### 4. Force/Torque Sensors

    Force/torque sensors are important for manipulation tasks and contact detection.

    ```xml
    <gazebo reference="ft_sensor_joint">
      <sensor name="ft_sensor" type="force_torque">
        <always_on>true</always_on>
        <update_rate>100</update_rate>
        <force_torque>
          <frame>sensor</frame>
          <measure_direction>child_to_parent</measure_direction>
        </force_torque>
      </sensor>
    </gazebo>
    ```

    ## Noise Models and Realism

    ### Adding Realistic Noise

    Real sensors have noise characteristics that should be simulated for realistic testing:

    ```xml
    <sensor name="realistic_camera" type="camera">
      <camera name="head">
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.05</stddev>  <!-- 5% noise level -->
        </noise>
      </camera>
    </sensor>
    ```

    ### Custom Noise Models

    For more sophisticated noise modeling:

    ```xml
    <sensor name="advanced_camera" type="camera">
      <camera name="head">
        <image>
          <width>640</width>
          <height>480</height>
        </image>
        <noise>
          <type>gaussian_quantized</type>
          <mean>0.0</mean>
          <stddev>0.01</stddev>
          <quantization>0.01</quantization>  <!-- Quantization level -->
        </noise>
      </camera>
    </sensor>
    ```

    ## ROS 2 Sensor Integration

    ### Sensor Message Types

    Gazebo sensors automatically publish to standard ROS 2 topics:

    - **Cameras**: `sensor_msgs/msg/Image` and `sensor_msgs/msg/CameraInfo`
    - **LiDAR**: `sensor_msgs/msg/LaserScan` or `sensor_msgs/msg/PointCloud2`
    - **IMU**: `sensor_msgs/msg/Imu`
    - **Force/Torque**: `geometry_msgs/msg/WrenchStamped`

    ### Custom Sensor Processing

    You can create custom sensor processing nodes:

    ```python
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, Imu, LaserScan
    from cv_bridge import CvBridge
    import numpy as np

    class SensorProcessor(Node):
        def __init__(self):
            super().__init__('sensor_processor')

            # Initialize CV bridge
            self.cv_bridge = CvBridge()

            # Subscribe to various sensors
            self.camera_sub = self.create_subscription(
                Image, '/camera/image_raw', self.camera_callback, 10
            )
            self.imu_sub = self.create_subscription(
                Imu, '/imu/data', self.imu_callback, 10
            )
            self.lidar_sub = self.create_subscription(
                LaserScan, '/scan', self.lidar_callback, 10
            )

            # Publishers for processed data
            self.processed_pub = self.create_publisher(
                Image, '/processed_image', 10
            )

        def camera_callback(self, msg):
            """Process camera image"""
            try:
                # Convert ROS image to OpenCV format
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

                # Apply processing (e.g., edge detection)
                processed_image = cv2.Canny(cv_image, 50, 150)

                # Convert back to ROS format and publish
                processed_msg = self.cv_bridge.cv2_to_imgmsg(processed_image, encoding='mono8')
                self.processed_pub.publish(processed_msg)

            except Exception as e:
                self.get_logger().error(f'Camera processing error: {e}')

        def imu_callback(self, msg):
            """Process IMU data"""
            # Extract orientation, angular velocity, linear acceleration
            orientation = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
            angular_velocity = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
            linear_acceleration = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]

            # Process data as needed
            self.get_logger().info(f'Orientation: {orientation}')

        def lidar_callback(self, msg):
            """Process LiDAR data"""
            # Access range data
            ranges = np.array(msg.ranges)

            # Filter out invalid readings
            valid_ranges = ranges[np.isfinite(ranges)]

            # Process for navigation, obstacle detection, etc.
            if len(valid_ranges) > 0:
                min_distance = np.min(valid_ranges)
                self.get_logger().info(f'Min obstacle distance: {min_distance:.2f}m')

    def main(args=None):
        rclpy.init(args=args)
        processor = SensorProcessor()

        try:
            rclpy.spin(processor)
        except KeyboardInterrupt:
            pass
        finally:
            processor.destroy_node()
            rclpy.shutdown()

    if __name__ == '__main__':
        main()
    ```

    ## Multi-Sensor Fusion

    ### Sensor Placement Strategy

    For humanoid robots, strategic sensor placement is crucial:

    ```xml
    <!-- Head sensors for perception -->
    <gazebo reference="head_camera_link">
      <sensor name="head_camera" type="camera">...</sensor>
    </gazebo>

    <!-- Torso IMU for balance -->
    <gazebo reference="torso_imu_link">
      <sensor name="torso_imu" type="imu">...</sensor>
    </gazebo>

    <!-- Foot force sensors for balance -->
    <gazebo reference="left_foot_sensor_link">
      <sensor name="left_foot_ft" type="force_torque">...</sensor>
    </gazebo>

    <!-- Arm sensors for manipulation -->
    <gazebo reference="right_hand_camera_link">
      <sensor name="right_hand_camera" type="camera">...</sensor>
    </gazebo>
    ```

    ### Sensor Fusion Node

    Create a fusion node that combines multiple sensor inputs:

    ```python
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Imu, LaserScan
    from geometry_msgs.msg import PoseStamped
    from tf2_ros import TransformListener, Buffer
    import numpy as np

    class SensorFusion(Node):
        def __init__(self):
            super().__init__('sensor_fusion')

            # Initialize TF buffer for coordinate transformations
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

            # Sensor subscribers
            self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
            self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)

            # State estimation publisher
            self.pose_pub = self.create_publisher(PoseStamped, '/estimated_pose', 10)

            # Robot state
            self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion
            self.position = np.array([0.0, 0.0, 0.0])  # Position
            self.last_update_time = self.get_clock().now()

            # Timer for state updates
            self.timer = self.create_timer(0.01, self.update_state)  # 100Hz

        def imu_callback(self, msg):
            """Update orientation from IMU"""
            self.orientation = np.array([
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w
            ])

        def lidar_callback(self, msg):
            """Process LiDAR for position updates (simplified)"""
            # In practice, use more sophisticated localization
            # This is a simplified example
            ranges = np.array(msg.ranges)
            valid_ranges = ranges[np.isfinite(ranges)]

            if len(valid_ranges) > 0:
                # Simple odometry update based on environment
                # In practice, use particle filters, EKF, etc.
                pass

        def update_state(self):
            """Publish fused state estimate"""
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'

            msg.pose.position.x = float(self.position[0])
            msg.pose.position.y = float(self.position[1])
            msg.pose.position.z = float(self.position[2])

            msg.pose.orientation.x = float(self.orientation[0])
            msg.pose.orientation.y = float(self.orientation[1])
            msg.pose.orientation.z = float(self.orientation[2])
            msg.pose.orientation.w = float(self.orientation[3])

            self.pose_pub.publish(msg)

    def main(args=None):
        rclpy.init(args=args)
        fusion_node = SensorFusion()

        try:
            rclpy.spin(fusion_node)
        except KeyboardInterrupt:
            pass
        finally:
            fusion_node.destroy_node()
            rclpy.shutdown()

    if __name__ == '__main__':
        main()
    ```

    ## Calibration and Validation

    ### Sensor Calibration

    Simulated sensors should match real sensor characteristics:

    ```xml
    <sensor name="calibrated_camera" type="camera">
      <camera name="head">
        <horizontal_fov>1.089</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <!-- Calibration parameters matching real camera -->
        <distortion>
          <k1>0.1</k1>
          <k2>-0.2</k2>
          <k3>0.05</k3>
          <p1>0.001</p1>
          <p2>0.002</p2>
        </distortion>
      </camera>
    </sensor>
    ```

    ### Validation Techniques

    1. **Compare with Real Sensors**: Validate simulation output against real hardware
    2. **Synthetic Data Generation**: Use simulation to generate labeled training data
    3. **Cross-Validation**: Test algorithms in both simulation and reality

    ## Performance Considerations

    ### Optimizing Sensor Simulation

    1. **Reduce Update Rates**: Lower update rates for sensors that don't need high frequency
    2. **Simplify Noise Models**: Use simpler noise models during development
    3. **Selective Visualization**: Only visualize sensors when debugging

    ### Multi-Threading Sensors

    Gazebo can run sensors in separate threads for better performance:

    ```xml
    <sensor name="high_freq_sensor" type="camera">
      <update_rate>120</update_rate>
      <!-- Configure for separate thread if needed -->
      <always_on>true</always_on>
      <visualize>false</visualize>  <!-- Disable visualization for performance -->
    </sensor>
    ```

    ## Troubleshooting Common Issues

    ### Sensor Data Quality
    - **Check coordinate frames**: Ensure sensor data is in correct frame
    - **Verify noise parameters**: Adjust noise levels to match real sensors
    - **Validate update rates**: Ensure appropriate for sensor type

    ### Performance Issues
    - **Reduce resolution**: Lower image resolution or LiDAR samples
    - **Limit visualization**: Disable sensor visualization when not needed
    - **Optimize world complexity**: Simplify environment for faster sensor simulation

    ### Integration Problems
    - **Topic names**: Verify ROS topic names match expectations
    - **Message types**: Ensure correct message types are being published
    - **Timing issues**: Check for message synchronization problems

    ## Advanced Sensor Configurations

    ### Custom Sensor Plugins

    For specialized sensors, create custom plugins:

    ```cpp
    #include <gazebo/gazebo.hh>
    #include <gazebo/sensors/Sensor.hh>
    #include <gazebo/sensors/SensorTypes.hh>

    class CustomSensor : public gazebo::Sensor
    {
    public:
      CustomSensor() : Sensor() {}
      virtual ~CustomSensor() {}

    protected:
      virtual bool UpdateImpl(const bool _force) override
      {
        // Custom sensor logic here
        return true;
      }

      virtual void Fini() override
      {
        Sensor::Fini();
      }
    };

    // Register the sensor
    GZ_REGISTER_SENSOR("custom_sensor", CustomSensor)
    ```

    ### Dynamic Sensor Configuration

    Sensors can be reconfigured at runtime:

    ```python
    import rclpy
    from rclpy.node import Node
    from dynamic_reconfigure.server import Server
    from my_robot_msgs.cfg import SensorConfig

    class DynamicSensorConfig(Node):
        def __init__(self):
            super().__init__('dynamic_sensor_config')

            # Create dynamic reconfigure server
            self.srv = Server(SensorConfig, self.config_callback)

        def config_callback(self, config, level):
            self.get_logger().info(f"Reconfiguring sensors: {config}")

            # Apply new configuration
            # This would typically send commands to Gazebo
            return config

    def main(args=None):
        rclpy.init(args=args)
        config_node = DynamicSensorConfig()
        rclpy.spin(config_node)
    ```

    ## Hands-on Exercise

    Create a complete sensor simulation setup that includes:

    1. A humanoid robot with multiple sensor types (camera, IMU, LiDAR)
    2. Proper noise models matching real sensor characteristics
    3. A sensor processing node that fuses multiple sensor inputs
    4. ROS 2 integration for publishing sensor data
    5. Validation of sensor outputs against expected values

    This exercise will give you hands-on experience with setting up realistic sensor simulation for humanoid robots and understanding how to process and fuse multiple sensor inputs for Physical AI applications.
  </div>
  <div className="urdu">
    # سینسرز کی سیمولیشن

    ## Lidar شامل کرنا

    Lidar کی نقل کرنے کے لیے، ہم اپنے URDF میں ایک سینسر پلگ ان شامل کرتے ہیں۔

    ```xml
    <gazebo reference="lidar_link">
      <sensor name="lidar" type="gpu_lidar">
        <update_rate>10</update_rate>
        <ray>
          <scan>
            <horizontal>
              <samples>720</samples>
              <min_angle>-3.14</min_angle>
              <max_angle>3.14</max_angle>
            </horizontal>
          </scan>
        </ray>
      </sensor>
    </gazebo>
    ```

    ## کیمرہ شامل کرنا

    Lidar کی طرح، ایک کیمرہ بھی سینسر ٹیگ کے ذریعے شامل کیا جاتا ہے۔

    *   **Topic**: `/camera/image_raw`
    *   **Resolution**: 640x480 (معیاری)
  </div>
</BilingualChapter>
