---
id: isaac-sim
title: "NVIDIA Isaac Sim - Photorealistic Simulation and Synthetic Data Generation"
sidebar_position: 2
---

import BilingualChapter from '@site/src/components/BilingualChapter';

<BilingualChapter>
  <div className="english">
    # NVIDIA Isaac Sim - Photorealistic Simulation and Synthetic Data Generation

    ## Introduction

    NVIDIA Isaac Sim is a comprehensive robotics simulation application built on NVIDIA Omniverse. It provides photorealistic rendering, accurate physics simulation, and synthetic data generation capabilities that are essential for developing advanced AI systems for humanoid robots. Isaac Sim enables the creation of diverse, realistic training environments that can significantly accelerate the development of perception, navigation, and manipulation algorithms.

    The platform combines the power of NVIDIA's RTX graphics technology with advanced simulation capabilities to create virtual environments that closely match real-world conditions, making Sim-to-Real transfer more effective.

    ## Installing Isaac Sim

    ### System Requirements

    Isaac Sim has demanding hardware requirements:
    - **GPU**: NVIDIA RTX 4070 Ti (12GB VRAM) or higher (RTX 3090/4090 recommended)
    - **CPU**: Intel Core i7 (13th Gen+) or AMD Ryzen 9
    - **RAM**: 64GB DDR5 (32GB minimum)
    - **OS**: Ubuntu 22.04 LTS (recommended)

    ### Installation Process

    1. **Install NVIDIA Drivers**:
       ```bash
       sudo apt update
       sudo apt install nvidia-driver-535
       sudo reboot
       ```

    2. **Install Isaac Sim**:
       Isaac Sim can be installed via NVIDIA Omniverse Launcher or Docker:

       **Option 1: Omniverse Launcher** (Recommended for development)
       - Download and install NVIDIA Omniverse Launcher
       - Launch Isaac Sim from the application catalog

       **Option 2: Docker** (For deployment):
       ```bash
       docker pull nvcr.io/nvidia/isaac-sim:4.0.0
       ```

    3. **Verify Installation**:
       ```bash
       # Check if Isaac Sim is accessible
       docker run --rm -it --gpus all -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" \
         --name isaac_sim_test \
         -v ${PWD}:/workspace/isaac-sim/shared_folder \
         -v ~/.Xauthority:/root/.Xauthority \
         --network=host \
         --shm-size=10.0g \
         nvcr.io/nvidia/isaac-sim:4.0.0
       ```

    ## Isaac Sim Architecture

    ### Core Components

    Isaac Sim consists of several key components:

    1. **Omniverse Nucleus**: Central server for asset management and collaboration
    2. **Kit Framework**: Extensible application framework
    3. **PhysX Engine**: NVIDIA's physics simulation engine
    4. **RTX Renderer**: Photorealistic rendering pipeline
    5. **ROS 2 Bridge**: Integration with ROS 2 ecosystem
    6. **Synthetic Data Tools**: Data generation and annotation capabilities

    ### USD (Universal Scene Description)

    Isaac Sim uses USD as its core scene representation format:
    - **Scalable**: Handles complex scenes with millions of objects
    - **Collaborative**: Multiple users can work on the same scene
    - **Extensible**: Custom schemas and plugins can be added
    - **Interoperable**: Works with other 3D tools and pipelines

    ## Creating Photorealistic Environments

    ### Environment Setup

    Isaac Sim provides several built-in environments and allows for custom environment creation:

    ```python
    # Example Python script to create a simple environment
    import omni
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.prims import create_prim
    from omni.isaac.core.utils.nucleus import get_assets_root_path

    # Initialize the world
    world = World(stage_units_in_meters=1.0)

    # Add a ground plane
    create_prim(
        prim_path="/World/ground_plane",
        prim_type="Plane",
        position=[0, 0, 0],
        scale=[10, 10, 1]
    )

    # Add lighting
    create_prim(
        prim_path="/World/Room/Light",
        prim_type="DistantLight",
        position=[0, 0, 10],
        attributes={"color": [0.8, 0.8, 0.8], "intensity": 3000}
    )

    # Add a simple humanoid robot
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        print("Could not find Isaac Sim assets. Please check your installation.")
    else:
        add_reference_to_stage(
            usd_path=assets_root_path + "/Isaac/Robots/Franka/franka.usd",
            prim_path="/World/Robot"
        )
    ```

    ### Material and Lighting Configuration

    For photorealistic rendering, proper material and lighting setup is crucial:

    ```python
    from omni.isaac.core.utils.prims import get_prim_at_path
    from pxr import UsdShade, Gf

    # Create a realistic material
    def create_realistic_material(prim_path, base_color, roughness=0.5, metallic=0.0):
        # Create material prim
        material = UsdShade.Material.Define(world.stage, prim_path)

        # Create shader
        shader = UsdShade.Shader.Define(world.stage, prim_path + "/Shader")
        shader.CreateIdAttr("OmniPBR")

        # Set material properties
        shader.CreateInput("diffuse_color", Sdf.ValueTypeNames.Color3f).Set(base_color)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)

        # Connect shader to material
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "out")

        return material

    # Apply realistic materials to objects
    realistic_material = create_realistic_material(
        "/World/Materials/RealisticMaterial",
        base_color=Gf.Vec3f(0.8, 0.6, 0.2),
        roughness=0.3,
        metallic=0.1
    )
    ```

    ## Synthetic Data Generation

    ### Camera Setup for Data Collection

    Isaac Sim provides advanced camera systems for synthetic data generation:

    ```python
    from omni.isaac.sensor import Camera
    import numpy as np

    # Create a camera for data collection
    camera = Camera(
        prim_path="/World/Robot/HeadCamera",
        position=np.array([0.0, 0.0, 0.0]),
        frequency=30,
        resolution=(640, 480)
    )

    # Enable various sensor outputs
    camera.add_ground_truth_to_frame()
    camera.add_distance_to_image_prim_to_frame()
    camera.add_world_normals_to_frame()
    ```

    ### Data Annotation Tools

    Isaac Sim includes powerful tools for generating annotated training data:

    ```python
    # Semantic segmentation
    def setup_semantic_segmentation(camera):
        """Setup semantic segmentation for the camera"""
        from omni.isaac.synthetic_utils import SyntheticDataHelper

        sd_helper = SyntheticDataHelper()
        sd_helper.initialize(camera)

        # Generate semantic segmentation
        semantic_data = sd_helper.get_semantic_segmentation()
        return semantic_data

    # Bounding box annotations
    def generate_bounding_boxes(objects):
        """Generate 2D bounding box annotations for objects in the scene"""
        bounding_boxes = []

        for obj in objects:
            # Get 2D bounding box in camera coordinates
            bbox_2d = camera.get_bounding_box_2d(obj)
            bounding_boxes.append({
                'object_name': obj.name,
                'bbox': bbox_2d,
                'class_id': obj.class_id
            })

        return bounding_boxes
    ```

    ### Domain Randomization

    Domain randomization helps make models more robust to real-world variations:

    ```python
    import random

    class DomainRandomizer:
        def __init__(self):
            self.lighting_conditions = [
                {'intensity': 1000, 'color': [0.8, 0.8, 0.6]},  # Warm indoor
                {'intensity': 5000, 'color': [1.0, 1.0, 1.0]},  # Bright outdoor
                {'intensity': 2000, 'color': [0.7, 0.7, 0.9]}   # Cool indoor
            ]

            self.material_properties = [
                {'roughness': 0.1, 'metallic': 0.0},  # Smooth plastic
                {'roughness': 0.7, 'metallic': 0.2},  # Rough metal
                {'roughness': 0.4, 'metallic': 0.0}   # Matte surface
            ]

        def randomize_environment(self):
            """Randomize lighting and materials in the environment"""
            # Randomize lighting
            light_config = random.choice(self.lighting_conditions)
            # Apply to light prim

            # Randomize materials
            for material in self.material_properties:
                # Apply random material properties
                pass

    # Use domain randomization during data generation
    randomizer = DomainRandomizer()
    for episode in range(1000):  # Generate 1000 different environments
        randomizer.randomize_environment()
        # Collect data from this randomized environment
    ```

    ## Integration with ROS 2

    ### ROS Bridge Configuration

    Isaac Sim provides comprehensive ROS 2 integration:

    ```python
    # Example ROS 2 node for Isaac Sim integration
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo, Imu, LaserScan
    from geometry_msgs.msg import Twist, PoseStamped
    from nav_msgs.msg import Odometry
    import numpy as np

    class IsaacSimROSBridge(Node):
        def __init__(self):
            super().__init__('isaac_sim_ros_bridge')

            # Publishers for simulated sensors
            self.image_pub = self.create_publisher(Image, '/camera/rgb/image_raw', 10)
            self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
            self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
            self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
            self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

            # Subscribers for robot commands
            self.cmd_vel_sub = self.create_subscription(
                Twist, '/cmd_vel', self.cmd_vel_callback, 10
            )

            # Timer for publishing sensor data
            self.timer = self.create_timer(0.033, self.publish_sensor_data)  # ~30Hz

            # Isaac Sim interfaces
            self.camera_interface = None
            self.imu_interface = None
            self.lidar_interface = None

        def cmd_vel_callback(self, msg):
            """Handle velocity commands from ROS"""
            # Process velocity command and send to simulated robot
            linear_vel = msg.linear.x
            angular_vel = msg.angular.z

            # Apply to simulated robot in Isaac Sim
            self.apply_robot_velocity(linear_vel, angular_vel)

        def publish_sensor_data(self):
            """Publish sensor data from Isaac Sim"""
            # Get data from Isaac Sim sensors
            rgb_image = self.get_camera_image()
            depth_image = self.get_depth_image()
            imu_data = self.get_imu_data()
            lidar_data = self.get_lidar_data()
            odometry_data = self.get_odometry_data()

            # Publish to ROS topics
            if rgb_image is not None:
                self.image_pub.publish(rgb_image)
            if depth_image is not None:
                self.depth_pub.publish(depth_image)
            if imu_data is not None:
                self.imu_pub.publish(imu_data)
            if lidar_data is not None:
                self.scan_pub.publish(lidar_data)
            if odometry_data is not None:
                self.odom_pub.publish(odometry_data)

        def get_camera_image(self):
            """Get RGB image from Isaac Sim camera"""
            # Implementation to get image from Isaac Sim
            pass

        def get_depth_image(self):
            """Get depth image from Isaac Sim camera"""
            # Implementation to get depth from Isaac Sim
            pass

        def get_imu_data(self):
            """Get IMU data from Isaac Sim"""
            # Implementation to get IMU data from Isaac Sim
            pass

        def get_lidar_data(self):
            """Get LiDAR data from Isaac Sim"""
            # Implementation to get LiDAR data from Isaac Sim
            pass

        def get_odometry_data(self):
            """Get odometry data from Isaac Sim"""
            # Implementation to get odometry from Isaac Sim
            pass

        def apply_robot_velocity(self, linear, angular):
            """Apply velocity command to simulated robot"""
            # Implementation to control simulated robot
            pass

def main(args=None):
    rclpy.init(args=args)
    bridge = IsaacSimROSBridge()

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
    ```

    ## Humanoid Robot Simulation in Isaac Sim

    ### Advanced Physics for Humanoid Locomotion

    Isaac Sim provides specialized physics capabilities for humanoid robots:

    ```python
    # Humanoid-specific physics configuration
    def setup_humanoid_physics(humanoid_robot):
        """Configure physics properties for stable humanoid locomotion"""

        # Set up joint properties for natural movement
        for joint in humanoid_robot.joints:
            # Configure joint limits based on human anatomy
            joint.set_position_limit(
                lower=-2.0,  # Example: -115 degrees
                upper=2.0    # Example: 115 degrees
            )

            # Set appropriate damping for natural movement
            joint.set_damping(0.1)

            # Set stiffness for controlled movement
            joint.set_stiffness(1000.0)

    # Balance control for bipedal locomotion
    class BalanceController:
        def __init__(self, robot):
            self.robot = robot
            self.com_estimator = CenterOfMassEstimator(robot)
            self.foot_sensors = [robot.left_foot_sensor, robot.right_foot_sensor]

        def compute_balance_correction(self):
            """Compute balance correction torques based on COM and foot sensors"""
            # Estimate center of mass position
            com_pos = self.com_estimator.get_com_position()
            com_vel = self.com_estimator.get_com_velocity()

            # Get foot contact information
            left_contact = self.foot_sensors[0].get_contact_force()
            right_contact = self.foot_sensors[1].get_contact_force()

            # Compute balance correction using inverted pendulum model
            balance_torques = self.compute_inverted_pendulum_control(com_pos, com_vel)

            return balance_torques
    ```

    ### Advanced Rendering Features

    Isaac Sim's RTX rendering capabilities enable highly realistic simulation:

    ```python
    # Configure advanced rendering settings
    def configure_advanced_rendering():
        """Configure RTX rendering for maximum realism"""

        # Enable ray tracing effects
        settings = {
            'rtx': {
                'ray_tracing': True,
                'global_illumination': True,
                'denoising': True,
                'motion_blur': True,
                'depth_of_field': True
            },
            'rendering': {
                'samples_per_pixel': 16,
                'max_bounces': 8,
                'light_sampling_quality': 'high'
            }
        }

        # Apply settings to renderer
        # Implementation depends on Isaac Sim API
        pass
    ```

    ## Synthetic Data Pipeline

    ### Automated Data Generation

    Creating an automated pipeline for synthetic data generation:

    ```python
    import os
    import json
    import cv2
    import numpy as np
    from datetime import datetime

    class SyntheticDataPipeline:
        def __init__(self, output_dir="synthetic_data"):
            self.output_dir = output_dir
            self.episode_count = 0
            self.scene_variations = []

            # Create output directory structure
            os.makedirs(f"{output_dir}/images", exist_ok=True)
            os.makedirs(f"{output_dir}/labels", exist_ok=True)
            os.makedirs(f"{output_dir}/metadata", exist_ok=True)

        def generate_episode(self):
            """Generate one episode of synthetic data"""
            episode_dir = f"{self.output_dir}/episode_{self.episode_count:04d}"
            os.makedirs(episode_dir, exist_ok=True)

            # Randomize environment
            self.randomize_environment()

            # Generate trajectory with robot
            trajectory = self.generate_robot_trajectory()

            # Collect data along trajectory
            data_samples = []
            for step, pose in enumerate(trajectory):
                # Move robot to pose
                self.move_robot_to_pose(pose)

                # Collect sensor data
                rgb_img = self.get_camera_image()
                depth_img = self.get_depth_image()
                segmentation = self.get_semantic_segmentation()

                # Save data
                sample_data = {
                    'timestamp': step,
                    'pose': pose,
                    'rgb_path': f"{episode_dir}/rgb_{step:04d}.png",
                    'depth_path': f"{episode_dir}/depth_{step:04d}.png",
                    'seg_path': f"{episode_dir}/seg_{step:04d}.png",
                    'annotations': self.generate_annotations(segmentation)
                }

                # Save images
                cv2.imwrite(sample_data['rgb_path'], cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(sample_data['depth_path'], depth_img)
                cv2.imwrite(sample_data['seg_path'], segmentation)

                data_samples.append(sample_data)

            # Save episode metadata
            metadata = {
                'episode_id': self.episode_count,
                'scene_config': self.get_scene_config(),
                'robot_config': self.get_robot_config(),
                'sample_count': len(data_samples),
                'timestamp': datetime.now().isoformat()
            }

            with open(f"{episode_dir}/metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            self.episode_count += 1
            return data_samples

        def generate_robot_trajectory(self):
            """Generate a random but feasible robot trajectory"""
            # Implementation for generating realistic robot movements
            pass

        def generate_annotations(self, segmentation):
            """Generate object detection and segmentation annotations"""
            # Find unique objects in segmentation
            unique_ids = np.unique(segmentation)

            annotations = []
            for obj_id in unique_ids:
                if obj_id == 0:  # Background
                    continue

                # Find bounding box for object
                y_coords, x_coords = np.where(segmentation == obj_id)
                bbox = [int(np.min(x_coords)), int(np.min(y_coords)),
                       int(np.max(x_coords)), int(np.max(y_coords))]

                annotations.append({
                    'object_id': int(obj_id),
                    'bbox': bbox,
                    'pixel_count': len(y_coords)
                })

            return annotations

        def randomize_environment(self):
            """Randomize environment properties for domain randomization"""
            # Randomize lighting
            # Randomize materials
            # Randomize object positions
            # Randomize textures
            pass

    # Usage example
    pipeline = SyntheticDataPipeline("humanoid_navigation_dataset")
    for episode in range(1000):  # Generate 1000 episodes
        samples = pipeline.generate_episode()
        print(f"Generated episode {episode} with {len(samples)} samples")
    ```

    ## Performance Optimization

    ### Multi-GPU Setup

    For large-scale synthetic data generation:

    ```python
    # Multi-GPU configuration for Isaac Sim
    def setup_multi_gpu():
        """Configure Isaac Sim for multi-GPU operation"""

        # Enable multi-GPU rendering
        multi_gpu_config = {
            'rendering': {
                'multi_gpu': True,
                'primary_gpu': 0,
                'secondary_gpus': [1, 2, 3],  # Additional GPUs
                'workload_distribution': 'automatic'
            },
            'physics': {
                'multi_gpu_physics': True,
                'gpu_count': 2
            }
        }

        # Apply configuration
        # Implementation depends on Isaac Sim API
        pass
    ```

    ### Batch Processing

    Efficient batch processing for large datasets:

    ```python
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    def batch_generate_data(batch_config):
        """Generate a batch of synthetic data"""
        pipeline = SyntheticDataPipeline(batch_config['output_dir'])

        for episode in range(batch_config['episodes_per_batch']):
            samples = pipeline.generate_episode()

        return f"Completed batch {batch_config['batch_id']} with {batch_config['episodes_per_batch']} episodes"

    def parallel_data_generation(num_batches=8, episodes_per_batch=100):
        """Generate synthetic data in parallel"""

        batch_configs = []
        for i in range(num_batches):
            config = {
                'batch_id': i,
                'episodes_per_batch': episodes_per_batch,
                'output_dir': f"synthetic_data_batch_{i:02d}"
            }
            batch_configs.append(config)

        # Use process pool for parallel execution
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            results = list(executor.map(batch_generate_data, batch_configs))

        for result in results:
            print(result)
    ```

    ## Best Practices for Isaac Sim

    ### 1. Scene Complexity Management

    - Start with simple scenes and gradually add complexity
    - Use level-of-detail (LOD) systems for complex environments
    - Implement occlusion culling for performance

    ### 2. Asset Optimization

    - Use appropriate polygon counts for real-time performance
    - Implement texture streaming for large environments
    - Use instancing for repeated objects

    ### 3. Physics Tuning

    - Start with realistic physics parameters
    - Adjust for simulation stability vs. performance
    - Validate physics behavior against real-world data

    ### 4. Data Quality Assurance

    - Implement automated quality checks for generated data
    - Use statistical validation to ensure data diversity
    - Regularly validate synthetic data against real data

    ## Troubleshooting Common Issues

    ### Performance Issues
    - Reduce scene complexity if frame rate drops
    - Check GPU memory usage and adjust accordingly
    - Use simplified collision meshes for performance

    ### Rendering Artifacts
    - Verify material properties and textures
    - Check lighting setup and exposure settings
    - Ensure proper camera calibration parameters

    ### Physics Instability
    - Adjust physics time step and solver parameters
    - Verify joint limits and constraints
    - Check mass and inertia properties

    ## Hands-on Exercise

    Create a complete Isaac Sim environment that includes:

    1. A photorealistic indoor scene with varied lighting
    2. A humanoid robot with proper physics configuration
    3. Multiple sensor types (camera, depth, IMU, LiDAR)
    4. A synthetic data generation pipeline
    5. Domain randomization for robust training data
    6. ROS 2 integration for standard message types

    This exercise will give you hands-on experience with Isaac Sim's advanced capabilities for generating synthetic data and creating photorealistic environments for Physical AI applications.
  </div>
  <div className="urdu">
    # NVIDIA Isaac Sim: فوٹو ریئلسٹک سیمولیشن اور مصنوعی ڈیٹا جنریشن

    ## تعارف

    NVIDIA Isaac Sim ایک جامع روبوٹکس سیمولیشن ایپلی کیشن ہے جو NVIDIA Omniverse پر بنائی گئی ہے۔ یہ فوٹو ریئلسٹک رینڈرنگ، درست فزکس سیمولیشن، اور مصنوعی ڈیٹا جنریشن (synthetic data generation) کی صلاحیتیں فراہم کرتی ہے جو ہیومنائیڈ روبوٹس کے لیے جدید AI سسٹمز تیار کرنے کے لیے ضروری ہیں۔ Isaac Sim متنوع، حقیقت پسندانہ تربیتی ماحول بنانے کے قابل بناتا ہے جو پرسیپشن، نیویگیشن، اور مینیپولیشن الگورتھمز کی ترقی کو نمایاں طور پر تیز کر سکتا ہے۔

    یہ پلیٹ فارم NVIDIA کی RTX گرافکس ٹیکنالوجی کی طاقت کو جدید سیمولیشن صلاحیتوں کے ساتھ جوڑتا ہے تاکہ ورچوئل ماحول تخلیق کیا جا سکے جو حقیقی دنیا کے حالات سے قریب سے میل کھاتا ہو، جس سے Sim-to-Real ٹرانسفر زیادہ موثر ہوتا ہے۔

    ## Isaac Sim کی تنصیب

    ### سسٹم کے تقاضے

    Isaac Sim کے لیے طاقتور ہارڈویئر کی ضرورت ہوتی ہے:
    *   **GPU**: NVIDIA RTX 4070 Ti (12GB VRAM) یا اس سے زیادہ (RTX 3090/4090 تجویز کردہ)
    *   **CPU**: Intel Core i7 (13th Gen+) یا AMD Ryzen 9
    *   **RAM**: 64GB DDR5 (کم از کم 32GB)
    *   **OS**: Ubuntu 22.04 LTS (تجویز کردہ)

    ### انسٹالیشن کا عمل

    1.  **NVIDIA ڈرائیورز انسٹال کریں**:
        ```bash
        sudo apt update
        sudo apt install nvidia-driver-535
        sudo reboot
        ```

    2.  **Isaac Sim انسٹال کریں**:
        Isaac Sim کو NVIDIA Omniverse Launcher یا Docker کے ذریعے انسٹال کیا جا سکتا ہے:

        **آپشن 1: Omniverse Launcher** (ڈویلپمنٹ کے لیے تجویز کردہ)
        *   NVIDIA Omniverse Launcher ڈاؤن لوڈ اور انسٹال کریں۔
        *   ایپلیکیشن کیٹلاگ سے Isaac Sim لانچ کریں۔

        **آپشن 2: Docker** (تعیناتی کے لیے):
        ```bash
        docker pull nvcr.io/nvidia/isaac-sim:4.0.0
        ```

    3.  **تنصیب کی تصدیق کریں**:
        ```bash
        # چیک کریں کہ آیا Isaac Sim قابل رسائی ہے
        docker run --rm -it --gpus all -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" \
          --name isaac_sim_test \
          -v ${PWD}:/workspace/isaac-sim/shared_folder \
          -v ~/.Xauthority:/root/.Xauthority \
          --network=host \
          --shm-size=10.0g \
          nvcr.io/nvidia/isaac-sim:4.0.0
        ```

    ## Isaac Sim آرکیٹیکچر

    ### بنیادی اجزاء

    Isaac Sim کئی اہم اجزاء پر مشتمل ہے:

    1.  **Omniverse Nucleus**: اثاثہ جات کے انتظام اور تعاون کے لیے مرکزی سرور
    2.  **Kit Framework**: ایکسٹینسیبل ایپلی کیشن فریم ورک
    3.  **PhysX Engine**: NVIDIA کا فزکس سیمولیشن انجن
    4.  **RTX Renderer**: فوٹو ریئلسٹک رینڈرنگ پائپ لائن
    5.  **ROS 2 Bridge**: ROS 2 ایکو سسٹم کے ساتھ انضمام
    6.  **Synthetic Data Tools**: ڈیٹا جنریشن اور اینوٹیشن کی صلاحیتیں

    ### USD (Universal Scene Description)

    Isaac Sim USD کو اپنے بنیادی منظر کی نمائندگی کے فارمیٹ کے طور پر استعمال کرتا ہے:
    *   **قابل توسیع**: لاکھوں اشیاء کے ساتھ پیچیدہ مناظر کو سنبھالتا ہے۔
    *   **باہمی تعاون**: متعدد صارفین ایک ہی منظر پر کام کر سکتے ہیں۔
    *   **ایکسٹینسیبل**: کسٹم اسکیموں اور پلگ انز کو شامل کیا جا سکتا ہے۔
    *   **انٹروپریبل**: دوسرے 3D ٹولز اور پائپ لائنز کے ساتھ کام کرتا ہے۔

    ## فوٹو ریئلسٹک ماحول بنانا

    ### ماحول کا سیٹ اپ

    Isaac Sim کئی بلٹ ان ماحول فراہم کرتا ہے اور کسٹم ماحول بنانے کی اجازت دیتا ہے:

    ```python
    # ایک سادہ ماحول بنانے کے لیے مثال Python اسکرپٹ
    import omni
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.prims import create_prim
    from omni.isaac.core.utils.nucleus import get_assets_root_path

    # دنیا کو شروع کریں
    world = World(stage_units_in_meters=1.0)

    # ایک زمینی ہوائی جہاز شامل کریں
    create_prim(
        prim_path="/World/ground_plane",
        prim_type="Plane",
        position=[0, 0, 0],
        scale=[10, 10, 1]
    )

    # روشنی شامل کریں
    create_prim(
        prim_path="/World/Room/Light",
        prim_type="DistantLight",
        position=[0, 0, 10],
        attributes={"color": [0.8, 0.8, 0.8], "intensity": 3000}
    )

    # ایک سادہ ہیومنائیڈ روبوٹ شامل کریں
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        print("Could not find Isaac Sim assets. Please check your installation.")
    else:
        add_reference_to_stage(
            usd_path=assets_root_path + "/Isaac/Robots/Franka/franka.usd",
            prim_path="/World/Robot"
        )
    ```

    ### میٹریل اور لائٹنگ کنفیگریشن

    فوٹو ریئلسٹک رینڈرنگ کے لیے، مناسب میٹریل اور لائٹنگ سیٹ اپ بہت اہم ہے:

    ```python
    from omni.isaac.core.utils.prims import get_prim_at_path
    from pxr import UsdShade, Gf

    # ایک حقیقت پسندانہ مواد بنائیں
    def create_realistic_material(prim_path, base_color, roughness=0.5, metallic=0.0):
        # مواد پرائم بنائیں
        material = UsdShade.Material.Define(world.stage, prim_path)

        # شیڈر بنائیں
        shader = UsdShade.Shader.Define(world.stage, prim_path + "/Shader")
        shader.CreateIdAttr("OmniPBR")

        # مواد کی خصوصیات سیٹ کریں
        shader.CreateInput("diffuse_color", Sdf.ValueTypeNames.Color3f).Set(base_color)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)

        # شیڈر کو مواد سے جوڑیں
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "out")

        return material

    # اشیاء پر حقیقت پسندانہ مواد لگائیں
    realistic_material = create_realistic_material(
        "/World/Materials/RealisticMaterial",
        base_color=Gf.Vec3f(0.8, 0.6, 0.2),
        roughness=0.3,
        metallic=0.1
    )
    ```

    ## مصنوعی ڈیٹا جنریشن (Synthetic Data Generation)

    ### ڈیٹا اکٹھا کرنے کے لیے کیمرہ سیٹ اپ

    Isaac Sim مصنوعی ڈیٹا جنریشن کے لیے جدید کیمرہ سسٹمز فراہم کرتا ہے:

    ```python
    from omni.isaac.sensor import Camera
    import numpy as np

    # ڈیٹا اکٹھا کرنے کے لیے کیمرہ بنائیں
    camera = Camera(
        prim_path="/World/Robot/HeadCamera",
        position=np.array([0.0, 0.0, 0.0]),
        frequency=30,
        resolution=(640, 480)
    )

    # مختلف سینسر آؤٹ پٹس کو فعال کریں
    camera.add_ground_truth_to_frame()
    camera.add_distance_to_image_prim_to_frame()
    camera.add_world_normals_to_frame()
    ```

    ### ڈیٹا اینوٹیشن ٹولز

    Isaac Sim میں اینوٹٹیڈ ٹریننگ ڈیٹا بنانے کے لیے طاقتور ٹولز شامل ہیں:

    ```python
    # سیمنٹک سیگمنٹیشن
    def setup_semantic_segmentation(camera):
        """کیمرے کے لیے سیمنٹک سیگمنٹیشن سیٹ اپ کریں"""
        from omni.isaac.synthetic_utils import SyntheticDataHelper

        sd_helper = SyntheticDataHelper()
        sd_helper.initialize(camera)

        # سیمنٹک سیگمنٹیشن بنائیں
        semantic_data = sd_helper.get_semantic_segmentation()
        return semantic_data

    # باؤنڈنگ باکس اینوٹیشنز
    def generate_bounding_boxes(objects):
        """منظر میں اشیاء کے لیے 2D باؤنڈنگ باکس اینوٹیشنز بنائیں"""
        bounding_boxes = []

        for obj in objects:
            # کیمرہ کوآرڈینیٹس میں 2D باؤنڈنگ باکس حاصل کریں
            bbox_2d = camera.get_bounding_box_2d(obj)
            bounding_boxes.append({
                'object_name': obj.name,
                'bbox': bbox_2d,
                'class_id': obj.class_id
            })

        return bounding_boxes
    ```

    ### ڈومین رینڈمائزیشن (Domain Randomization)

    ڈومین رینڈمائزیشن ماڈلز کو حقیقی دنیا کی مختلف حالتوں کے لیے زیادہ مضبوط بنانے میں مدد کرتی ہے:

    ```python
    import random

    class DomainRandomizer:
        def __init__(self):
            self.lighting_conditions = [
                {'intensity': 1000, 'color': [0.8, 0.8, 0.6]},  # گرم انڈور
                {'intensity': 5000, 'color': [1.0, 1.0, 1.0]},  # روشن آؤٹ ڈور
                {'intensity': 2000, 'color': [0.7, 0.7, 0.9]}   # ٹھنڈا انڈور
            ]

            self.material_properties = [
                {'roughness': 0.1, 'metallic': 0.0},  # ہموار پلاسٹک
                {'roughness': 0.7, 'metallic': 0.2},  # کھردری دھات
                {'roughness': 0.4, 'metallic': 0.0}   # دھندلا سطح
            ]

        def randomize_environment(self):
            """ماحول میں روشنی اور مواد کو تصادفی طور پر تبدیل کریں"""
            # روشنی کو تصادفی بنائیں
            light_config = random.choice(self.lighting_conditions)
            # لائٹ پرائم پر لاگو کریں

            # مواد کو تصادفی بنائیں
            for material in self.material_properties:
                # بے ترتیب مواد کی خصوصیات لاگو کریں
                pass

    # ڈیٹا جنریشن کے دوران ڈومین رینڈمائزیشن کا استعمال کریں
    randomizer = DomainRandomizer()
    for episode in range(1000):  # 1000 مختلف ماحول بنائیں
        randomizer.randomize_environment()
        # اس رینڈمائزڈ ماحول سے ڈیٹا اکٹھا کریں
    ```

    ## ROS 2 کے ساتھ انٹیگریشن

    ### ROS Bridge کنفیگریشن

    Isaac Sim جامع ROS 2 انٹیگریشن فراہم کرتا ہے:

    ```python
    # Isaac Sim انٹیگریشن کے لیے مثال ROS 2 نوڈ
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo, Imu, LaserScan
    from geometry_msgs.msg import Twist, PoseStamped
    from nav_msgs.msg import Odometry
    import numpy as np

    class IsaacSimROSBridge(Node):
        def __init__(self):
            super().__init__('isaac_sim_ros_bridge')

            # سیمولیٹڈ سینسرز کے لیے پبلشرز
            self.image_pub = self.create_publisher(Image, '/camera/rgb/image_raw', 10)
            self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
            self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
            self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
            self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

            # روبوٹ کمانڈز کے لیے سبسکرائبرز
            self.cmd_vel_sub = self.create_subscription(
                Twist, '/cmd_vel', self.cmd_vel_callback, 10
            )

            # سینسر ڈیٹا شائع کرنے کے لیے ٹائمر
            self.timer = self.create_timer(0.033, self.publish_sensor_data)  # ~30Hz

            # Isaac Sim انٹرفیس
            self.camera_interface = None
            self.imu_interface = None
            self.lidar_interface = None

        def cmd_vel_callback(self, msg):
            """ROS سے رفتار کے کمانڈز کو ہینڈل کریں"""
            # رفتار کمانڈ پر عمل کریں اور سیمولیٹڈ روبوٹ کو بھیجیں
            linear_vel = msg.linear.x
            angular_vel = msg.angular.z

            # Isaac Sim میں سیمولیٹڈ روبوٹ پر لاگو کریں
            self.apply_robot_velocity(linear_vel, angular_vel)

        def publish_sensor_data(self):
            """Isaac Sim سے سینسر ڈیٹا شائع کریں"""
            # Isaac Sim سینسرز سے ڈیٹا حاصل کریں
            rgb_image = self.get_camera_image()
            depth_image = self.get_depth_image()
            imu_data = self.get_imu_data()
            lidar_data = self.get_lidar_data()
            odometry_data = self.get_odometry_data()

            # ROS عنوانات پر شائع کریں
            if rgb_image is not None:
                self.image_pub.publish(rgb_image)
            if depth_image is not None:
                self.depth_pub.publish(depth_image)
            if imu_data is not None:
                self.imu_pub.publish(imu_data)
            if lidar_data is not None:
                self.scan_pub.publish(lidar_data)
            if odometry_data is not None:
                self.odom_pub.publish(odometry_data)

        def get_camera_image(self):
            """Isaac Sim کیمرے سے RGB تصویر حاصل کریں"""
            # Isaac Sim سے تصویر حاصل کرنے کے لیے نفاذ
            pass

        def get_depth_image(self):
            """Isaac Sim کیمرے سے گہرائی کی تصویر حاصل کریں"""
            # Isaac Sim سے گہرائی حاصل کرنے کے لیے نفاذ
            pass

        def get_imu_data(self):
            """Isaac Sim سے IMU ڈیٹا حاصل کریں"""
            # Isaac Sim سے IMU ڈیٹا حاصل کرنے کے لیے نفاذ
            pass

        def get_lidar_data(self):
            """Isaac Sim سے LiDAR ڈیٹا حاصل کریں"""
            # Isaac Sim سے LiDAR ڈیٹا حاصل کرنے کے لیے نفاذ
            pass

        def get_odometry_data(self):
            """Isaac Sim سے اوڈومیٹری ڈیٹا حاصل کریں"""
            # Isaac Sim سے اوڈومیٹری حاصل کرنے کے لیے نفاذ
            pass

        def apply_robot_velocity(self, linear, angular):
            """سیمولیٹڈ روبوٹ پر رفتار کمانڈ کا اطلاق کریں"""
            # سیمولیٹڈ روبوٹ کو کنٹرول کرنے کے لیے نفاذ
            pass

def main(args=None):
    rclpy.init(args=args)
    bridge = IsaacSimROSBridge()

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
    ```

    ## Isaac Sim میں ہیومنائیڈ روبوٹ سیمولیشن

    ### ہیومنائیڈ لوکوموشن کے لیے جدید فزکس

    Isaac Sim ہیومنائیڈ روبوٹس کے لیے فزکس کی خصوصی صلاحیتیں فراہم کرتا ہے:

    ```python
    # ہیومنائیڈ کے لیے مخصوص فزکس کنفیگریشن
    def setup_humanoid_physics(humanoid_robot):
        """مستحکم ہیومنائیڈ لوکوموشن کے لیے فزکس کی خصوصیات کو کنفیگر کریں"""

        # قدرتی حرکت کے لیے مشترکہ خصوصیات سیٹ اپ کریں
        for joint in humanoid_robot.joints:
            # انسانی اناٹومی کی بنیاد پر مشترکہ حدود کو ترتیب دیں
            joint.set_position_limit(
                lower=-2.0,  # مثال: -115 ڈگری
                upper=2.0    # مثال: 115 ڈگری
            )

            # قدرتی حرکت کے لیے مناسب ڈیمپنگ سیٹ کریں
            joint.set_damping(0.1)

            # کنٹرول شدہ حرکت کے لیے سختی (stiffness) سیٹ کریں
            joint.set_stiffness(1000.0)

    # دو ٹانگوں والے لوکوموشن کے لیے بیلنس کنٹرول
    class BalanceController:
        def __init__(self, robot):
            self.robot = robot
            self.com_estimator = CenterOfMassEstimator(robot)
            self.foot_sensors = [robot.left_foot_sensor, robot.right_foot_sensor]

        def compute_balance_correction(self):
            """COM اور فٹ سینسرز کی بنیاد پر بیلنس کریکشن ٹارک کا حساب لگائیں"""
            # سینٹر آف ماس پوزیشن کا اندازہ لگائیں
            com_pos = self.com_estimator.get_com_position()
            com_vel = self.com_estimator.get_com_velocity()

            # پیروں کے رابطے کی معلومات حاصل کریں
            left_contact = self.foot_sensors[0].get_contact_force()
            right_contact = self.foot_sensors[1].get_contact_force()

            # الٹے پینڈولم ماڈل کا استعمال کرتے ہوئے بیلنس کریکشن کا حساب لگائیں
            balance_torques = self.compute_inverted_pendulum_control(com_pos, com_vel)

            return balance_torques
    ```

    ### جدید رینڈرنگ خصوصیات

    Isaac Sim کی RTX رینڈرنگ کی صلاحیتیں انتہائی حقیقت پسندانہ سیمولیشن کو قابل بناتی ہیں:

    ```python
    # جدید رینڈرنگ سیٹنگز کنفیگر کریں
    def configure_advanced_rendering():
        """زیادہ سے زیادہ حقیقت پسندی کے لیے RTX رینڈرنگ کو کنفیگر کریں"""

        # رے ٹریسنگ اثرات کو فعال کریں
        settings = {
            'rtx': {
                'ray_tracing': True,
                'global_illumination': True,
                'denoising': True,
                'motion_blur': True,
                'depth_of_field': True
            },
            'rendering': {
                'samples_per_pixel': 16,
                'max_bounces': 8,
                'light_sampling_quality': 'high'
            }
        }

        # رینڈرر پر سیٹنگز لاگو کریں
        # نفاذ کا انحصار Isaac Sim API پر ہے
        pass
    ```

    ## مصنوعی ڈیٹا پائپ لائن (Synthetic Data Pipeline)

    ### خودکار ڈیٹا جنریشن

    مصنوعی ڈیٹا جنریشن کے لیے خودکار پائپ لائن بنانا:

    ```python
    import os
    import json
    import cv2
    import numpy as np
    from datetime import datetime

    class SyntheticDataPipeline:
        def __init__(self, output_dir="synthetic_data"):
            self.output_dir = output_dir
            self.episode_count = 0
            self.scene_variations = []

            # آؤٹ پٹ ڈائرکٹری کا ڈھانچہ بنائیں
            os.makedirs(f"{output_dir}/images", exist_ok=True)
            os.makedirs(f"{output_dir}/labels", exist_ok=True)
            os.makedirs(f"{output_dir}/metadata", exist_ok=True)

        def generate_episode(self):
            """مصنوعی ڈیٹا کا ایک ایپیسوڈ بنائیں"""
            episode_dir = f"{self.output_dir}/episode_{self.episode_count:04d}"
            os.makedirs(episode_dir, exist_ok=True)

            # ماحول کو تصادفی بنائیں
            self.randomize_environment()

            # روبوٹ کے ساتھ ٹریجیکٹری بنائیں
            trajectory = self.generate_robot_trajectory()

            # ٹریجیکٹری کے ساتھ ڈیٹا اکٹھا کریں
            data_samples = []
            for step, pose in enumerate(trajectory):
                # روبوٹ کو پوز پر منتقل کریں
                self.move_robot_to_pose(pose)

                # سینسر ڈیٹا اکٹھا کریں
                rgb_img = self.get_camera_image()
                depth_img = self.get_depth_image()
                segmentation = self.get_semantic_segmentation()

                # ڈیٹا محفوظ کریں
                sample_data = {
                    'timestamp': step,
                    'pose': pose,
                    'rgb_path': f"{episode_dir}/rgb_{step:04d}.png",
                    'depth_path': f"{episode_dir}/depth_{step:04d}.png",
                    'seg_path': f"{episode_dir}/seg_{step:04d}.png",
                    'annotations': self.generate_annotations(segmentation)
                }

                # تصاویر محفوظ کریں
                cv2.imwrite(sample_data['rgb_path'], cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(sample_data['depth_path'], depth_img)
                cv2.imwrite(sample_data['seg_path'], segmentation)

                data_samples.append(sample_data)

            # ایپیسوڈ میٹا ڈیٹا محفوظ کریں
            metadata = {
                'episode_id': self.episode_count,
                'scene_config': self.get_scene_config(),
                'robot_config': self.get_robot_config(),
                'sample_count': len(data_samples),
                'timestamp': datetime.now().isoformat()
            }

            with open(f"{episode_dir}/metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            self.episode_count += 1
            return data_samples

        def generate_robot_trajectory(self):
            """ایک بے ترتیب لیکن قابل عمل روبوٹ ٹریجیکٹری بنائیں"""
            # حقیقت پسندانہ روبوٹ کی نقل و حرکت پیدا کرنے کے لیے نفاذ
            pass

        def generate_annotations(self, segmentation):
            """آبجیکٹ ڈیٹیکشن اور سیگمنٹیشن اینوٹیشنز بنائیں"""
            # سیگمنٹیشن میں منفرد اشیاء تلاش کریں
            unique_ids = np.unique(segmentation)

            annotations = []
            for obj_id in unique_ids:
                if obj_id == 0:  # پس منظر
                    continue

                # آبجیکٹ کے لیے باؤنڈنگ باکس تلاش کریں
                y_coords, x_coords = np.where(segmentation == obj_id)
                bbox = [int(np.min(x_coords)), int(np.min(y_coords)),
                       int(np.max(x_coords)), int(np.max(y_coords))]

                annotations.append({
                    'object_id': int(obj_id),
                    'bbox': bbox,
                    'pixel_count': len(y_coords)
                })

            return annotations

        def randomize_environment(self):
            """ڈومین رینڈمائزیشن کے لیے ماحولیاتی خصوصیات کو تصادفی بنائیں"""
            # روشنی کو تصادفی بنائیں
            # مواد کو تصادفی بنائیں
            # اشیاء کی پوزیشنوں کو تصادفی بنائیں
            # ساخت کو تصادفی بنائیں
            pass

    # استعمال کی مثال
    pipeline = SyntheticDataPipeline("humanoid_navigation_dataset")
    for episode in range(1000):  # 1000 ایپیسوڈز بنائیں
        samples = pipeline.generate_episode()
        print(f"Generated episode {episode} with {len(samples)} samples")
    ```

    ## کارکردگی کی اصلاح (Performance Optimization)

    ### ملٹی GPU سیٹ اپ

    بڑے پیمانے پر مصنوعی ڈیٹا جنریشن کے لیے:

    ```python
    # Isaac Sim کے لیے ملٹی GPU کنفیگریشن
    def setup_multi_gpu():
        """ملٹی GPU آپریشن کے لیے Isaac Sim کو کنفیگر کریں"""

        # ملٹی GPU رینڈرنگ کو فعال کریں
        multi_gpu_config = {
            'rendering': {
                'multi_gpu': True,
                'primary_gpu': 0,
                'secondary_gpus': [1, 2, 3],  # اضافی GPUs
                'workload_distribution': 'automatic'
            },
            'physics': {
                'multi_gpu_physics': True,
                'gpu_count': 2
            }
        }

        # کنفیگریشن کا اطلاق کریں
        # نفاذ کا انحصار Isaac Sim API پر ہے
        pass
    ```

    ### بیچ پروسیسنگ

    بڑے ڈیٹاسیٹس کے لیے موثر بیچ پروسیسنگ:

    ```python
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    def batch_generate_data(batch_config):
        """مصنوعی ڈیٹا کا ایک بیچ بنائیں"""
        pipeline = SyntheticDataPipeline(batch_config['output_dir'])

        for episode in range(batch_config['episodes_per_batch']):
            samples = pipeline.generate_episode()

        return f"Completed batch {batch_config['batch_id']} with {batch_config['episodes_per_batch']} episodes"

    def parallel_data_generation(num_batches=8, episodes_per_batch=100):
        """متوازی طور پر مصنوعی ڈیٹا بنائیں"""

        batch_configs = []
        for i in range(num_batches):
            config = {
                'batch_id': i,
                'episodes_per_batch': episodes_per_batch,
                'output_dir': f"synthetic_data_batch_{i:02d}"
            }
            batch_configs.append(config)

        # متوازی عمل کے لیے پروسیس پول کا استعمال کریں
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            results = list(executor.map(batch_generate_data, batch_configs))

        for result in results:
            print(result)
    ```

    ## Isaac Sim کے لیے بہترین طریقے

    ### 1. منظر کی پیچیدگی کا انتظام

    *   سادہ مناظر سے شروع کریں اور آہستہ آہستہ پیچیدگی میں اضافہ کریں۔
    *   پیچیدہ ماحول کے لیے لیول آف ڈیٹیل (LOD) سسٹم استعمال کریں۔
    *   کارکردگی کے لیے occlusion culling نافذ کریں۔

    ### 2. اثاثہ (Asset) کی اصلاح

    *   ریئل ٹائم کارکردگی کے لیے مناسب پولی گون شمار استعمال کریں۔
    *   بڑے ماحول کے لیے ٹیکسچر اسٹریمنگ نافذ کریں۔
    *   بار بار آنے والی اشیاء کے لیے انسٹنسنگ (instancing) کا استعمال کریں۔

    ### 3. فزکس ٹیوننگ

    *   حقیقت پسندانہ فزکس پیرامیٹرز کے ساتھ شروع کریں۔
    *   سیمولیشن استحکام بمقابلہ کارکردگی کے لیے ایڈجسٹ کریں۔
    *   حقیقی دنیا کے ڈیٹا کے خلاف فزکس کے رویے کی تصدیق کریں۔

    ### 4. ڈیٹا کوالٹی ایشورنس

    *   تیار کردہ ڈیٹا کے لیے خودکار معیار کی جانچ نافذ کریں۔
    *   ڈیٹا کے تنوع کو یقینی بنانے کے لیے شماریاتی توثیق کا استعمال کریں۔
    *   حقیقی ڈیٹا کے خلاف مصنوعی ڈیٹا کی باقاعدگی سے تصدیق کریں۔

    ## عام مسائل کا حل (Troubleshooting)

    ### کارکردگی کے مسائل
    *   اگر فریم ریٹ کم ہو جائے تو منظر کی پیچیدگی کو کم کریں۔
    *   GPU میموری کے استعمال کو چیک کریں اور اس کے مطابق ایڈجسٹ کریں۔
    *   کارکردگی کے لیے آسان تصادم میش (collision meshes) استعمال کریں۔

    ### رینڈرنگ آرٹفیکٹس (Artifacts)
    *   مواد کی خصوصیات اور ٹیکسچرز کی تصدیق کریں۔
    *   لائٹنگ سیٹ اپ اور ایکسپوژر سیٹنگز کو چیک کریں۔
    *   کیمرے کی انشانکن (calibration) کے مناسب پیرامیٹرز کو یقینی بنائیں۔

    ### فزکس عدم استحکام
    *   فزکس ٹائم اسٹیپ اور سولور (solver) پیرامیٹرز کو ایڈجسٹ کریں۔
    *   مشترکہ حدود اور رکاوٹوں کی تصدیق کریں۔
    *   بڑے پیمانے (mass) اور جڑتا (inertia) کی خصوصیات کو چیک کریں۔

    ## ہینڈس آن مشق

    ایک مکمل Isaac Sim ماحول بنائیں جس میں شامل ہوں:

    1.  مختلف روشنی کے ساتھ ایک فوٹو ریئلسٹک انڈور منظر۔
    2.  مناسب فزکس کنفیگریشن کے ساتھ ایک ہیومنائیڈ روبوٹ۔
    3.  متعدد سینسر کی اقسام (کیمرہ، گہرائی، IMU، LiDAR)۔
    4.  ایک مصنوعی ڈیٹا جنریشن پائپ لائن۔
    5.  مضبوط تربیتی ڈیٹا کے لیے ڈومین رینڈمائزیشن۔
    6.  معیاری پیغام کی اقسام کے لیے ROS 2 انٹیگریشن۔

    یہ مشق آپ کو Physical AI ایپلی کیشنز کے لیے مصنوعی ڈیٹا بنانے اور فوٹو ریئلسٹک ماحول تخلیق کرنے کے لیے Isaac Sim کی جدید صلاحیتوں کے ساتھ عملی تجربہ فراہم کرے گی۔
  </div>
</BilingualChapter>