---
id: nav2-path-planning
title: "Nav2 Path Planning for Bipedal Humanoid Movement"
sidebar_position: 4
---

# Nav2 Path Planning for Bipedal Humanoid Movement

## Introduction

Navigation 2 (Nav2) is the state-of-the-art navigation stack for ROS 2, providing advanced path planning, obstacle avoidance, and navigation capabilities. For humanoid robots, Nav2 requires specialized configuration to handle the unique challenges of bipedal locomotion, including balance constraints, step planning, and dynamic stability requirements.

This section covers Nav2 configuration and customization for humanoid robots, focusing on path planning algorithms that account for bipedal movement patterns and stability requirements.

## Nav2 Architecture Overview

### Core Components

Nav2 consists of several key components that work together:

1. **Global Planner**: Creates optimal paths from start to goal
2. **Local Planner**: Executes short-term navigation and obstacle avoidance
3. **Costmap 2D**: Maintains obstacle and cost information
4. **Behavior Trees**: Orchestrates navigation behaviors
5. **Recovery Behaviors**: Handles navigation failures

### Navigation Stack Flow

```
Goal Request → Global Planner → Path → Local Planner → Robot Controller
                    ↓              ↓
               Costmap (Static)  Costmap (Local)
```

## Installing and Setting Up Nav2

### Installation

```bash
# Install Nav2 packages
sudo apt update
sudo apt install ros-humble-navigation2
sudo apt install ros-humble-nav2-bringup
sudo apt install ros-humble-nav2-gui
sudo apt install ros-humble-nav2-rviz-plugins
```

### Basic Launch

```bash
# Launch Nav2 with default configuration
ros2 launch nav2_bringup navigation_launch.py

# Launch with simulation
ros2 launch nav2_bringup tb3_simulation_launch.py
```

## Nav2 Configuration for Humanoid Robots

### Basic Configuration File

Create a configuration file for humanoid navigation (`humanoid_nav2_params.yaml`):

```yaml
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

amcl_map_client:
  ros__parameters:
    use_sim_time: True

amcl_rclcpp_node:
  ros__parameters:
    use_sim_time: True

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: "map"
    robot_base_frame: "base_link"
    odom_topic: "/odom"
    bt_loop_duration: 10
    default_server_timeout: 20
    # Specify the path where the BT XML files are located
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_have_feedback_condition_bt_node
    - nav2_is_path_valid_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node

bt_navigator_rclcpp_node:
  ros__parameters:
    use_sim_time: True

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
      vx_max: 0.5
      vx_min: -0.2
      vy_max: 0.5
      wz_max: 1.0
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      stateful: True
      progress_checker: "progress_checker"
      goal_checker: "goal_checker"
      costmap_converter_plugin: "costmap_converter"
      costmap_converter_spin_thread: True
      costmap_converter_frequency: 5
      # Humanoid-specific parameters
      step_size: 0.3  # Maximum step size for bipedal locomotion
      balance_constraint: 0.8  # Balance stability factor

controller_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

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
  local_costmap_client:
    ros__parameters:
      use_sim_time: True
  local_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: True

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
  global_costmap_client:
    ros__parameters:
      use_sim_time: True
  global_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: True

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      # Use GridBased planner with humanoid-specific parameters
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
      # Humanoid-specific parameters
      step_size: 0.3  # Path step size for bipedal constraints
      min_distance_from_obstacle: 0.5  # Safety distance for humanoid

planner_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

recoveries_server:
  ros__parameters:
    costmap_topic: "local_costmap/costmap_raw"
    footprint_topic: "local_costmap/published_footprint"
    cycle_frequency: 10.0
    recovery_plugins: ["spin", "backup", "wait"]
    spin:
      plugin: "nav2_recoveries/Spin"
      # Humanoid-specific spin recovery
      spin_dist: 1.57  # 90 degrees
      time_allowance: 10.0
    backup:
      plugin: "nav2_recoveries/BackUp"
      # Humanoid-specific backup parameters
      backup_dist: -0.3  # Back up 30cm
      backup_speed: 0.05
      time_allowance: 10.0
    wait:
      plugin: "nav2_recoveries/Wait"
      # Humanoid-specific wait recovery
      wait_duration: 5.0

robot_state_publisher:
  ros__parameters:
    use_sim_time: True
```

## Humanoid-Specific Path Planning Challenges

### Bipedal Locomotion Constraints

Humanoid robots face unique challenges for path planning:

1. **Balance Requirements**: Must maintain center of mass within support polygon
2. **Step Size Limitations**: Limited step length and height
3. **Dynamic Stability**: Need to maintain stability during movement
4. **Foot Placement**: Precise foot placement required for stable walking

### Step Planning Integration

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np

class HumanoidStepPlanner(Node):
    def __init__(self):
        super().__init__('humanoid_step_planner')

        # Subscribe to Nav2 global plan
        self.path_sub = self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            10
        )

        # Publish refined step plan
        self.step_plan_pub = self.create_publisher(
            Path,
            '/step_plan',
            10
        )

        # Publish visualization markers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/step_markers',
            10
        )

        # Humanoid-specific parameters
        self.step_length = 0.3  # meters
        self.step_width = 0.2   # meters (side-step)
        self.max_step_height = 0.1  # meters (for stairs)
        self.support_polygon_radius = 0.15  # Support polygon around foot

        self.get_logger().info('Humanoid Step Planner initialized')

    def path_callback(self, msg):
        """Process global path and generate step-by-step plan"""
        if len(msg.poses) < 2:
            return

        # Convert global path to step plan considering humanoid constraints
        step_plan = self.generate_step_plan(msg.poses)

        # Publish refined step plan
        step_path_msg = Path()
        step_path_msg.header = msg.header
        step_path_msg.poses = step_plan

        self.step_plan_pub.publish(step_path_msg)

        # Publish visualization
        self.publish_step_markers(step_plan)

    def generate_step_plan(self, global_poses):
        """Generate step-by-step plan considering humanoid constraints"""
        step_poses = []

        # Start with current position
        if len(global_poses) > 0:
            step_poses.append(global_poses[0])

        # Process the path to generate feasible steps
        current_pos = np.array([
            global_poses[0].pose.position.x,
            global_poses[0].pose.position.y
        ])

        for i in range(1, len(global_poses)):
            target_pos = np.array([
                global_poses[i].pose.position.x,
                global_poses[i].pose.position.y
            ])

            # Calculate distance to target
            dist = np.linalg.norm(target_pos - current_pos)

            # Generate intermediate steps if needed
            if dist > self.step_length:
                # Calculate number of steps needed
                num_steps = int(np.ceil(dist / self.step_length))

                for step in range(1, num_steps + 1):
                    # Calculate intermediate position
                    ratio = step / num_steps
                    intermediate_pos = current_pos + ratio * (target_pos - current_pos)

                    # Create pose for this step
                    pose = PoseStamped()
                    pose.header = global_poses[i].header
                    pose.pose.position.x = float(intermediate_pos[0])
                    pose.pose.position.y = float(intermediate_pos[1])
                    pose.pose.position.z = 0.0  # Ground level

                    # Set orientation to face direction of movement
                    if step > 1:  # Not the first step
                        prev_pos = np.array([
                            step_poses[-1].pose.position.x,
                            step_poses[-1].pose.position.y
                        ])
                        direction = intermediate_pos - prev_pos
                        yaw = np.arctan2(direction[1], direction[0])

                        # Convert to quaternion
                        pose.pose.orientation.z = float(np.sin(yaw / 2))
                        pose.pose.orientation.w = float(np.cos(yaw / 2))

                    step_poses.append(pose)

            else:
                # Direct step to target
                pose = PoseStamped()
                pose.header = global_poses[i].header
                pose.pose.position.x = float(target_pos[0])
                pose.pose.position.y = float(target_pos[1])
                pose.pose.position.z = 0.0
                step_poses.append(pose)

            current_pos = target_pos

        return step_poses

    def publish_step_markers(self, step_poses):
        """Publish visualization markers for steps"""
        marker_array = MarkerArray()

        for i, pose in enumerate(step_poses):
            # Create marker for this step
            marker = Marker()
            marker.header = pose.header
            marker.ns = "steps"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose = pose.pose
            marker.pose.position.z = 0.05  # Slightly above ground

            marker.scale.x = 0.1  # Cylinder diameter
            marker.scale.y = 0.1
            marker.scale.z = 0.01  # Very thin cylinder

            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.7

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidStepPlanner()

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

## Advanced Path Planning Algorithms for Humanoids

### Humanoid-Aware Global Planner

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionServer
import numpy as np
from scipy.spatial import KDTree

class HumanoidAwarePlanner(Node):
    def __init__(self):
        super().__init__('humanoid_aware_planner')

        # Action server for path computation
        self._action_server = ActionServer(
            self,
            ComputePathToPose,
            'compute_path_to_pose',
            self.execute_path_planning
        )

        # Costmap subscription
        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.costmap_callback,
            10
        )

        # Humanoid-specific parameters
        self.robot_width = 0.4  # Humanoid width
        self.robot_length = 0.5 # Humanoid length
        self.step_size = 0.3    # Maximum step size
        self.turn_radius = 0.6  # Minimum turning radius for humanoid
        self.clearance = 0.6    # Safety clearance

        self.costmap = None
        self.costmap_resolution = 0.05
        self.costmap_origin = [0, 0]

    def costmap_callback(self, msg):
        """Update internal costmap representation"""
        self.costmap = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.costmap_resolution = msg.info.resolution
        self.costmap_origin = [msg.info.origin.position.x, msg.info.origin.position.y]

    def execute_path_planning(self, goal_handle):
        """Execute path planning with humanoid constraints"""
        goal = goal_handle.request.goal
        start = goal_handle.request.start

        self.get_logger().info(f'Computing path from ({start.pose.position.x}, {start.pose.position.y}) to ({goal.pose.position.x}, {goal.pose.position.y})')

        # Generate path considering humanoid constraints
        path = self.compute_humanoid_path(start.pose, goal.pose)

        if path is not None:
            # Create result
            result = ComputePathToPose.Result()
            result.path = path
            goal_handle.succeed()
            return result
        else:
            self.get_logger().error('Failed to find valid path')
            goal_handle.abort()
            return ComputePathToPose.Result()

    def compute_humanoid_path(self, start_pose, goal_pose):
        """Compute path considering humanoid-specific constraints"""
        # Convert poses to grid coordinates
        start_grid = self.world_to_grid(
            start_pose.position.x,
            start_pose.position.y
        )
        goal_grid = self.world_to_grid(
            goal_pose.position.x,
            goal_pose.position.y
        )

        # Check if start and goal are in valid areas
        if not self.is_valid_humanoid_position(start_grid[0], start_grid[1]):
            self.get_logger().error('Start position is not valid for humanoid')
            return None

        if not self.is_valid_humanoid_position(goal_grid[0], goal_grid[1]):
            self.get_logger().error('Goal position is not valid for humanoid')
            return None

        # Use A* with humanoid-specific constraints
        path_grid = self.a_star_humanoid(start_grid, goal_grid)

        if path_grid is None:
            return None

        # Convert grid path back to world coordinates
        path = Path()
        path.header.frame_id = 'map'

        for grid_pos in path_grid:
            world_pos = self.grid_to_world(grid_pos[0], grid_pos[1])

            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = 'map'
            pose_stamped.pose.position.x = world_pos[0]
            pose_stamped.pose.position.y = world_pos[1]
            pose_stamped.pose.position.z = 0.0

            # For now, keep orientation as is - in practice, you'd want to set proper orientation
            pose_stamped.pose.orientation.w = 1.0

            path.poses.append(pose_stamped)

        return path

    def is_valid_humanoid_position(self, x, y):
        """Check if position is valid for humanoid considering size and clearance"""
        if self.costmap is None:
            return False

        # Check bounds
        if x < 0 or x >= self.costmap.shape[1] or y < 0 or y >= self.costmap.shape[0]:
            return False

        # Get humanoid footprint in grid cells
        robot_radius_grid = int(self.clearance / self.costmap_resolution)

        # Check area around the position
        for dx in range(-robot_radius_grid, robot_radius_grid + 1):
            for dy in range(-robot_radius_grid, robot_radius_grid + 1):
                check_x, check_y = x + dx, y + dy
                if (0 <= check_x < self.costmap.shape[1] and
                    0 <= check_y < self.costmap.shape[0]):
                    if self.costmap[check_y, check_x] >= 50:  # Consider as occupied if cost >= 50
                        return False

        return True

    def a_star_humanoid(self, start, goal):
        """A* algorithm with humanoid constraints"""
        # Simplified A* implementation
        # In practice, you'd use a more sophisticated approach considering
        # humanoid kinematics and dynamics

        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            # Get node with lowest f_score
            current = min(open_set, key=lambda x: x[0])[1]
            open_set = [item for item in open_set if item[1] != current]

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            # Check neighbors (8-connected)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue

                    neighbor = (current[0] + dx, current[1] + dy)

                    # Check if neighbor is valid
                    if not self.is_valid_humanoid_position(neighbor[0], neighbor[1]):
                        continue

                    tentative_g_score = g_score[current] + self.distance(current, neighbor)

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                        open_set.append((f_score[neighbor], neighbor))

        return None  # No path found

    def heuristic(self, a, b):
        """Heuristic function for A*"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def distance(self, a, b):
        """Distance between two grid points"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        grid_x = int((x - self.costmap_origin[0]) / self.costmap_resolution)
        grid_y = int((y - self.costmap_origin[1]) / self.costmap_resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, x, y):
        """Convert grid coordinates to world coordinates"""
        world_x = x * self.costmap_resolution + self.costmap_origin[0]
        world_y = y * self.costmap_resolution + self.costmap_origin[1]
        return (world_x, world_y)

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidAwarePlanner()

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

## Local Planner for Humanoid Robots

### Humanoid-Specific Local Planner

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformListener, Buffer
import numpy as np

class HumanoidLocalPlanner(Node):
    def __init__(self):
        super().__init__('humanoid_local_planner')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_sub = self.create_subscription(
            Path, '/step_plan', self.path_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        # TF listener for transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Robot state
        self.current_pose = None
        self.current_twist = None
        self.global_plan = []
        self.current_waypoint_idx = 0
        self.safe_to_proceed = True

        # Humanoid-specific parameters
        self.linear_vel_max = 0.3  # m/s
        self.angular_vel_max = 0.5  # rad/s
        self.waypoint_dist_thresh = 0.2  # m
        self.horizon_distance = 1.0  # m
        self.balance_threshold = 0.1  # m (max deviation from center)

        # Control parameters
        self.linear_kp = 1.0
        self.angular_kp = 2.0

        # Timer for control loop
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

    def path_callback(self, msg):
        """Update global plan"""
        self.global_plan = msg.poses
        self.current_waypoint_idx = 0
        self.get_logger().info(f'Updated global plan with {len(self.global_plan)} waypoints')

    def odom_callback(self, msg):
        """Update current robot state"""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def scan_callback(self, msg):
        """Process laser scan for obstacle detection"""
        # Check for obstacles in path
        min_distance = min([d for d in msg.ranges if 0 < d < float('inf')], default=float('inf'))

        # Determine if it's safe to proceed
        self.safe_to_proceed = min_distance > 0.5  # 50cm safety margin

    def control_loop(self):
        """Main control loop for humanoid navigation"""
        if not self.global_plan or self.current_waypoint_idx >= len(self.global_plan):
            # Stop if no plan or reached end
            self.publish_velocity_command(0.0, 0.0)
            return

        if not self.current_pose or not self.safe_to_proceed:
            # Stop if no pose or unsafe conditions
            self.publish_velocity_command(0.0, 0.0)
            return

        # Get current target waypoint
        target_pose = self.global_plan[self.current_waypoint_idx].pose

        # Calculate distance to target
        dist_to_target = np.sqrt(
            (target_pose.position.x - self.current_pose.position.x)**2 +
            (target_pose.position.y - self.current_pose.position.y)**2
        )

        # Check if reached current waypoint
        if dist_to_target < self.waypoint_dist_thresh:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.global_plan):
                # Reached goal
                self.publish_velocity_command(0.0, 0.0)
                return

        # Calculate control commands
        linear_vel, angular_vel = self.calculate_control_commands(target_pose)

        # Apply humanoid constraints
        linear_vel = max(-self.linear_vel_max, min(linear_vel, self.linear_vel_max))
        angular_vel = max(-self.angular_vel_max, min(angular_vel, self.angular_vel_max))

        # Publish velocity command
        self.publish_velocity_command(linear_vel, angular_vel)

    def calculate_control_commands(self, target_pose):
        """Calculate linear and angular velocity commands"""
        # Calculate error
        dx = target_pose.position.x - self.current_pose.position.x
        dy = target_pose.position.y - self.current_pose.position.y

        # Calculate distance and angle to target
        dist_to_target = np.sqrt(dx**2 + dy**2)
        target_angle = np.arctan2(dy, dx)

        # Calculate current robot angle from orientation
        current_yaw = self.quaternion_to_yaw(self.current_pose.orientation)

        # Calculate angle error
        angle_error = target_angle - current_yaw
        # Normalize angle error to [-π, π]
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi

        # Simple proportional control
        linear_vel = self.linear_kp * dist_to_target
        angular_vel = self.angular_kp * angle_error

        # Adjust for humanoid-specific requirements
        if abs(angle_error) > 0.5:  # Need to turn significantly
            linear_vel *= 0.3  # Slow down while turning

        return linear_vel, angular_vel

    def quaternion_to_yaw(self, quaternion):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def publish_velocity_command(self, linear_vel, angular_vel):
        """Publish velocity command to robot"""
        cmd_msg = Twist()
        cmd_msg.linear.x = float(linear_vel)
        cmd_msg.angular.z = float(angular_vel)
        self.cmd_vel_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidLocalPlanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop robot before shutting down
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0
        cmd_msg.angular.z = 0.0
        node.cmd_vel_pub.publish(cmd_msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Behavior Trees for Humanoid Navigation

### Custom Behavior Tree for Humanoid Navigation

Create a custom behavior tree XML file (`humanoid_navigate_to_pose_w_replanning_and_recovery.xml`):

```xml
<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <PipelineSequence name="NavigateWithReplanning">
            <RecoveryNode number_of_retries="6" name="NavigateRecovery">
                <PipelineSequence name="NavigateWithSmoothing">
                    <GoalUpdated/>
                    <ComputePathToPose name="ComputePath" path_topic="local_plan" use_start_pose="false"/>
                    <FollowPath name="FollowPath" path_topic="local_plan"/>
                </PipelineSequence>
                <ReactiveFallback name="RecoveryFallback">
                    <GoalUpdated/>
                    <PipelineSequence name="RecoveryActions">
                        <ClearEntireCostmap name="ClearLocalCostmap-1" service_name="local_costmap/clear_entirely_local_costmap"/>
                        <ClearEntireCostmap name="ClearGlobalCostmap-1" service_name="global_costmap/clear_entirely_global_costmap"/>
                        <RecoveryNode number_of_retries="2" name="WallFollowRecovery">
                            <WaitUntilReady name="WaitForPath"/>
                            <ComputePathToPose name="ComputePath-2" path_topic="local_plan" use_start_pose="false"/>
                            <FollowPath name="FollowPath-2" path_topic="local_plan"/>
                        </RecoveryNode>
                    </PipelineSequence>
                </ReactiveFallback>
            </RecoveryNode>
            <ReactiveSequence name="GoalChecker">
                <GoalUpdated/>
                <IsGoalReached name="GoalReached"/>
            </ReactiveSequence>
        </PipelineSequence>
    </BehaviorTree>
</root>
```

## Integration with Isaac ROS

### Combining Isaac ROS with Nav2

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from sensor_msgs.msg import Imu
import numpy as np

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

        # Subscribe to IMU for balance information
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Navigation state
        self.current_pose = None
        self.balance_ok = True
        self.navigation_active = False

        # Balance thresholds
        self.roll_threshold = 0.3  # radians
        self.pitch_threshold = 0.3  # radians

    def pose_callback(self, msg):
        """Update current pose from Isaac ROS"""
        self.current_pose = msg

    def imu_callback(self, msg):
        """Check balance status from IMU"""
        # Convert quaternion to roll/pitch/yaw
        quat = msg.orientation
        roll, pitch, yaw = self.quaternion_to_rpy(quat)

        # Check if within balance thresholds
        self.balance_ok = (abs(roll) < self.roll_threshold and
                          abs(pitch) < self.pitch_threshold)

        if not self.balance_ok:
            self.get_logger().warn(f'Balance threshold exceeded: roll={roll:.2f}, pitch={pitch:.2f}')

    def quaternion_to_rpy(self, quaternion):
        """Convert quaternion to roll, pitch, yaw"""
        import math

        # Convert quaternion to RPY (Tait-Bryan angles)
        q = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q[3] * q[0] + q[1] * q[2])
        cosr_cosp = 1 - 2 * (q[0] * q[0] + q[1] * q[1])
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (q[3] * q[1] - q[2] * q[0])
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q[3] * q[2] + q[0] * q[1])
        cosy_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def navigate_to_pose(self, x, y, theta=0.0):
        """Navigate to specified pose with balance checks"""
        if not self.balance_ok:
            self.get_logger().error('Robot is not balanced, cannot navigate')
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0

        # Convert theta to quaternion
        goal_msg.pose.pose.orientation.z = float(np.sin(theta / 2))
        goal_msg.pose.pose.orientation.w = float(np.cos(theta / 2))

        self.nav_client.wait_for_server()
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.navigation_result_callback)

        self.navigation_active = True
        return True

    def navigation_result_callback(self, future):
        """Handle navigation result"""
        result = future.result()
        if result:
            self.get_logger().info('Navigation completed successfully')
        else:
            self.get_logger().error('Navigation failed')

        self.navigation_active = False

def main(args=None):
    rclpy.init(args=args)
    node = IsaacROSNav2Integrator()

    # Example: Navigate to a specific location after checking balance
    import time
    time.sleep(2)  # Allow time for initial pose estimation

    if node.balance_ok:
        node.navigate_to_pose(5.0, 5.0)  # Navigate to (5, 5)
    else:
        node.get_logger().error('Initial balance check failed')

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

## Performance Optimization and Tuning

### Parameter Tuning Guidelines

```yaml
# Tuning guidelines for humanoid navigation
tuning_guidelines:
  # For narrow corridors (humanoid width ~0.4m)
  local_costmap:
    robot_radius: 0.4  # Account for humanoid width
    inflation_radius: 0.6  # Extra safety for bipedal stability

  # For dynamic obstacle avoidance
  controller_server:
    # Humanoid moves more cautiously
    max_vel_x: 0.3  # Slower than wheeled robots
    max_vel_theta: 0.4
    # More conservative acceleration limits
    acc_lim_x: 0.5
    acc_lim_theta: 0.5

  # For step planning
  planner_server:
    # Plan with smaller steps for humanoid
    costmap_resolution: 0.05  # Higher resolution for precise planning
    # Allow more time for complex humanoid paths
    planner_frequency: 10.0
```

## Troubleshooting Common Issues

### Navigation Failures
- **Stuck in local minima**: Increase inflation radius in costmaps
- **Oscillating behavior**: Adjust controller parameters (lower gains)
- **Excessive computation**: Reduce costmap resolution or planner frequency

### Balance-Related Issues
- **Frequent falls during navigation**: Reduce speed and acceleration
- **Poor turning performance**: Adjust turning radius and step planning
- **Stability problems**: Implement balance feedback control

## Best Practices for Humanoid Navigation

1. **Safety First**: Always maintain safety margins larger than for wheeled robots
2. **Balance Awareness**: Integrate balance feedback into navigation decisions
3. **Step Planning**: Consider discrete step planning for bipedal locomotion
4. **Gradual Acceleration**: Implement smooth velocity profiles for stable walking
5. **Recovery Strategies**: Develop humanoid-specific recovery behaviors

## Hands-on Exercise

Create a complete Nav2 setup for humanoid navigation that includes:

1. Custom Nav2 configuration for humanoid robots
2. Step planning algorithm considering bipedal constraints
3. Integration with Isaac ROS for enhanced perception
4. Balance-aware navigation with IMU feedback
5. Behavior trees customized for humanoid navigation
6. Performance tuning and validation

This exercise will give you hands-on experience with Nav2 path planning specifically tailored for bipedal humanoid movement and the unique challenges it presents.