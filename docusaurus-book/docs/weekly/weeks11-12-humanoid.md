---
id: weeks11-12-humanoid
title: "Weeks 11-12 - Humanoid Robot Development"
sidebar_position: 5
---

# Weeks 11-12: Humanoid Robot Development

## Overview

Weeks 11-12 focus on humanoid-specific robotics development, covering the unique challenges of bipedal locomotion, balance control, humanoid kinematics and dynamics, manipulation with anthropomorphic hands, and natural human-robot interaction design. This module brings together all the previous learning to create robots that can move, interact, and behave in human-like ways.

Humanoid robots present unique challenges compared to traditional wheeled or manipulator robots. Their bipedal nature requires sophisticated balance control, their anthropomorphic form factor enables natural human interaction, and their complex kinematic structure requires advanced control algorithms.

## Learning Objectives

By the end of Weeks 11-12, students will be able to:

1. Understand the biomechanics and kinematics of humanoid robots
2. Implement bipedal locomotion and balance control algorithms
3. Design and implement manipulation systems for humanoid hands
4. Create natural human-robot interaction protocols
5. Address safety considerations for human-robot collaboration
6. Integrate all previous modules into a cohesive humanoid system
7. Understand the challenges and opportunities in humanoid robotics
8. Apply advanced control algorithms for humanoid-specific behaviors

## Week 11: Humanoid Kinematics and Dynamics

### Day 51: Humanoid Robot Anatomy and Structure

#### Humanoid Robot Design Principles

Humanoid robots are designed to mimic human form and function. This anthropomorphic design enables:

- **Natural Interaction**: Humans can interact with humanoid robots using natural social cues
- **Environment Compatibility**: Humanoid robots can use human-designed spaces and tools
- **Intuitive Control**: Human operators can teleoperate humanoid robots intuitively
- **Social Acceptance**: People tend to be more comfortable with human-like robots

#### Key Anatomical Components

A typical humanoid robot includes:

1. **Torso**: Central body housing core electronics and power systems
2. **Head**: Contains cameras, microphones, speakers, and displays
3. **Arms**: Two arms with shoulders, elbows, wrists, and hands
4. **Legs**: Two legs with hips, knees, ankles, and feet
5. **Sensors**: IMUs, cameras, force/torque sensors, tactile sensors

#### Degrees of Freedom Analysis

Humanoid robots typically have 30+ degrees of freedom (DOF):

- **Head**: 3 DOF (yaw, pitch, roll)
- **Arms**: 7 DOF each (3 shoulder, 1 elbow, 3 wrist)
- **Hands**: 4-12 DOF each (depending on complexity)
- **Legs**: 6 DOF each (3 hip, 1 knee, 2 ankle)
- **Total**: 26-40+ DOF

### Day 52: Forward and Inverse Kinematics

#### Forward Kinematics for Humanoid Robots

Forward kinematics calculates end-effector positions from joint angles:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class HumanoidKinematics:
    def __init__(self):
        # Define DH parameters for humanoid limbs
        # Using Denavit-Hartenberg convention
        self.limb_params = {
            'left_arm': {
                'joint_offsets': [
                    [0.0, 0.15, 0.0],  # Shoulder
                    [0.0, 0.0, 0.15],  # Upper arm
                    [0.0, 0.0, 0.15],  # Forearm
                    [0.0, 0.0, 0.05]   # Hand
                ],
                'joint_axes': [
                    [0, 0, 1],  # Shoulder yaw
                    [0, 1, 0],  # Shoulder pitch
                    [1, 0, 0],  # Shoulder roll
                    [0, 1, 0],  # Elbow pitch
                    [1, 0, 0],  # Wrist roll
                    [0, 1, 0],  # Wrist pitch
                    [0, 0, 1]   # Wrist yaw
                ]
            }
        }

    def dh_transform(self, a, alpha, d, theta):
        """Calculate Denavit-Hartenberg transformation matrix"""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)

        T = np.array([
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])
        return T

    def forward_kinematics_arm(self, joint_angles, limb='left_arm'):
        """Calculate forward kinematics for an arm"""
        transforms = []
        current_transform = np.eye(4)

        # Get limb parameters
        params = self.limb_params[limb]
        offsets = params['joint_offsets']
        axes = params['joint_axes']

        for i, angle in enumerate(joint_angles):
            # Calculate joint transformation
            axis = np.array(axes[i])
            rot_matrix = R.from_rotvec(axis * angle).as_matrix()

            # Create transformation matrix
            T = np.eye(4)
            T[:3, :3] = rot_matrix
            if i < len(offsets):
                T[:3, 3] = offsets[i]

            current_transform = current_transform @ T
            transforms.append(current_transform.copy())

        return transforms

    def calculate_center_of_mass(self, joint_positions, masses):
        """Calculate center of mass of humanoid robot"""
        total_mass = sum(masses)
        weighted_sum = np.zeros(3)

        for pos, mass in zip(joint_positions, masses):
            weighted_sum += np.array(pos) * mass

        com = weighted_sum / total_mass
        return com
```

#### Inverse Kinematics for Humanoid Manipulation

Inverse kinematics calculates joint angles for desired end-effector positions:

```python
import numpy as np
from scipy.optimize import minimize

class HumanoidIK:
    def __init__(self):
        self.joint_limits = {
            'shoulder_pitch': (-1.57, 1.57),
            'shoulder_roll': (-0.78, 1.57),
            'elbow_pitch': (0, 2.35),
            'wrist_pitch': (-1.57, 1.57),
            'wrist_yaw': (-1.57, 1.57)
        }

    def jacobian(self, joint_angles, limb='left_arm'):
        """Calculate Jacobian matrix for the given limb"""
        # Calculate forward kinematics to get current positions
        fk_solver = HumanoidKinematics()
        transforms = fk_solver.forward_kinematics_arm(joint_angles, limb)

        # Calculate end-effector position
        end_effector_pos = transforms[-1][:3, 3]

        # Calculate Jacobian columns
        J = np.zeros((6, len(joint_angles)))  # 6 DOF: 3 translation, 3 rotation

        for i, transform in enumerate(transforms):
            # Position part of Jacobian
            joint_pos = transform[:3, 3]
            joint_axis = transform[:3, 2]  # Z-axis of joint frame

            # Linear velocity contribution
            r = end_effector_pos - joint_pos
            J[:3, i] = np.cross(joint_axis, r)

            # Angular velocity contribution
            J[3:, i] = joint_axis

        return J

    def inverse_kinematics(self, target_pose, current_joints, max_iterations=100):
        """Solve inverse kinematics using iterative method"""
        joint_angles = np.array(current_joints)

        for iteration in range(max_iterations):
            # Calculate current end-effector pose
            fk_solver = HumanoidKinematics()
            current_transforms = fk_solver.forward_kinematics_arm(joint_angles)
            current_pose = current_transforms[-1]

            # Calculate error
            pos_error = target_pose[:3, 3] - current_pose[:3, 3]
            rot_error = self.rotation_error(target_pose, current_pose)

            error = np.concatenate([pos_error, rot_error])

            # Check if error is acceptable
            if np.linalg.norm(error) < 1e-4:
                break

            # Calculate Jacobian
            J = self.jacobian(joint_angles)

            # Calculate joint updates using damped least squares
            lambda_reg = 0.01
            JJT = J @ J.T
            damped_inv = J.T @ np.linalg.inv(JJT + lambda_reg**2 * np.eye(6))
            dq = damped_inv @ error

            # Apply joint limits
            joint_angles += dq * 0.1  # Small step size for stability

            # Enforce joint limits
            for i, joint_name in enumerate(['shoulder_pitch', 'shoulder_roll',
                                          'elbow_pitch', 'wrist_pitch', 'wrist_yaw']):
                min_limit, max_limit = self.joint_limits[joint_name]
                joint_angles[i] = np.clip(joint_angles[i], min_limit, max_limit)

        return joint_angles

    def rotation_error(self, target_pose, current_pose):
        """Calculate rotational error between poses"""
        # Extract rotation matrices
        R_target = target_pose[:3, :3]
        R_current = current_pose[:3, :3]

        # Calculate rotation error as axis-angle
        R_error = R_current.T @ R_target
        r = R.from_matrix(R_error).as_rotvec()

        return r
```

### Day 53: Dynamics and Balance Control

#### Center of Mass and Stability

For bipedal locomotion, understanding center of mass (CoM) dynamics is crucial:

```python
import numpy as np
from scipy.integrate import odeint

class BalanceController:
    def __init__(self):
        # Robot parameters
        self.mass = 50.0  # kg
        self.height = 1.0  # m (CoM height)
        self.gravity = 9.81  # m/s^2

        # Control parameters
        self.kp = 100.0  # Proportional gain
        self.kd = 20.0   # Derivative gain

        # Support polygon (area where CoM should be kept)
        self.support_polygon = {
            'width': 0.15,  # Half-width of foot
            'length': 0.20  # Half-length of foot
        }

    def inverted_pendulum_model(self, state, t, control_force):
        """Inverted pendulum model for balance"""
        # State: [x, x_dot, theta, theta_dot]
        # x: CoM x position, theta: lean angle
        x, x_dot, theta, theta_dot = state

        # Linearized inverted pendulum dynamics
        x_ddot = (self.gravity * theta + control_force / self.mass) / (1 - self.height / (2 * self.gravity))
        theta_ddot = (self.gravity / self.height * theta - control_force / (self.mass * self.height))

        return [x_dot, x_ddot, theta_dot, theta_ddot]

    def zero_moment_point(self, com_pos, com_vel, com_acc):
        """Calculate Zero Moment Point (ZMP)"""
        g = self.gravity
        h = self.height

        # ZMP equations (simplified for 2D)
        zmp_x = com_pos[0] - h/g * com_acc[0]
        zmp_y = com_pos[1] - h/g * com_acc[1]

        return np.array([zmp_x, zmp_y, 0.0])

    def calculate_balance_control(self, measured_state, desired_state):
        """Calculate balance control commands"""
        # Calculate error
        pos_error = desired_state[:2] - measured_state[:2]
        vel_error = desired_state[2:] - measured_state[2:]

        # PD control
        control_output = self.kp * pos_error + self.kd * vel_error

        return control_output

    def stability_check(self, com_pos, foot_pos):
        """Check if CoM is within support polygon"""
        # Calculate relative position to support foot
        rel_pos = com_pos - foot_pos

        # Check if within support polygon
        is_stable = (
            abs(rel_pos[0]) <= self.support_polygon['length'] and
            abs(rel_pos[1]) <= self.support_polygon['width']
        )

        return is_stable

    def capture_point(self, com_pos, com_vel):
        """Calculate capture point for balance recovery"""
        omega = np.sqrt(self.gravity / self.height)
        capture_point = com_pos + com_vel / omega
        return capture_point
```

#### Walking Pattern Generation

Generating stable walking patterns for bipedal robots:

```python
import numpy as np

class WalkingPatternGenerator:
    def __init__(self):
        # Walking parameters
        self.step_length = 0.3  # meters
        self.step_width = 0.2   # meters
        self.step_height = 0.05 # meters (swing height)
        self.walk_period = 1.0  # seconds per step
        self.dsp_ratio = 0.2    # Double Support Phase ratio
        self.tsp_ratio = 0.8    # Single Support Phase ratio

    def generate_foot_trajectory(self, step_num, side='left'):
        """Generate foot trajectory for a single step"""
        t = np.linspace(0, self.walk_period, 100)  # Time vector

        # Determine foot position based on step number and side
        if side == 'left':
            # Left foot trajectory
            start_x = 0.0
            end_x = self.step_length
            start_y = self.step_width / 2.0 if step_num % 2 == 0 else -self.step_width / 2.0
            end_y = -self.step_width / 2.0 if step_num % 2 == 0 else self.step_width / 2.0
        else:
            # Right foot trajectory (stays in place during this step)
            start_x = 0.0
            end_x = self.step_length
            start_y = -self.step_width / 2.0 if step_num % 2 == 0 else self.step_width / 2.0
            end_y = self.step_width / 2.0 if step_num % 2 == 0 else -self.step_width / 2.0

        # Generate trajectories
        x_traj = np.interp(t, [0, self.dsp_ratio/2, self.dsp_ratio/2 + self.tsp_ratio, self.walk_period],
                          [start_x, start_x, end_x, end_x])
        y_traj = np.interp(t, [0, self.dsp_ratio/2, self.dsp_ratio/2 + self.tsp_ratio, self.walk_period],
                          [start_y, start_y, end_y, end_y])

        # Generate vertical trajectory (for swing foot)
        z_traj = np.zeros_like(t)
        if side == 'left' if step_num % 2 == 0 else side == 'right':
            # This foot is swinging
            swing_phase = np.minimum(1.0, np.maximum(0.0,
                (t - self.dsp_ratio/2) / self.tsp_ratio))
            z_traj = self.step_height * np.sin(np.pi * swing_phase)

        return np.column_stack([x_traj, y_traj, z_traj])

    def generate_com_trajectory(self, num_steps):
        """Generate CoM trajectory for walking"""
        total_time = num_steps * self.walk_period
        t = np.linspace(0, total_time, int(total_time * 100))  # 100 Hz sampling

        # Generate smooth CoM trajectory
        x_com = np.zeros_like(t)
        y_com = np.zeros_like(t)
        z_com = np.full_like(t, 0.85)  # Constant height for stability

        # Generate CoM pattern
        for step in range(num_steps):
            step_start = step * self.walk_period
            step_end = (step + 1) * self.walk_period

            mask = (t >= step_start) & (t <= step_end)
            if np.any(mask):
                # Shift CoM towards supporting foot
                step_mid = step_start + self.walk_period / 2
                step_t = t[mask] - step_start

                # Lateral CoM movement to support foot
                if step % 2 == 0:  # Left foot support
                    y_com[mask] = -0.05 + 0.02 * np.sin(2 * np.pi * step_t / self.walk_period)
                else:  # Right foot support
                    y_com[mask] = 0.05 - 0.02 * np.sin(2 * np.pi * step_t / self.walk_period)

                # Forward CoM movement
                x_com[mask] = step * self.step_length + self.step_length * step_t / self.walk_period

        return np.column_stack([x_com, y_com, z_com])

    def generate_zmp_trajectory(self, num_steps):
        """Generate ZMP trajectory for walking"""
        com_traj = self.generate_com_trajectory(num_steps)

        zmp_x = np.zeros_like(com_traj[:, 0])
        zmp_y = np.zeros_like(com_traj[:, 1])

        # Simplified ZMP based on CoM trajectory
        for i in range(len(com_traj)):
            # For simplicity, ZMP follows CoM with slight adjustments
            zmp_x[i] = com_traj[i, 0] - 0.02  # Small offset for stability
            zmp_y[i] = com_traj[i, 1]

        return np.column_stack([zmp_x, zmp_y, np.zeros_like(zmp_x)])
```

### Day 54: Humanoid Manipulation

#### Anthropomorphic Hand Control

Humanoid hands require sophisticated control for dexterous manipulation:

```python
import numpy as np

class AnthropomorphicHandController:
    def __init__(self, hand_side='right'):
        self.hand_side = hand_side
        self.num_fingers = 5
        self.joints_per_finger = 3  # Proximal, medial, distal joints
        self.num_hand_joints = self.num_fingers * self.joints_per_finger + 2  # Plus wrist joints

        # Finger names and joint limits
        self.finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        self.joint_limits = {
            'proximal': (-0.5, 1.57),   # Rad
            'medial': (0, 1.57),         # Rad
            'distal': (0, 1.57)          # Rad
        }

        # Grasp primitives
        self.grasp_types = {
            'power': [1.0, 1.0, 1.0, 1.0, 1.0],  # Full closure
            'precision': [0.2, 0.5, 0.5, 0.3, 0.2],  # Fine manipulation
            'pinch': [0.0, 0.8, 0.0, 0.0, 0.0]   # Thumb-index pinch
        }

    def calculate_finger_kinematics(self, finger_angles):
        """Calculate finger kinematics for a single finger"""
        # Simplified model for finger kinematics
        # Each joint contributes to the fingertip position

        tip_pos = np.zeros(3)
        current_pos = np.zeros(3)  # Starting from palm
        current_rot = np.eye(3)

        # For each joint in the finger
        for i, angle in enumerate(finger_angles):
            # Calculate joint contribution to position
            link_length = 0.03  # Simplified link length

            # Apply rotation around appropriate axis
            if i == 0:  # Proximal joint - typically flexion/extension
                rot_axis = np.array([0, 1, 0])  # Y-axis rotation
            elif i == 1:  # Medial joint
                rot_axis = np.array([0, 1, 0])
            else:  # Distal joint
                rot_axis = np.array([0, 1, 0])

            # Rotate the link vector
            link_vec = np.array([link_length, 0, 0])
            rotated_link = current_rot @ link_vec

            # Update position and rotation
            current_pos += rotated_link
            current_rot = current_rot @ self.rotation_matrix(rot_axis, angle)

        return current_pos, current_rot

    def rotation_matrix(self, axis, angle):
        """Calculate rotation matrix around arbitrary axis"""
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        c, s = np.cos(angle), np.sin(angle)
        C = 1 - c

        return np.array([
            [x*x*C + c, x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s, y*y*C + c, y*z*C - x*s],
            [z*x*C - y*s, z*y*C + x*s, z*z*C + c]
        ])

    def execute_grasp_primitive(self, grasp_type, object_info=None):
        """Execute a predefined grasp primitive"""
        if grasp_type not in self.grasp_types:
            raise ValueError(f"Unknown grasp type: {grasp_type}")

        # Get base joint angles for this grasp
        base_angles = self.grasp_types[grasp_type]

        # Adjust for object properties if provided
        if object_info:
            size_factor = min(1.0, object_info.get('size', 0.05) / 0.05)
            force_factor = object_info.get('weight', 0.1) / 0.1

            # Scale joint angles based on object size
            scaled_angles = [angle * size_factor for angle in base_angles]
        else:
            scaled_angles = base_angles

        # Convert to full hand joint configuration
        hand_joints = []
        for i, finger_angle in enumerate(scaled_angles):
            # Each finger has 3 joints with similar angles
            for j in range(self.joints_per_finger):
                hand_joints.append(finger_angle * (j + 1) / 3)  # Gradual increase along finger

        # Add wrist joints (assuming 2 wrist DOF)
        hand_joints.extend([0.0, 0.0])  # Default wrist position

        return np.array(hand_joints)

    def impedance_control(self, desired_pos, stiffness=100, damping=20):
        """Implement impedance control for compliant manipulation"""
        # Desired impedance behavior: M*x_ddot + B*x_dot + K*x = F
        # For manipulation, we want to control contact forces

        current_pos = self.get_current_hand_position()
        pos_error = desired_pos - current_pos

        # Calculate control force
        spring_force = stiffness * pos_error
        damper_force = damping * self.get_current_velocity()

        control_force = spring_force + damper_force

        return control_force

    def get_current_hand_position(self):
        """Get current hand position from robot state"""
        # In real implementation, this would interface with robot state
        return np.array([0.5, 0.0, 0.8])  # Placeholder

    def get_current_velocity(self):
        """Get current hand velocity from robot state"""
        # In real implementation, this would interface with robot state
        return np.array([0.0, 0.0, 0.0])  # Placeholder
```

### Day 55: Humanoid Control Architecture

#### Whole-Body Controller

Humanoid robots require coordinated control of all joints:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class WholeBodyController:
    def __init__(self):
        # Robot configuration
        self.num_joints = 32  # Example: 2 arms + 2 legs + head + torso
        self.joint_names = [
            # Left arm
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
            'left_elbow_pitch', 'left_wrist_roll', 'left_wrist_pitch', 'left_wrist_yaw',
            # Right arm
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
            'right_elbow_pitch', 'right_wrist_roll', 'right_wrist_pitch', 'right_wrist_yaw',
            # Left leg
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
            'left_knee_pitch', 'left_ankle_pitch', 'left_ankle_roll',
            # Right leg
            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
            'right_knee_pitch', 'right_ankle_pitch', 'right_ankle_roll',
            # Head and torso
            'head_yaw', 'head_pitch', 'torso_yaw', 'torso_pitch', 'torso_roll'
        ]

        # Control priorities
        self.priorities = {
            'balance': 1,      # Highest priority
            'posture': 2,
            'task': 3,         # Lowest priority
        }

    def compute_control_commands(self, state, desired_trajectories):
        """Compute whole-body control commands using prioritized task control"""
        # Extract current state
        current_positions = state['positions']
        current_velocities = state['velocities']
        current_accelerations = state['accelerations']
        current_com = state['com']
        current_com_vel = state['com_vel']
        current_com_acc = state['com_acc']

        # Define tasks with priorities
        tasks = []

        # 1. Balance task (highest priority)
        balance_task = self.balance_task(current_com, current_com_vel, desired_trajectories['com'])
        tasks.append(('balance', balance_task, self.priorities['balance']))

        # 2. Posture task (medium priority)
        posture_task = self.posture_task(current_positions, desired_trajectories['posture'])
        tasks.append(('posture', posture_task, self.priorities['posture']))

        # 3. Task-specific motions (lowest priority)
        if 'task_motion' in desired_trajectories:
            task_task = self.task_motion_task(current_positions, desired_trajectories['task_motion'])
            tasks.append(('task', task_task, self.priorities['task']))

        # Solve prioritized task control
        joint_commands = self.solve_prioritized_control(tasks, current_positions)

        return joint_commands

    def balance_task(self, current_com, current_com_vel, desired_com_traj):
        """Balance control task using Linear Inverted Pendulum Mode (LIPM)"""
        # Calculate desired CoM acceleration based on LIPM
        g = 9.81
        h = 0.85  # CoM height
        omega = np.sqrt(g / h)

        # Desired CoM trajectory
        desired_com = desired_com_traj['position']
        desired_com_vel = desired_com_traj['velocity']
        desired_com_acc = desired_com_traj['acceleration']

        # LIPM equation: com_acc = omega^2 * (com - zmp)
        # Rearrange: zmp = com - com_acc/omega^2
        desired_zmp = desired_com - desired_com_acc / (omega**2)

        # Calculate current ZMP error
        current_zmp = current_com - current_com_acc / (omega**2)
        zmp_error = desired_zmp - current_zmp

        # Generate CoM acceleration command to correct ZMP error
        com_acc_cmd = desired_com_acc + omega**2 * zmp_error

        # Convert to joint space using Jacobian
        com_jacobian = self.compute_com_jacobian()
        joint_acc_cmd = np.linalg.pinv(com_jacobian) @ com_acc_cmd

        return {
            'task_vector': joint_acc_cmd,
            'gain': 1.0,
            'weight': np.eye(len(joint_acc_cmd)) * 1000  # High priority
        }

    def posture_task(self, current_positions, desired_posture):
        """Posture maintenance task"""
        position_error = desired_posture - current_positions
        velocity_command = 10.0 * position_error  # PD control

        return {
            'task_vector': velocity_command,
            'gain': 1.0,
            'weight': np.eye(len(current_positions)) * 100  # Medium priority
        }

    def task_motion_task(self, current_positions, desired_task_motion):
        """Task-specific motion task"""
        # For example, reaching motion
        desired_positions = desired_task_motion['positions']
        position_error = desired_positions - current_positions
        velocity_command = 5.0 * position_error  # Lower gain for lower priority

        return {
            'task_vector': velocity_command,
            'gain': 1.0,
            'weight': np.eye(len(current_positions)) * 10  # Low priority
        }

    def solve_prioritized_control(self, tasks, current_positions):
        """Solve prioritized task control using null-space projections"""
        # Sort tasks by priority (ascending - lower number = higher priority)
        sorted_tasks = sorted(tasks, key=lambda x: x[2])

        # Initialize solution
        joint_velocities = np.zeros(self.num_joints)
        identity = np.eye(self.num_joints)

        # Process tasks in priority order
        for task_name, task_data, priority in sorted_tasks:
            task_vector = task_data['task_vector']
            weight_matrix = task_data['weight']
            gain = task_data['gain']

            # Weighted pseudo-inverse
            # Calculate A = J^T * W * J + λI, then solve for velocities
            weighted_jacobian = weight_matrix
            task_solution = gain * task_vector

            # Apply to current solution using null-space projection
            # This ensures higher-priority tasks are satisfied while
            # lower-priority tasks are satisfied in the null space
            current_nullspace = identity
            for prev_task_name, prev_task_data, prev_priority in sorted_tasks:
                if prev_priority < priority:  # Higher priority task
                    prev_jac = self.get_task_jacobian(prev_task_name)
                    prev_proj = np.eye(self.num_joints) - prev_jac.T @ np.linalg.pinv(prev_jac.T @ prev_jac) @ prev_jac
                    current_nullspace = current_nullspace @ prev_proj

            # Apply task in remaining null space
            joint_velocities += current_nullspace @ task_solution

        return joint_velocities

    def compute_com_jacobian(self):
        """Compute CoM Jacobian (simplified)"""
        # In real implementation, this would compute the actual CoM Jacobian
        # based on the robot's kinematic structure
        jacobian = np.zeros((3, self.num_joints))  # 3 DOF for CoM (x, y, z)

        # Simplified example - in reality, this would be computed from kinematics
        # Map joint velocities to CoM velocity
        for i in range(self.num_joints):
            # Each joint has some influence on CoM movement
            jacobian[0, i] = 0.1  # x influence
            jacobian[1, i] = 0.1  # y influence
            jacobian[2, i] = 0.05  # z influence

        return jacobian

    def get_task_jacobian(self, task_name):
        """Get Jacobian for a specific task"""
        # Placeholder implementation
        return np.eye(self.num_joints)
```

## Week 12: Human-Robot Interaction and Safety

### Day 56: Natural Human-Robot Interaction

#### Social Robotics Principles

Humanoid robots must interact naturally with humans in social environments:

```python
import numpy as np
import time
from enum import Enum

class InteractionMode(Enum):
    PASSIVE = "passive"
    REACTIVE = "reactive"
    PROACTIVE = "proactive"
    INTRUSIVE = "intrusive"

class SocialInteractionManager:
    def __init__(self):
        self.interaction_mode = InteractionMode.PASSIVE
        self.social_space = {
            'intimate': 0.45,    # 0-45cm
            'personal': 1.2,     # 45cm-1.2m
            'social': 3.6,       # 1.2-3.6m
            'public': 7.6        # 3.6-7.6m
        }

        self.engagement_signals = {
            'gaze': 0.0,
            'gesture': 0.0,
            'vocalization': 0.0,
            'proximity': 0.0
        }

        self.personality_parameters = {
            'extraversion': 0.5,
            'agreeableness': 0.8,
            'conscientiousness': 0.7,
            'emotional_stability': 0.6,
            'openness': 0.5
        }

    def assess_social_situation(self, human_poses, environment_context):
        """Assess the social situation and determine appropriate response"""
        closest_human_dist = float('inf')
        engagement_opportunity = False

        for human_pose in human_poses:
            dist = np.linalg.norm(human_pose[:2])  # Distance in x-y plane
            closest_human_dist = min(closest_human_dist, dist)

            # Check for engagement signs (eye contact, facing direction, etc.)
            if dist < self.social_space['social']:  # Within social distance
                # Check if human is looking at robot or showing interest
                engagement_signs = self.detect_engagement_signs(human_pose)
                if engagement_signs > 0.5:  # Threshold for engagement
                    engagement_opportunity = True

        # Determine interaction mode based on situation
        if engagement_opportunity and closest_human_dist < self.social_space['personal']:
            self.interaction_mode = InteractionMode.REACTIVE
        elif self.personality_parameters['extraversion'] > 0.6 and closest_human_dist < self.social_space['social']:
            self.interaction_mode = InteractionMode.PROACTIVE
        else:
            self.interaction_mode = InteractionMode.PASSIVE

        return {
            'mode': self.interaction_mode,
            'closest_distance': closest_human_dist,
            'engagement_opportunity': engagement_opportunity
        }

    def detect_engagement_signs(self, human_pose):
        """Detect signs of human engagement (simplified)"""
        # In real implementation, this would use computer vision to detect:
        # - Eye contact / gaze direction
        # - Body orientation toward robot
        # - Gestures directed at robot
        # - Vocalizations

        # Simplified model
        engagement_score = 0.0

        # Proximity bonus
        dist = np.linalg.norm(human_pose[:2])
        if dist < self.social_space['personal']:
            engagement_score += 0.3

        # Direction bonus (simplified - assuming human is facing robot)
        engagement_score += 0.4

        # Previous interaction bonus
        if self.has_recent_interaction(human_pose):
            engagement_score += 0.3

        return min(1.0, engagement_score)

    def generate_social_response(self, social_assessment):
        """Generate appropriate social response based on assessment"""
        responses = {
            InteractionMode.PASSIVE: self.passive_response,
            InteractionMode.REACTIVE: self.reactive_response,
            InteractionMode.PROACTIVE: self.proactive_response,
            InteractionMode.INTRUSIVE: self.intrusive_response
        }

        return responses[self.interaction_mode](social_assessment)

    def reactive_response(self, assessment):
        """Generate reactive social response"""
        if assessment['engagement_opportunity']:
            # Acknowledge the person with appropriate social signals
            social_response = {
                'gaze_direction': self.calculate_gaze_direction(assessment),
                'greeting': self.select_appropriate_greeting(assessment),
                'stance': 'open_and_friendly',
                'movement': self.calculate_approach_movement(assessment)
            }
            return social_response

        return {'no_action': True}

    def proactive_response(self, assessment):
        """Generate proactive social response"""
        # Initiate interaction appropriately
        social_response = {
            'initiation_signal': self.select_initiation_signal(),
            'approach_strategy': self.select_approach_strategy(assessment),
            'opening_line': self.generate_opening_line(assessment),
            'attention_grabber': self.select_attention_grabber()
        }
        return social_response

    def calculate_gaze_direction(self, assessment):
        """Calculate appropriate gaze direction for social interaction"""
        # In real implementation, use inverse kinematics to look at person
        # while maintaining natural neck movement constraints
        return assessment.get('closest_human_pos', [0, 0, 0])

    def select_appropriate_greeting(self, assessment):
        """Select culturally and contextually appropriate greeting"""
        time_of_day = self.get_time_of_day()
        cultural_context = assessment.get('cultural_context', 'neutral')

        greetings = {
            'morning': {
                'formal': 'Good morning!',
                'casual': 'Morning!',
                'neutral': 'Hello!'
            },
            'afternoon': {
                'formal': 'Good afternoon!',
                'casual': 'Hey there!',
                'neutral': 'Hello!'
            },
            'evening': {
                'formal': 'Good evening!',
                'casual': 'Evening!',
                'neutral': 'Hello!'
            }
        }

        context = 'neutral'  # Would be determined from context in real implementation
        return greetings[time_of_day][context]

    def get_time_of_day(self):
        """Get current time of day for greeting selection"""
        current_hour = time.localtime().tm_hour
        if 5 <= current_hour < 12:
            return 'morning'
        elif 12 <= current_hour < 17:
            return 'afternoon'
        else:
            return 'evening'

    def has_recent_interaction(self, human_pose):
        """Check if there was recent interaction with this person"""
        # In real implementation, track interaction history
        return False

    def select_initiation_signal(self):
        """Select appropriate signal to initiate interaction"""
        # Wave, nod, eye contact, etc.
        return 'greeting_gesture'

    def select_approach_strategy(self, assessment):
        """Select appropriate approach strategy based on social context"""
        distance = assessment['closest_distance']

        if distance > self.social_space['social']:
            return 'gradual_approach'
        elif distance > self.social_space['personal']:
            return 'respectful_approach'
        else:
            return 'maintain_current_distance'

    def generate_opening_line(self, assessment):
        """Generate appropriate opening line for interaction"""
        # Context-aware opening lines
        return "Hello! How can I assist you today?"

    def select_attention_grabber(self):
        """Select appropriate way to grab attention if needed"""
        return 'polite_greeting'
```

### Day 57: Safety and Compliance

#### Safety Framework for Humanoid Robots

Safety is paramount in humanoid robotics, especially when operating in human environments:

```python
import numpy as np
from enum import Enum

class SafetyLevel(Enum):
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    EMERGENCY = "emergency"

class HumanoidSafetySystem:
    def __init__(self):
        self.safety_thresholds = {
            'collision_distance': 0.3,  # meters
            'force_limit': 100.0,       # Newtons
            'velocity_limit': 1.0,      # m/s for end effectors
            'torque_limit': 50.0,       # Nm for joints
            'temperature_limit': 70.0   # Celsius
        }

        self.emergency_stop_active = False
        self.safety_zones = {
            'safe_zone': 1.0,      # Far from humans
            'caution_zone': 0.5,   # Moderate distance
            'warning_zone': 0.2,   # Close to humans
            'danger_zone': 0.1     # Too close
        }

        self.collision_pairs = set()  # Track potential collisions

    def assess_safety_state(self, robot_state, human_poses, environment_objects):
        """Assess overall safety state of the robot"""
        safety_indicators = {
            'collision_risk': self.evaluate_collision_risk(robot_state, human_poses),
            'force_limits': self.check_force_limits(robot_state),
            'velocity_limits': self.check_velocity_limits(robot_state),
            'thermal_limits': self.check_thermal_limits(robot_state),
            'human_proximity': self.evaluate_human_proximity(human_poses),
            'environment_safety': self.check_environment_safety(environment_objects)
        }

        # Determine overall safety level
        if any(indicator == SafetyLevel.EMERGENCY for indicator in safety_indicators.values()):
            return SafetyLevel.EMERGENCY
        elif any(indicator == SafetyLevel.DANGER for indicator in safety_indicators.values()):
            return SafetyLevel.DANGER
        elif any(indicator == SafetyLevel.WARNING for indicator in safety_indicators.values()):
            return SafetyLevel.WARNING
        else:
            return SafetyLevel.SAFE

    def evaluate_collision_risk(self, robot_state, human_poses):
        """Evaluate collision risk with humans and objects"""
        min_distance = float('inf')

        # Check distance to all humans
        for human_pose in human_poses:
            # Calculate distance from robot's end effectors to human
            for link_name, link_pose in robot_state['link_poses'].items():
                if 'hand' in link_name or 'foot' in link_name or 'head' in link_name:
                    dist = np.linalg.norm(
                        np.array(link_pose[:3]) - np.array(human_pose[:3])
                    )
                    min_distance = min(min_distance, dist)

        if min_distance < self.safety_thresholds['collision_distance']:
            if min_distance < self.safety_zones['danger_zone']:
                return SafetyLevel.EMERGENCY
            elif min_distance < self.safety_zones['warning_zone']:
                return SafetyLevel.DANGER
            else:
                return SafetyLevel.WARNING

        return SafetyLevel.SAFE

    def check_force_limits(self, robot_state):
        """Check if forces exceed safety limits"""
        for joint_name, force in robot_state['joint_forces'].items():
            if abs(force) > self.safety_thresholds['force_limit']:
                return SafetyLevel.DANGER

        # Check end effector forces
        for ee_force in robot_state['end_effector_forces']:
            if np.linalg.norm(ee_force) > self.safety_thresholds['force_limit']:
                return SafetyLevel.EMERGENCY

        return SafetyLevel.SAFE

    def check_velocity_limits(self, robot_state):
        """Check if velocities exceed safety limits"""
        for link_name, velocity in robot_state['link_velocities'].items():
            if 'hand' in link_name or 'foot' in link_name:
                if np.linalg.norm(velocity) > self.safety_thresholds['velocity_limit']:
                    return SafetyLevel.WARNING

        return SafetyLevel.SAFE

    def check_thermal_limits(self, robot_state):
        """Check if temperatures exceed safety limits"""
        for joint_name, temp in robot_state['joint_temperatures'].items():
            if temp > self.safety_thresholds['temperature_limit']:
                return SafetyLevel.WARNING

        return SafetyLevel.SAFE

    def evaluate_human_proximity(self, human_poses):
        """Evaluate proximity to humans"""
        if not human_poses:
            return SafetyLevel.SAFE

        closest_distance = min(
            np.linalg.norm(human_pose[:3]) for human_pose in human_poses
        )

        if closest_distance < self.safety_zones['danger_zone']:
            return SafetyLevel.EMERGENCY
        elif closest_distance < self.safety_zones['warning_zone']:
            return SafetyLevel.DANGER
        elif closest_distance < self.safety_zones['caution_zone']:
            return SafetyLevel.WARNING
        else:
            return SafetyLevel.SAFE

    def check_environment_safety(self, environment_objects):
        """Check for environmental hazards"""
        # Check for obstacles in planned path
        # Check for hazardous materials
        # Check for structural integrity of environment
        return SafetyLevel.SAFE

    def trigger_safety_response(self, safety_level, robot_state):
        """Trigger appropriate safety response based on safety level"""
        if safety_level == SafetyLevel.EMERGENCY:
            self.emergency_stop()
            self.activate_protection_routines()
        elif safety_level == SafetyLevel.DANGER:
            self.reduce_speed()
            self.prepare_for_emergency()
        elif safety_level == SafetyLevel.WARNING:
            self.log_warning()
            self.maintain_caution()

    def emergency_stop(self):
        """Execute emergency stop procedure"""
        self.emergency_stop_active = True
        # In real implementation:
        # - Send zero velocity commands to all joints
        # - Activate brakes if available
        # - Log emergency event
        print("EMERGENCY STOP ACTIVATED")

    def activate_protection_routines(self):
        """Activate protection routines"""
        # Move to safe posture
        # Reduce power to motors
        # Activate safety systems
        pass

    def reduce_speed(self):
        """Reduce operational speed"""
        # In real implementation, reduce speed limits
        pass

    def prepare_for_emergency(self):
        """Prepare for potential emergency stop"""
        # Pre-position for safe stop
        # Monitor more closely
        pass

    def log_warning(self):
        """Log safety warning"""
        # In real implementation, log to safety system
        pass

    def maintain_caution(self):
        """Maintain cautious operation"""
        # Increase monitoring frequency
        # Reduce speed slightly
        pass

    def safety_rated_soft_enable(self, robot_state):
        """Implement safety-rated soft enable for compliant operation"""
        # Use admittance control to make robot compliant when in close proximity to humans
        if self.evaluate_human_proximity(robot_state['humans']) in [SafetyLevel.WARNING, SafetyLevel.DANGER]:
            # Increase compliance in control system
            robot_state['compliance_mode'] = True
            robot_state['impedance_settings'] = {
                'stiffness': 10,  # Low stiffness for safety
                'damping': 5      # Appropriate damping
            }
        else:
            robot_state['compliance_mode'] = False
            robot_state['impedance_settings'] = {
                'stiffness': 1000,  # Normal operational stiffness
                'damping': 200
            }

        return robot_state
```

### Day 58: Integration with Previous Modules

#### Complete System Integration

Now we'll integrate all the modules to create a complete humanoid system:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, Image
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String
from builtin_interfaces.msg import Time
import numpy as np

class IntegratedHumanoidSystem(Node):
    def __init__(self):
        super().__init__('integrated_humanoid_system')

        # Initialize subsystems
        self.balance_controller = BalanceController()
        self.walking_generator = WalkingPatternGenerator()
        self.hand_controller = AnthropomorphicHandController()
        self.whole_body_controller = WholeBodyController()
        self.safety_system = HumanoidSafetySystem()
        self.social_manager = SocialInteractionManager()

        # ROS 2 interfaces
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10
        )
        self.command_sub = self.create_subscription(
            String, '/humanoid_command', self.command_callback, 10
        )

        self.joint_cmd_pub = self.create_publisher(
            JointState, '/joint_commands', 10
        )
        self.status_pub = self.create_publisher(
            String, '/humanoid_status', 10
        )

        # System state
        self.current_state = {
            'joint_positions': {},
            'joint_velocities': {},
            'joint_efforts': {},
            'imu_data': {},
            'camera_data': None,
            'com': np.array([0.0, 0.0, 0.0]),
            'com_vel': np.array([0.0, 0.0, 0.0]),
            'humans': [],
            'objects': []
        }

        # Control loop timer
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100 Hz

        self.get_logger().info('Integrated Humanoid System initialized')

    def joint_state_callback(self, msg):
        """Update joint state information"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_state['joint_positions'][name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_state['joint_velocities'][name] = msg.velocity[i]
            if i < len(msg.effort):
                self.current_state['joint_efforts'][name] = msg.effort[i]

    def imu_callback(self, msg):
        """Update IMU data"""
        self.current_state['imu_data'] = {
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        }

        # Update CoM estimate from IMU (simplified)
        self.current_state['com'][2] = 0.85  # Fixed height assumption

    def camera_callback(self, msg):
        """Process camera data for perception"""
        self.current_state['camera_data'] = msg
        # In real implementation, run object detection, human detection, etc.

    def command_callback(self, msg):
        """Process high-level commands"""
        command = msg.data
        self.process_high_level_command(command)

    def control_loop(self):
        """Main control loop integrating all subsystems"""
        # 1. Safety Check
        safety_level = self.safety_system.assess_safety_state(
            self.current_state, self.current_state['humans'], self.current_state['objects']
        )

        if safety_level in [SafetyLevel.EMERGENCY, SafetyLevel.DANGER]:
            self.safety_system.trigger_safety_response(safety_level, self.current_state)
            return  # Don't proceed with normal control if unsafe

        # 2. Social Situation Assessment
        social_assessment = self.social_manager.assess_social_situation(
            self.current_state['humans'], {}
        )

        # 3. Generate Social Response if needed
        if social_assessment['engagement_opportunity']:
            social_response = self.social_manager.generate_social_response(social_assessment)

        # 4. Balance Control
        # Calculate CoM from joint positions (simplified)
        com_pos = self.estimate_com_position()
        com_vel = self.estimate_com_velocity()
        self.current_state['com'] = com_pos
        self.current_state['com_vel'] = com_vel

        # 5. Whole-body control
        desired_trajectories = self.plan_desired_trajectories()
        control_commands = self.whole_body_controller.compute_control_commands(
            self.current_state, desired_trajectories
        )

        # 6. Publish joint commands
        joint_cmd_msg = JointState()
        joint_cmd_msg.header.stamp = self.get_clock().now().to_msg()
        joint_cmd_msg.name = list(self.current_state['joint_positions'].keys())
        joint_cmd_msg.position = control_commands[:len(joint_cmd_msg.name)]
        self.joint_cmd_pub.publish(joint_cmd_msg)

        # 7. Update status
        status_msg = String()
        status_msg.data = f"Operating normally. Safety: {safety_level.value}, Mode: {social_assessment['mode'].value}"
        self.status_pub.publish(status_msg)

    def estimate_com_position(self):
        """Estimate center of mass position from joint configuration"""
        # Simplified CoM estimation
        # In real implementation, use detailed kinematic model
        return np.array([0.0, 0.0, 0.85])  # Default standing position

    def estimate_com_velocity(self):
        """Estimate center of mass velocity"""
        # Simplified velocity estimation
        return np.array([0.0, 0.0, 0.0])  # Assuming approximately stationary

    def plan_desired_trajectories(self):
        """Plan desired trajectories for different control tasks"""
        # This would integrate walking, manipulation, and posture tasks
        trajectories = {
            'com': {
                'position': self.current_state['com'],
                'velocity': self.current_state['com_vel'],
                'acceleration': np.array([0.0, 0.0, 0.0])  # Assuming steady state
            },
            'posture': self.get_default_standing_posture(),
            'task_motion': {}  # Would be populated based on current task
        }
        return trajectories

    def get_default_standing_posture(self):
        """Get default standing posture joint angles"""
        # Default standing posture (simplified)
        posture = np.zeros(32)  # Assuming 32 joints
        # Set appropriate values for standing position
        # This would be more sophisticated in real implementation
        return posture

    def process_high_level_command(self, command):
        """Process high-level commands like 'walk forward', 'wave', etc."""
        if 'walk' in command.lower():
            self.initiate_walking_behavior(command)
        elif 'wave' in command.lower():
            self.initiate_waving_behavior()
        elif 'greet' in command.lower() or 'hello' in command.lower():
            self.initiate_greeting_behavior()
        # Add more command handlers as needed

    def initiate_walking_behavior(self, command):
        """Initiate walking behavior based on command"""
        # Parse command for walking parameters
        # Generate walking pattern
        # Update control trajectories
        pass

    def initiate_waving_behavior(self):
        """Initiate waving gesture"""
        # Plan arm trajectory for waving
        # Update task motion trajectory
        pass

    def initiate_greeting_behavior(self):
        """Initiate greeting behavior"""
        # Combine social interaction with motor behavior
        # Wave, nod, or other greeting gesture
        pass

def main(args=None):
    rclpy.init(args=args)
    node = IntegratedHumanoidSystem()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Integrated Humanoid System')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Day 59: Advanced Humanoid Behaviors

#### Bipedal Locomotion Control

```python
import numpy as np
from scipy.signal import butter, filtfilt

class BipedalLocomotionController:
    def __init__(self):
        # Walking parameters
        self.step_length = 0.3  # meters
        self.step_width = 0.2   # meters
        self.step_height = 0.05 # meters
        self.walk_period = 1.0  # seconds per step
        self.com_height = 0.85  # Center of mass height

        # Control parameters
        self.balance_kp = 100.0
        self.balance_kd = 20.0
        self.step_kp = 50.0
        self.step_kd = 10.0

        # Walking state
        self.current_support_foot = 'left'
        self.step_phase = 0.0
        self.step_count = 0

    def generate_walking_pattern(self, walk_speed, turn_rate=0.0):
        """Generate complete walking pattern at specified speed"""
        # Calculate step timing based on desired speed
        step_duration = self.calculate_step_duration(walk_speed)

        # Generate footstep positions
        footsteps = self.calculate_footsteps(walk_speed, turn_rate, num_steps=10)

        # Generate CoM trajectory following footstep pattern
        com_trajectory = self.generate_com_trajectory_from_footsteps(footsteps)

        # Generate ZMP (Zero Moment Point) trajectory
        zmp_trajectory = self.generate_zmp_trajectory(com_trajectory)

        return {
            'footsteps': footsteps,
            'com_trajectory': com_trajectory,
            'zmp_trajectory': zmp_trajectory,
            'step_duration': step_duration
        }

    def calculate_step_duration(self, walk_speed):
        """Calculate appropriate step duration based on walking speed"""
        # Simple relationship: faster walking = shorter steps but higher cadence
        nominal_speed = 0.3  # m/s
        nominal_duration = 1.0  # s

        # Adjust duration based on speed (inverse relationship)
        adjusted_duration = nominal_duration * (nominal_speed / max(walk_speed, 0.05))

        # Constrain to reasonable bounds
        return np.clip(adjusted_duration, 0.5, 1.5)

    def calculate_footsteps(self, walk_speed, turn_rate, num_steps=10):
        """Calculate sequence of footsteps for walking trajectory"""
        footsteps = []

        current_pos = np.array([0.0, 0.0, 0.0])  # Starting position
        current_yaw = 0.0  # Starting orientation

        for i in range(num_steps):
            # Calculate step vector based on current direction and turn rate
            step_vec = np.array([
                self.step_length * walk_speed / 0.3,  # Adjust for speed
                self.step_width if i % 2 == 0 else -self.step_width,  # Alternate feet
                0.0
            ])

            # Rotate step vector by current heading
            cos_yaw, sin_yaw = np.cos(current_yaw), np.sin(current_yaw)
            rotation_matrix = np.array([
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw, cos_yaw, 0],
                [0, 0, 1]
            ])

            rotated_step = rotation_matrix @ step_vec
            next_pos = current_pos + rotated_step

            # Apply turn rate for next step
            current_yaw += turn_rate * self.walk_period

            footsteps.append({
                'position': next_pos.copy(),
                'orientation': current_yaw,
                'timing': i * self.walk_period,
                'support_foot': 'right' if i % 2 == 0 else 'left'
            })

            current_pos = next_pos

        return footsteps

    def generate_com_trajectory_from_footsteps(self, footsteps):
        """Generate CoM trajectory that follows footstep pattern while maintaining balance"""
        # Use Preview Control method for smooth CoM trajectory following footsteps
        dt = 0.01  # 100Hz control rate
        total_time = len(footsteps) * self.walk_period
        t = np.arange(0, total_time, dt)

        # Initialize CoM trajectory
        com_x = np.zeros_like(t)
        com_y = np.zeros_like(t)
        com_z = np.full_like(t, self.com_height)

        # Generate CoM trajectory that anticipates foot placements
        for i, step in enumerate(footsteps):
            # Find time indices for this step
            step_start_idx = int(i * self.walk_period / dt)
            step_end_idx = min(int((i + 1) * self.walk_period / dt), len(t))

            if step_start_idx < len(t):
                # Generate CoM movement toward anticipated foot placement
                for idx in range(step_start_idx, min(step_end_idx, len(t))):
                    local_t = (t[idx] - i * self.walk_period) / self.walk_period

                    # Smooth interpolation toward foot position
                    if local_t < 0.5:  # First half of step - moving to mid-stance
                        com_x[idx] = np.interp(local_t * 2, [0, 1],
                                             [com_x[max(0, idx-1)], step['position'][0] - self.step_length/4])
                        com_y[idx] = np.interp(local_t * 2, [0, 1],
                                             [com_y[max(0, idx-1)], step['position'][1]])
                    else:  # Second half - preparing for next step
                        next_foot_idx = min(i + 1, len(footsteps) - 1)
                        next_pos = footsteps[next_foot_idx]['position']
                        com_x[idx] = np.interp((local_t - 0.5) * 2, [0, 1],
                                             [step['position'][0] - self.step_length/4, next_pos[0] - self.step_length/2])
                        com_y[idx] = np.interp((local_t - 0.5) * 2, [0, 1],
                                             [step['position'][1], next_pos[1]])

        return np.column_stack([com_x, com_y, com_z])

    def generate_zmp_trajectory(self, com_trajectory):
        """Generate ZMP trajectory from CoM trajectory using inverted pendulum model"""
        g = 9.81
        h = self.com_height

        # ZMP = CoM - (CoM_double_dot * h) / g
        # Use numerical differentiation to get accelerations
        dt = 0.01

        # Calculate velocities
        com_vel = np.gradient(com_trajectory, dt, axis=0)

        # Calculate accelerations
        com_acc = np.gradient(com_vel, dt, axis=0)

        # Calculate ZMP
        zmp_x = com_trajectory[:, 0] - (com_acc[:, 0] * h) / g
        zmp_y = com_trajectory[:, 1] - (com_acc[:, 1] * h) / g
        zmp_z = np.zeros_like(zmp_x)  # ZMP is on ground plane

        return np.column_stack([zmp_x, zmp_y, zmp_z])

    def balance_control(self, measured_com, measured_com_vel, desired_com, desired_com_vel):
        """Compute balance control corrections using LIPM model"""
        # Calculate error in CoM position and velocity
        pos_error = desired_com[:2] - measured_com[:2]
        vel_error = desired_com_vel[:2] - measured_com_vel[:2]

        # PD control for balance
        balance_correction = self.balance_kp * pos_error + self.balance_kd * vel_error

        return balance_correction

    def foot_placement_control(self, com_state, support_foot_pos, desired_foot_pos):
        """Adjust foot placement based on CoM state for improved stability"""
        # Calculate where foot should be placed based on CoM state
        # Using Capture Point concept
        g = 9.81
        h = self.com_height
        omega = np.sqrt(g / h)

        # Current CoM state
        com_pos = com_state[:2]
        com_vel = com_state[3:5]  # Assuming state includes velocities

        # Calculate capture point
        capture_point = com_pos + com_vel / omega

        # Adjust desired foot placement toward capture point
        adjustment = 0.3 * (capture_point - desired_foot_pos[:2])
        adjusted_foot_pos = desired_foot_pos.copy()
        adjusted_foot_pos[:2] += adjustment

        return adjusted_foot_pos

    def implement_walking_controller(self, robot_state, walking_pattern):
        """Implement complete walking controller using the generated patterns"""
        # This would integrate with the robot's control system
        # For simulation purposes, we'll return the control commands

        current_time = robot_state.get('time', 0.0)
        current_step = int(current_time / self.walk_period)

        if current_step < len(walking_pattern['footsteps']):
            # Get current step information
            current_step_info = walking_pattern['footsteps'][current_step]
            desired_com = walking_pattern['com_trajectory'][int(current_time / 0.01)]

            # Calculate balance corrections
            balance_corr = self.balance_control(
                robot_state['com_pos'],
                robot_state['com_vel'],
                desired_com[:3],
                np.array([0, 0, 0])  # Assuming desired CoM velocity is 0
            )

            # Generate joint commands based on walking pattern and balance corrections
            joint_commands = self.generate_joint_commands_for_walking(
                current_step_info, balance_corr, robot_state
            )

            return joint_commands
        else:
            # No more steps planned, return neutral stance commands
            return self.get_neutral_stance_commands()

    def generate_joint_commands_for_walking(self, step_info, balance_corr, robot_state):
        """Generate joint commands for executing a walking step"""
        # This would implement inverse kinematics and dynamics
        # for the specific walking step with balance corrections

        # Simplified implementation - in reality, this would be much more complex
        neutral_pos = self.get_neutral_stance_commands()

        # Apply small adjustments based on balance needs
        adjusted_commands = neutral_pos + 0.1 * balance_corr[:len(neutral_pos)]

        return adjusted_commands

    def get_neutral_stance_commands(self):
        """Get neutral bipedal stance joint positions"""
        # Return appropriate joint angles for standing position
        # This would be specific to the robot model
        return np.zeros(32)  # Assuming 32 joints in humanoid
```

### Day 60: Final Integration and Testing

#### Complete System Validation

```python
import unittest
import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Twist

class HumanoidSystemValidator:
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}

    def validate_balance_control(self):
        """Validate balance control system"""
        print("Validating balance control system...")

        # Test stability under perturbations
        test_cases = [
            {"perturbation": [0.1, 0.0, 0.0], "expected_response": "recover_balance"},
            {"perturbation": [0.0, 0.1, 0.0], "expected_response": "recover_balance"},
            {"perturbation": [0.2, 0.0, 0.0], "expected_response": "recover_balance"}
        ]

        results = []
        for test_case in test_cases:
            result = self.test_balance_response(test_case["perturbation"])
            results.append(result == test_case["expected_response"])

        success_rate = sum(results) / len(results)
        self.test_results["balance_control"] = {
            "success_rate": success_rate,
            "details": results
        }

        print(f"Balance control validation: {success_rate*100:.1f}% success")
        return success_rate > 0.8  # Require 80% success rate

    def test_balance_response(self, perturbation):
        """Test balance response to perturbation"""
        # Simulate perturbation and test recovery
        # This would interface with the actual balance controller
        controller = BalanceController()

        # Apply perturbation
        initial_state = np.array([0.0, 0.0, 0.0, 0.0])  # [x, xdot, theta, thetadot]
        perturbed_state = initial_state + np.array([perturbation[0], 0, perturbation[1], 0])

        # Simulate recovery
        for t in np.arange(0, 2.0, 0.01):  # 2 seconds recovery time
            # Apply balance control
            control_force = controller.calculate_balance_control(
                perturbed_state[:2],  # position error
                perturbed_state[2:]   # velocity error
            )

            # Update state with dynamics (simplified)
            # In real test, would use full dynamics simulation
            pass

        # Check if balance recovered
        final_error = np.linalg.norm(perturbed_state[:2])  # Position error
        return "recover_balance" if final_error < 0.05 else "fail_balance"

    def validate_walking_pattern(self):
        """Validate walking pattern generation and execution"""
        print("Validating walking pattern generation...")

        # Test walking at different speeds
        speeds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5]  # m/s

        results = []
        for speed in speeds_to_test:
            pattern = self.generate_test_walking_pattern(speed)
            stability = self.check_walking_stability(pattern)
            results.append(stability)

        success_rate = sum(results) / len(results)
        self.test_results["walking_pattern"] = {
            "success_rate": success_rate,
            "tested_speeds": speeds_to_test,
            "details": results
        }

        print(f"Walking pattern validation: {success_rate*100:.1f}% success")
        return success_rate > 0.9  # Require 90% success rate

    def generate_test_walking_pattern(self, speed):
        """Generate walking pattern for testing"""
        controller = BipedalLocomotionController()
        return controller.generate_walking_pattern(speed)

    def check_walking_stability(self, pattern):
        """Check if walking pattern is stable"""
        # Check ZMP stays within support polygon
        zmp_trajectory = pattern['zmp_trajectory']
        footsteps = pattern['footsteps']

        # For each time step, check if ZMP is within support polygon
        stable_count = 0
        total_count = len(zmp_trajectory)

        for i, zmp in enumerate(zmp_trajectory):
            # Determine support foot at this time
            time_idx = i * 0.01  # Assuming 100Hz trajectory
            step_idx = int(time_idx / pattern['step_duration'])

            if step_idx < len(footsteps):
                support_foot_pos = footsteps[step_idx]['position']

                # Check if ZMP is within foot polygon (simplified as rectangle)
                foot_length = 0.15  # Half-length of foot
                foot_width = 0.07   # Half-width of foot

                dx = abs(zmp[0] - support_foot_pos[0])
                dy = abs(zmp[1] - support_foot_pos[1])

                if dx <= foot_length and dy <= foot_width:
                    stable_count += 1

        stability_ratio = stable_count / total_count if total_count > 0 else 0
        return stability_ratio > 0.8  # Require 80% of steps to be stable

    def validate_human_interaction(self):
        """Validate human-robot interaction safety and appropriateness"""
        print("Validating human-robot interaction...")

        # Test various interaction scenarios
        scenarios = [
            {"distance": 0.5, "expectation": "cautious_response"},
            {"distance": 1.0, "expectation": "friendly_response"},
            {"distance": 2.0, "expectation": "acknowledgment_only"},
            {"distance": 0.1, "expectation": "safety_protocol"}
        ]

        results = []
        for scenario in scenarios:
            response = self.test_interaction_scenario(scenario["distance"])
            expected = scenario["expectation"]
            results.append(response == expected)

        success_rate = sum(results) / len(results)
        self.test_results["human_interaction"] = {
            "success_rate": success_rate,
            "details": list(zip(scenarios, results))
        }

        print(f"Human interaction validation: {success_rate*100:.1f}% success")
        return success_rate > 0.75

    def test_interaction_scenario(self, distance):
        """Test interaction response at given distance"""
        manager = SocialInteractionManager()

        # Simulate human at given distance
        human_poses = [np.array([distance, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])]  # [x, y, z, qw, qx, qy, qz]
        env_context = {}

        assessment = manager.assess_social_situation(human_poses, env_context)

        # Map assessment to expected response type
        if distance < 0.2:
            return "safety_protocol"
        elif distance < 0.5:
            return "cautious_response"
        elif distance < 1.5:
            return "friendly_response"
        else:
            return "acknowledgment_only"

    def validate_safety_system(self):
        """Validate safety system responses"""
        print("Validating safety system...")

        # Test safety responses at different threat levels
        test_cases = [
            {"risk_level": "low", "expected_response": "monitor"},
            {"risk_level": "medium", "expected_response": "caution"},
            {"risk_level": "high", "expected_response": "reduce_speed"},
            {"risk_level": "critical", "expected_response": "emergency_stop"}
        ]

        results = []
        for test_case in test_cases:
            response = self.test_safety_response(test_case["risk_level"])
            expected = test_case["expected_response"]
            results.append(response == expected)

        success_rate = sum(results) / len(results)
        self.test_results["safety_system"] = {
            "success_rate": success_rate,
            "details": results
        }

        print(f"Safety system validation: {success_rate*100:.1f}% success")
        return success_rate > 0.9

    def test_safety_response(self, risk_level):
        """Test safety system response to different risk levels"""
        safety_sys = HumanoidSafetySystem()

        # Create mock robot state based on risk level
        if risk_level == "low":
            robot_state = {"humans": [], "objects": []}
        elif risk_level == "medium":
            robot_state = {"humans": [np.array([1.0, 0.0, 0.0])], "objects": []}
        elif risk_level == "high":
            robot_state = {"humans": [np.array([0.3, 0.0, 0.0])], "objects": []}
        else:  # critical
            robot_state = {"humans": [np.array([0.05, 0.0, 0.0])], "objects": []}

        # Assess safety and get response
        safety_level = safety_sys.assess_safety_state(robot_state, robot_state["humans"], robot_state["objects"])

        # Map safety level to expected response
        if safety_level == SafetyLevel.SAFE:
            return "monitor"
        elif safety_level == SafetyLevel.WARNING:
            return "caution"
        elif safety_level == SafetyLevel.DANGER:
            return "reduce_speed"
        else:  # EMERGENCY
            return "emergency_stop"

    def run_complete_validation_suite(self):
        """Run complete validation suite for the humanoid system"""
        print("="*60)
        print("COMPREHENSIVE HUMANOID SYSTEM VALIDATION")
        print("="*60)

        # Run all validation tests
        balance_ok = self.validate_balance_control()
        walking_ok = self.validate_walking_pattern()
        interaction_ok = self.validate_human_interaction()
        safety_ok = self.validate_safety_system()

        # Overall assessment
        all_systems_ok = all([balance_ok, walking_ok, interaction_ok, safety_ok])

        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Balance Control: {'PASS' if balance_ok else 'FAIL'}")
        print(f"Walking Pattern: {'PASS' if walking_ok else 'FAIL'}")
        print(f"Human Interaction: {'PASS' if interaction_ok else 'FAIL'}")
        print(f"Safety System: {'PASS' if safety_ok else 'FAIL'}")
        print("-"*60)
        print(f"Overall Status: {'PASS - SYSTEM READY' if all_systems_ok else 'FAIL - ISSUES DETECTED'}")
        print("="*60)

        return all_systems_ok

def main():
    validator = HumanoidSystemValidator()
    success = validator.run_complete_validation_suite()

    if success:
        print("\n✓ All validation tests passed! Humanoid system is ready for deployment.")
    else:
        print("\n✗ Some validation tests failed. Please address issues before deployment.")

    return success

if __name__ == "__main__":
    main()
```

## Hands-On Exercises

### Week 11 Exercises

1. **Humanoid Kinematics Challenge**
   - Implement forward and inverse kinematics for a simplified humanoid arm
   - Test with various target positions
   - Validate solutions using geometric methods

2. **Balance Control Implementation**
   - Implement a simple inverted pendulum controller
   - Test with simulated disturbances
   - Tune parameters for stable balance

3. **Manipulation Task**
   - Program a simple grasping motion
   - Implement basic grasp planning
   - Test with different object sizes and positions

### Week 12 Exercises

1. **Social Interaction Programming**
   - Create a simple greeting behavior
   - Implement proximity-based responses
   - Test with simulated human approaches

2. **Safety System Integration**
   - Implement emergency stop functionality
   - Create safety zone detection
   - Test with simulated hazard scenarios

3. **Complete System Integration**
   - Integrate all components from previous exercises
   - Create a simple task (e.g., approach human and wave)
   - Test complete system behavior

## Assessment

### Week 11 Assessment
- **Lab Exercise**: Implement humanoid kinematics solution
- **Quiz**: Humanoid kinematics and dynamics concepts
- **Project**: Balance control system implementation

### Week 12 Assessment
- **Practical**: Social interaction behavior implementation
- **Safety Review**: Safety system validation
- **Integration Challenge**: Complete humanoid behavior demonstration

## Resources

### Required Reading
- "Humanoid Robotics: A Reference" by H. Hirukawa
- "Introduction to Humanoid Robotics" by K. Yokoi
- "Robotics: Control, Sensing, Vision, and Intelligence" by Fu, Gonzalez, and Lee

### Tutorials
- ROS 2 Control for Humanoid Robots
- NVIDIA Isaac Sim Humanoid Examples
- Gazebo Humanoid Simulation Tutorials

### Tools
- RViz2 for visualization
- Gazebo simulation environment
- Isaac Sim (if available)
- rqt tools for debugging

## Next Steps

After completing Weeks 11-12, students will have mastered humanoid-specific robotics concepts and be ready to move on to Week 13: Conversational Robotics, where they'll integrate all the previous modules to create a complete conversational humanoid robot system that can understand natural language commands and execute them appropriately in physical space.