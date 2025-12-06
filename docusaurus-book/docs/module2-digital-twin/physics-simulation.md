---
id: physics-simulation
title: "Physics Simulation in Gazebo"
sidebar_position: 3
---

# Physics Simulation in Gazebo

## Introduction

Physics simulation is the cornerstone of realistic robot simulation environments. In Gazebo, the physics engine calculates forces, torques, collisions, and movements to create realistic interactions between robots and their environment. For humanoid robots, accurate physics simulation is crucial for developing stable walking gaits, manipulation skills, and environmental interactions.

This section covers the physics simulation capabilities in Gazebo, including gravity, collisions, material properties, and how to tune parameters for realistic humanoid robot simulation.

## Physics Engines in Gazebo

Gazebo supports multiple physics engines, each with different characteristics:

### 1. ODE (Open Dynamics Engine)
- Default physics engine for Gazebo
- Good balance of speed and accuracy
- Well-suited for most humanoid robotics applications
- Supports complex collision detection

### 2. Bullet
- More robust for complex collision scenarios
- Better handling of stacked objects
- Slightly slower than ODE but more stable

### 3. DART (Dynamic Animation and Robotics Toolkit)
- Advanced humanoid simulation capabilities
- Better handling of complex articulated bodies
- Excellent for bipedal locomotion simulation

## Physics Configuration

### World Physics Settings

The physics configuration in your world file defines global simulation parameters:

```xml
<physics name="default_physics" type="ode">
  <!-- Time step for physics updates -->
  <max_step_size>0.001</max_step_size>

  <!-- Real-time update rate -->
  <real_time_update_rate>1000.0</real_time_update_rate>

  <!-- Real-time factor (1.0 = real-time speed) -->
  <real_time_factor>1.0</real_time_factor>

  <!-- Gravity vector (x, y, z in m/s²) -->
  <gravity>0 0 -9.8</gravity>

  <!-- ODE-specific parameters -->
  <ode>
    <!-- Solver type -->
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>

    <!-- Constraint parameters -->
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Key Parameters Explained

- **max_step_size**: Smaller values provide more accurate simulation but slower performance
- **real_time_update_rate**: How many physics updates per second
- **real_time_factor**: Controls simulation speed (1.0 = real-time, >1.0 = faster than real-time)
- **gravity**: Defines the gravitational field (standard is 9.8 m/s² downward)

## Collision Detection and Response

### Collision Properties

Each link in your robot needs proper collision properties:

```xml
<link name="link_name">
  <collision name="collision">
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
    <surface>
      <friction>
        <ode>
          <mu>0.5</mu>
          <mu2>0.5</mu2>
          <slip1>0.0</slip1>
          <slip2>0.0</slip2>
        </ode>
      </friction>
      <bounce>
        <restitution_coefficient>0.1</restitution_coefficient>
        <threshold>1e+06</threshold>
      </bounce>
      <contact>
        <ode>
          <soft_cfm>0</soft_cfm>
          <soft_erp>0.2</soft_erp>
          <kp>1e+13</kp>
          <kd>1</kd>
          <max_vel>0.01</max_vel>
          <min_depth>0</min_depth>
        </ode>
      </contact>
    </surface>
  </collision>
</link>
```

### Friction Parameters

- **mu**: Primary friction coefficient (0 = no friction, 1 = high friction)
- **mu2**: Secondary friction coefficient (for anisotropic friction)
- **slip1, slip2**: How much objects can slip past each other

### Bounce Parameters

- **restitution_coefficient**: How bouncy the surface is (0 = no bounce, 1 = perfectly elastic)
- **threshold**: Velocity threshold below which restitution is not applied

## Material Properties for Humanoid Robots

### Inertial Properties

Accurate inertial properties are crucial for realistic humanoid simulation:

```xml
<link name="upper_leg">
  <inertial>
    <!-- Mass in kg -->
    <mass>2.0</mass>

    <!-- Inertia matrix -->
    <inertia>
      <ixx>0.01</ixx>
      <ixy>0.0</ixy>
      <ixz>0.0</ixz>
      <iyy>0.01</iyy>
      <iyz>0.0</iyz>
      <izz>0.002</izz>
    </inertia>
  </inertial>
</link>
```

### Center of Mass Considerations

For humanoid robots, pay special attention to:
- Proper mass distribution across body parts
- Accurate center of mass positioning
- Moment of inertia values that reflect the actual shape

## Gravity and Environmental Forces

### Custom Gravity Fields

You can modify gravity for specific scenarios:

```xml
<!-- For moon simulation (1/6 Earth gravity) -->
<gravity>0 0 -1.63</gravity>

<!-- For zero-gravity environment -->
<gravity>0 0 0</gravity>
```

### Additional Forces

Gazebo supports other environmental forces:

```xml
<world name="custom_forces">
  <!-- Physics with custom forces -->
  <physics name="custom_physics" type="ode">
    <gravity>0 0 -9.8</gravity>
    <!-- Add wind, magnetic fields, etc. through plugins -->
  </physics>

  <!-- Custom force plugin -->
  <plugin name="custom_force" filename="libCustomForcePlugin.so">
    <force_vector>0.1 0 0</force_vector>
  </plugin>
</world>
```

## Tuning Physics for Humanoid Robots

### Stability Considerations

Humanoid robots are inherently unstable. Key tuning parameters:

1. **Time Step**: Use smaller time steps (0.001s or smaller) for stability
2. **Solver Iterations**: Increase iterations for more stable solutions
3. **Constraint Parameters**: Adjust ERP and CFM for stable joints

### Walking Gait Optimization

For bipedal locomotion simulation:

```xml
<physics name="humanoid_physics" type="ode">
  <max_step_size>0.0005</max_step_size>
  <real_time_update_rate>2000.0</real_time_update_rate>
  <ode>
    <solver>
      <type>quick</type>
      <iters>50</iters>  <!-- More iterations for stability -->
      <sor>1.0</sor>
    </solver>
    <constraints>
      <cfm>1e-5</cfm>    <!-- Constraint Force Mixing -->
      <erp>0.1</erp>     <!-- Error Reduction Parameter -->
    </constraints>
  </ode>
</physics>
```

## Collision Mesh Optimization

### Visual vs. Collision Geometry

Use different geometries for visual and collision:

```xml
<link name="foot">
  <!-- Detailed visual geometry -->
  <visual name="visual">
    <geometry>
      <mesh filename="package://humanoid_description/meshes/foot.dae"/>
    </geometry>
  </visual>

  <!-- Simplified collision geometry -->
  <collision name="collision">
    <geometry>
      <box size="0.2 0.1 0.05"/>  <!-- Simplified box for collision -->
    </geometry>
  </collision>
</link>
```

### Contact Stabilization

For stable foot-ground contact:

```xml
<collision name="foot_collision">
  <geometry>
    <box size="0.15 0.08 0.02"/>
  </geometry>
  <surface>
    <contact>
      <ode>
        <soft_erp>0.1</soft_erp>    <!-- Error reduction for contacts -->
        <soft_cfm>0.001</soft_cfm>  <!-- Constraint force mixing -->
        <kp>1e+6</kp>              <!-- Contact stiffness -->
        <kd>100</kd>               <!-- Contact damping -->
      </ode>
    </contact>
    <friction>
      <ode>
        <mu>0.8</mu>   <!-- High friction for stable walking -->
        <mu2>0.8</mu2>
      </ode>
    </friction>
  </surface>
</collision>
```

## Performance Optimization

### Balancing Accuracy and Speed

For real-time humanoid simulation:

1. **Coarse Collision Geometry**: Use simple shapes for collision
2. **Adaptive Time Stepping**: Adjust step size based on simulation complexity
3. **Selective Detail**: Add detail only where needed (e.g., hands for manipulation)

### Multi-Resolution Simulation

```xml
<!-- For different simulation needs -->
<physics name="fast_simulation" type="ode">
  <max_step_size>0.01</max_step_size>  <!-- Faster but less accurate -->
  <real_time_update_rate>100.0</real_time_update_rate>
</physics>

<physics name="detailed_simulation" type="ode">
  <max_step_size>0.0001</max_step_size>  <!-- Slower but more accurate -->
  <real_time_update_rate>10000.0</real_time_update_rate>
</physics>
```

## Debugging Physics Issues

### Common Problems and Solutions

1. **Robot Falls Through Ground**:
   - Check collision geometries
   - Verify surface parameters
   - Increase contact iterations

2. **Jittery Movement**:
   - Decrease time step
   - Adjust ERP/CFM parameters
   - Check inertial properties

3. **Unstable Joints**:
   - Verify joint limits and types
   - Check transmission elements
   - Adjust constraint parameters

### Physics Debugging Tools

Gazebo provides visualization tools for physics debugging:

```bash
# Enable contact visualization
gz sim -r my_world.sdf --render-engine ogre2 --verbose

# Use Gazebo GUI to visualize contact forces
# In the GUI: View -> Contacts
```

## Advanced Physics Features

### Soft Body Simulation

For more realistic interactions:

```xml
<model name="soft_object">
  <link name="soft_link">
    <collision name="collision">
      <geometry>
        <mesh filename="soft_body.stl"/>
      </geometry>
    </collision>
    <!-- Soft body properties through custom plugins -->
  </link>
</model>
```

### Fluid Simulation

For swimming or underwater robots:

```xml
<world name="underwater">
  <physics name="fluid_physics" type="ode">
    <gravity>0 0 -9.8</gravity>
  </physics>
  <!-- Fluid dynamics through plugins -->
  <plugin name="fluid_dynamics" filename="libFluidDynamicsPlugin.so">
    <density>1000</density>  <!-- Water density -->
    <viscosity>0.001</viscosity>
  </plugin>
</world>
```

## Physics Validation

### Comparing to Real-World Data

To validate your physics simulation:

1. **Record Real Robot Data**: Collect data from physical robots
2. **Simulate Same Scenarios**: Run identical tests in simulation
3. **Compare Results**: Analyze differences and adjust parameters

### Quantitative Metrics

Track key metrics:
- Joint position errors
- Balance stability
- Walking speed accuracy
- Energy consumption patterns

## Hands-on Exercise

Create a physics-validated simulation environment that includes:

1. A humanoid robot with accurate inertial properties
2. Proper collision geometries for stable walking
3. Tuned physics parameters for realistic movement
4. A test scenario to validate physics behavior
5. Documentation of physics parameter choices and their effects

This exercise will give you practical experience with configuring physics simulation for humanoid robots and understanding the impact of different parameters on robot behavior.