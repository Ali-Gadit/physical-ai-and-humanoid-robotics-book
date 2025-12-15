---
id: troubleshooting
title: "Troubleshooting Guides"
sidebar_position: 3
---

import BilingualChapter from '@site/src/components/BilingualChapter';

<BilingualChapter>
  <div className="english">
    # Troubleshooting Guides

    ## Overview

    This troubleshooting guide provides solutions for common issues encountered in Physical AI and humanoid robotics development using ROS 2, Gazebo, NVIDIA Isaac, and related technologies. The guide is organized by system component and includes diagnostic procedures, common solutions, and preventive measures.

    ## General Troubleshooting Principles

    ### 1. Systematic Approach
    - Identify the specific problem
    - Reproduce the issue consistently
    - Isolate variables one at a time
    - Document all changes and results
    - Verify the solution works

    ### 2. Diagnostic Tools
    - ROS 2 command-line tools (`ros2 topic`, `ros2 node`, `ros2 service`)
    - System monitoring tools (`htop`, `nvidia-smi`, `iotop`)
    - Log analysis tools (`journalctl`, ROS 2 logging)
    - Network debugging tools (`netstat`, `ping`, `nslookup`)

    ### 3. Documentation and Logs
    - Always check system logs first
    - Enable debug logging when needed
    - Document your troubleshooting steps
    - Keep system configurations backed up

    ## ROS 2 Troubleshooting

    ### Common Node Issues

    #### Node Not Found
    **Symptoms**: `ros2 run` command fails with "Node not found"

    **Causes**:
    - Package not built or installed
    - Package not sourced
    - Incorrect package name
    - Missing dependencies

    **Solutions**:
    ```bash
    # Check if package is built
    colcon build --packages-select your_package_name

    # Source the workspace
    source install/setup.bash

    # List available packages
    ros2 pkg list | grep your_package_name

    # Check package dependencies
    ros2 pkg dependencies your_package_name
    ```

    #### Node Won't Start
    **Symptoms**: Node crashes immediately or hangs during startup

    **Diagnosis**:
    ```bash
    # Run with debug output
    ros2 run your_package your_node --ros-args --log-level debug

    # Check for missing parameters
    ros2 param list

    # Monitor system resources
    htop
    ```

    **Solutions**:
    - Check for missing parameter declarations
    - Verify required dependencies are installed
    - Ensure sufficient system resources
    - Check for conflicting package versions

    ### Topic and Communication Issues

    #### Topic Not Publishing
    **Symptoms**: Publisher appears to be running but no messages are received

    **Diagnosis**:
    ```bash
    # Check topic status
    ros2 topic info /your_topic_name

    # Verify publisher is active
    ros2 node info your_publisher_node

    # Listen to topic
    ros2 topic echo /your_topic_name
    ```

    **Solutions**:
    - Check QoS profile compatibility
    - Verify network configuration (multicast)
    - Ensure correct message types
    - Check for rate limiting issues

    #### High Latency
    **Symptoms**: Messages arrive with significant delay

    **Solutions**:
    ```bash
    # Adjust QoS for low latency
    # Use BEST_EFFORT reliability for non-critical data
    # Use KEEP_LAST history with small depth
    # Increase network buffer sizes
    ```

    ### Service and Action Issues

    #### Service Call Times Out
    **Symptoms**: Service client times out waiting for response

    **Solutions**:
    - Verify service server is running
    - Check service type compatibility
    - Increase timeout values if processing is slow
    - Check for blocking operations in service callbacks

    ## Gazebo Simulation Troubleshooting

    ### Performance Issues

    #### Low Frame Rate
    **Symptoms**: Gazebo runs slowly, low FPS

    **Causes**:
    - Insufficient GPU resources
    - Complex models with high polygon counts
    - Physics engine overload
    - Sensor simulation overhead

    **Solutions**:
    ```bash
    # Reduce physics update rate
    gz sim -r --iterations-per-update 1

    # Simplify collision meshes
    # Use lower resolution textures
    # Reduce sensor update rates
    # Limit model complexity
    ```

    #### High CPU Usage
    **Symptoms**: High CPU usage, system slowdown

    **Solutions**:
    - Reduce physics update rate
    - Simplify physics models
    - Use faster collision algorithms
    - Limit simulation complexity

    ### Physics Simulation Issues

    #### Objects Falling Through Ground
    **Symptoms**: Models fall through static objects

    **Solutions**:
    ```xml
    <!-- Check collision geometry -->
    <collision name="collision">
      <geometry>
        <plane>  <!-- or appropriate shape -->
          <normal>0 0 1</normal>
          <size>100 100</size>
        </plane>
      </geometry>
    </collision>

    <!-- Verify static property -->
    <static>true</static>
    ```

    #### Unstable Joints
    **Symptoms**: Joint oscillation, instability

    **Solutions**:
    - Adjust ERP and CFM parameters
    - Reduce time step size
    - Increase solver iterations
    - Check joint limits and types

    ### Sensor Simulation Issues

    #### Camera Not Publishing
    **Symptoms**: Camera topics are empty or not updating

    **Solutions**:
    - Verify Gazebo rendering engine (OGRE2)
    - Check camera configuration
    - Verify sensor plugin loading
    - Check for graphics driver issues

    #### LiDAR Range Issues
    **Symptoms**: Unexpected range values, incorrect detection

    **Solutions**:
    - Check sensor range configuration
    - Verify scan resolution
    - Check for interference from other sensors
    - Validate sensor mounting position

    ## NVIDIA Isaac Troubleshooting

    ### Isaac Sim Issues

    #### Installation Problems
    **Symptoms**: Isaac Sim fails to launch or install

    **Requirements Check**:
    ```bash
    # Verify NVIDIA driver
    nvidia-smi

    # Check CUDA installation
    nvcc --version

    # Verify RTX GPU
    nvidia-smi --query-gpu=name,memory.total --format=csv

    # Check Omniverse installation
    ./run.sh --status
    ```

    #### Rendering Issues
    **Symptoms**: Poor rendering quality, crashes, or black screens

    **Solutions**:
    - Update to latest NVIDIA drivers
    - Verify RTX GPU compatibility
    - Check rendering engine settings
    - Adjust quality settings

    ### Isaac ROS Issues

    #### Visual SLAM Performance
    **Symptoms**: SLAM tracking fails, poor localization

    **Diagnosis**:
    ```bash
    # Check input topics
    ros2 topic hz /camera/left/image_rect_color
    ros2 topic hz /camera/right/image_rect_color
    ros2 topic hz /imu/data

    # Monitor computational load
    nvidia-smi
    ```

    **Solutions**:
    - Ensure adequate frame rate (30+ FPS)
    - Verify camera calibration
    - Check IMU data quality
    - Adjust tracking parameters

    #### GPU Memory Issues
    **Symptoms**: CUDA out of memory errors

    **Solutions**:
    ```bash
    # Monitor GPU memory
    watch nvidia-smi

    # Reduce batch sizes
    # Use mixed precision training
    # Clear GPU memory cache
    # Optimize model sizes
    ```

    ## Common Hardware Issues

    ### Jetson Platform Issues

    #### Overheating
    **Symptoms**: System thermal throttling, performance degradation

    **Prevention**:
    - Use adequate cooling (fans, heatsinks)
    - Monitor temperatures (`sudo tegrastats`)
    - Reduce computational load during peak usage
    - Optimize power management

    #### Power Supply Issues
    **Symptoms**: Unexpected shutdowns, brownouts

    **Solutions**:
    - Use recommended power supplies
    - Check for voltage drops
    - Monitor power consumption
    - Implement power management

    ### Sensor Integration Issues

    #### IMU Drift
    **Symptoms**: Orientation drift over time

    **Solutions**:
    - Implement sensor fusion (Kalman filters)
    - Regular calibration
    - Temperature compensation
    - Use magnetometer for absolute reference

    #### Camera Calibration
    **Symptoms**: Poor depth estimation, inaccurate measurements

    **Solutions**:
    ```bash
    # Use ROS 2 camera calibration tools
    ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.108 image:=/camera/image_raw camera:=/camera

    # Verify calibration results
    ros2 run image_view image_view --ros-args --remap image:=/camera/image_rect
    ```

    ## Network and Communication Issues

    ### ROS 2 Network Problems

    #### Nodes Cannot Communicate
    **Symptoms**: Nodes on different machines cannot see each other

    **Diagnosis**:
    ```bash
    # Check RMW implementation
    printenv | grep RMW

    # Verify network connectivity
    ping other_machine_ip

    # Check multicast settings
    ros2 daemon status
    ```

    **Solutions**:
    - Configure firewall for ROS 2 ports (8000-9000)
    - Set proper ROS_DOMAIN_ID
    - Use FastDDS or CycloneDDS configurations
    - Verify network interface settings

    ### Bandwidth Issues

    #### Network Saturation
    **Symptoms**: Message loss, high latency, dropped connections

    **Solutions**:
    - Reduce sensor data rates
    - Use compression for large messages
    - Prioritize critical topics
    - Implement bandwidth monitoring

    ## Development Environment Issues

    ### Build System Problems

    #### Package Build Failures
    **Symptoms**: `colcon build` fails with compilation errors

    **Diagnosis**:
    ```bash
    # Clean build directory
    rm -rf build/ install/ log/

    # Verbose build
    colcon build --event-handlers console_direct+

    # Build specific package
    colcon build --packages-select problematic_package
    ```

    **Solutions**:
    - Check for missing dependencies
    - Verify CMakeLists.txt and package.xml
    - Ensure proper compiler versions
    - Check for circular dependencies

    ### Python Environment Issues

    #### Dependency Conflicts
    **Symptoms**: Import errors, version conflicts

    **Solutions**:
    ```bash
    # Use virtual environments
    python3 -m venv ros_env
    source ros_env/bin/activate

    # Check for conflicting packages
    pip list | grep -i ros
    pip list | grep -i nvidia

    # Reinstall ROS 2 Python packages
    pip uninstall ros-rolling-* --yes
    ```

    ## Performance Optimization

    ### System Resource Management

    #### Memory Management
    ```bash
    # Monitor memory usage
    free -h
    cat /proc/meminfo

    # Optimize for real-time performance
    echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
    ```

    #### CPU Scheduling
    ```bash
    # For real-time performance
    sudo chrt -f 99 your_ros_process

    # CPU affinity for critical processes
    taskset -c 0-3 your_ros_process
    ```

    ### GPU Optimization

    #### CUDA Memory Management
    ```python
    # Clear GPU memory
    import torch
    torch.cuda.empty_cache()

    # Monitor GPU memory
    nvidia-smi -l 1
    ```

    ## Preventive Measures

    ### Regular Maintenance

    #### System Health Checks
    - Monitor system temperatures
    - Check disk space regularly
    - Update drivers and firmware
    - Backup configurations periodically

    #### Code Quality
    - Implement proper error handling
    - Use parameter validation
    - Include health monitoring
    - Plan for graceful degradation

    ### Documentation

    #### Configuration Management
    - Version control for configs
    - Document environment setup
    - Track hardware specifications
    - Maintain troubleshooting logs

    ## Emergency Procedures

    ### System Recovery

    #### ROS 2 Daemon Issues
    ```bash
    # Restart ROS 2 daemon
    ros2 daemon stop
    ros2 daemon start

    # Kill hanging processes
    pkill -f ros
    ```

    #### Simulation Recovery
    ```bash
    # Kill Gazebo processes
    pkill -f gz
    pkill -f gazebo

    # Clear simulation cache
    rm -rf ~/.gazebo/cache/*
    ```

    ## Getting Help

    ### Community Resources
    - ROS Answers: https://answers.ros.org/
    - NVIDIA Developer Forums: https://forums.developer.nvidia.com/
    - Gazebo Community: http://community.gazebosim.org/
    - Isaac Sim Documentation: https://docs.omniverse.nvidia.com/

    ### Support Channels
    - University technical support
    - Vendor support channels
    - Professional communities
    - Colleague networks

    ## Quick Reference

    ### Common Commands
    ```bash
    # ROS 2 status
    ros2 topic list
    ros2 node list
    ros2 service list

    # System monitoring
    nvidia-smi
    htop
    free -h

    # Network diagnostics
    ping localhost
    netstat -tuln
    ifconfig

    # Log monitoring
    journalctl -f
    ros2 launch --show-logs
    ```

    ### Emergency Shutdown
    ```bash
    # Safe robot stop
    ros2 topic pub /emergency_stop std_msgs/Bool '{data: true}'

    # Kill all ROS processes
    killall -9 ros2
    killall -9 gz
    ```

    This troubleshooting guide provides a comprehensive reference for diagnosing and resolving common issues in Physical AI and humanoid robotics development. Always remember to document your solutions and share knowledge with your team to build institutional expertise.
  </div>
  <div className="urdu">
    # خرابیوں کا سراغ لگانا (Troubleshooting)

    ## عام ROS 2 مسائل

    ### 1. Nodes ایک دوسرے کو نہیں دیکھ رہے

    *   **چیک کریں**: کیا `ROS_DOMAIN_ID` دونوں مشینوں پر ایک ہی ہے؟
    *   **چیک کریں**: کیا دونوں مشینیں ایک ہی نیٹ ورک پر ہیں؟
    *   **حل**: ملٹی کاسٹ (multicast) کی ترتیبات چیک کریں۔

    ### 2. Gazebo کریش ہو رہا ہے

    *   **چیک کریں**: کیا NVIDIA ڈرائیورز اپ ڈیٹ ہیں؟
    *   **چیک کریں**: کیا آپ کے پاس کافی RAM ہے؟
    *   **حل**: `verbose` موڈ میں چلائیں تاکہ غلطی کا پیغام دیکھ سکیں: `ign gazebo -v 4`
  </div>
</BilingualChapter>