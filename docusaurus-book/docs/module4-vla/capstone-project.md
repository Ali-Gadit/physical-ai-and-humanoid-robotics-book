---
id: capstone-project
title: "Capstone Project: The Autonomous Humanoid"
sidebar_position: 4
---

import BilingualChapter from '@site/src/components/BilingualChapter';

<BilingualChapter>
  <div className="english">
    # Capstone Project: The Autonomous Humanoid

    ## Introduction

    The Autonomous Humanoid capstone project represents the culmination of all the concepts learned throughout this course. This comprehensive project integrates all four modules—The Robotic Nervous System (ROS 2), The Digital Twin (Gazebo & Unity), The AI-Robot Brain (NVIDIA Isaac™), and Vision-Language-Action (VLA)—to create a fully autonomous humanoid robot capable of receiving voice commands, planning paths, navigating obstacles, identifying objects using computer vision, and manipulating them appropriately.

    This project demonstrates the full spectrum of Physical AI capabilities, from low-level control to high-level cognitive planning, and provides a realistic scenario that showcases the integration of all course components.

    ## Project Overview

    ### Project Goal

    Develop an autonomous humanoid robot system that can:
    1. Receive and understand voice commands using OpenAI Whisper
    2. Plan navigation paths using cognitive planning with LLMs
    3. Navigate through environments while avoiding obstacles
    4. Identify and locate objects using computer vision
    5. Manipulate objects to complete tasks
    6. Interact naturally with humans through speech

    ### System Architecture

    The Autonomous Humanoid system integrates multiple technologies:

    ```
    User Voice Command → Whisper ASR → LLM Cognitive Planning → ROS 2 Control → Physical/Virtual Robot
           ↓                    ↓                ↓                    ↓              ↓
       Natural Language    Speech-to-Text   Task Decomposition   Action Execution  Environment
       Understanding       Processing       Action Sequencing    Interface        Interaction
    ```

    ## Technical Requirements

    ### Hardware Requirements

    The project can be implemented in simulation or on physical hardware:

    #### Simulation Environment
    - **Minimum**: RTX 4070 Ti with 12GB VRAM
    - **Recommended**: RTX 3090/4090 with 24GB VRAM
    - **CPU**: Intel Core i7 (13th Gen+) or AMD Ryzen 9
    - **RAM**: 64GB DDR5
    - **OS**: Ubuntu 22.04 LTS

    #### Physical Implementation (Optional)
    - **Robot Platform**: Unitree H1 Humanoid or G1 Humanoid
    - **Edge Computing**: NVIDIA Jetson Orin Nano (8GB) or Orin NX (16GB)
    - **Sensors**: Intel RealSense D435i, IMU, ReSpeaker microphone array

    ### Software Stack

    - **ROS 2**: Robot Operating System Humble Hawksbill
    - **Isaac Sim**: For photorealistic simulation and synthetic data
    - **Isaac ROS**: Hardware-accelerated perception and navigation
    - **Gazebo**: Physics simulation environment
    - **OpenAI Whisper**: Speech recognition
    - **Large Language Model**: GPT-4, Claude, or equivalent for cognitive planning
    - **Computer Vision**: Isaac ROS Stereo DNN or custom models

    ## System Components

    ### 1. Voice Command Processing System

    The voice command processing system handles natural language input:

    ```python
    import whisper
    import torch
    import pyaudio
    import numpy as np
    from typing import Dict, List, Optional
    from dataclasses import dataclass

    @dataclass
    class VoiceCommand:
        """Represents a processed voice command"""
        original_text: str
        processed_text: str
        intent: str
        entities: Dict[str, str]
        confidence: float

    class VoiceCommandProcessor:
        def __init__(self):
            """Initialize voice command processing system"""
            # Load Whisper model
            self.whisper_model = whisper.load_model("base")

            # Initialize audio recording
            self.audio = pyaudio.PyAudio()
            self.rate = 16000
            self.chunk = 1024
            self.format = pyaudio.paInt16
            self.channels = 1

            # Command patterns for intent recognition
            self.command_patterns = {
                "navigation": [
                    r"move to (.+)",
                    r"go to (.+)",
                    r"navigate to (.+)",
                    r"walk to (.+)"
                ],
                "object_interaction": [
                    r"pick up (.+)",
                    r"grasp (.+)",
                    r"take (.+)",
                    r"get (.+)"
                ],
                "object_placement": [
                    r"put (.+) on (.+)",
                    r"place (.+) on (.+)",
                    r"move (.+) to (.+)"
                ],
                "object_search": [
                    r"find (.+)",
                    r"locate (.+)",
                    r"where is (.+)",
                    r"search for (.+)"
                ]
            }

        def record_voice_command(self, duration: int = 5) -> str:
            """Record voice command from microphone"""
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )

            frames = []
            for _ in range(0, int(self.rate / self.chunk * duration)):
                data = stream.read(self.chunk)
                frames.append(data)

            stream.stop_stream()
            stream.close()

            # Convert to numpy array
            audio_data = b''.join(frames)
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_np = audio_np.astype(np.float32) / 32768.0

            # Transcribe using Whisper
            result = self.whisper_model.transcribe(audio_np, fp16=False)
            return result["text"].strip()

        def process_command(self, text: str) -> Optional[VoiceCommand]:
            """Process text command and extract intent and entities"""
            text_lower = text.lower()

            # Identify intent
            intent = "unknown"
            entities = {}

            for intent_type, patterns in self.command_patterns.items():
                for pattern in patterns:
                    import re
                    match = re.search(pattern, text_lower)
                    if match:
                        intent = intent_type
                        # Extract entities based on pattern groups
                        groups = match.groups()
                        if intent_type == "navigation":
                            entities["destination"] = groups[0]
                        elif intent_type == "object_interaction":
                            entities["object"] = groups[0]
                        elif intent_type == "object_placement":
                            entities["object"] = groups[0]
                            entities["destination"] = groups[1]
                        elif intent_type == "object_search":
                            entities["object"] = groups[0]
                        break
                if intent != "unknown":
                    break

            return VoiceCommand(
                original_text=text,
                processed_text=text_lower,
                intent=intent,
                entities=entities,
                confidence=0.9  # Placeholder
            )
    ```

    ### 2. Cognitive Planning System

    The cognitive planning system translates high-level commands into executable actions:

    ```python
    import openai
    import json
    from typing import Dict, List, Any
    from dataclasses import dataclass
    from enum import Enum

    class ActionType(Enum):
        NAVIGATION = "navigation"
        MANIPULATION = "manipulation"
        PERCEPTION = "perception"
        INTERACTION = "interaction"

    @dataclass
    class RobotAction:
        """Represents a single robot action"""
        action_type: ActionType
        action_name: str
        parameters: Dict[str, Any]
        description: str
        priority: int = 1

    class CognitivePlanner:
        def __init__(self, api_key: str):
            """Initialize cognitive planning system"""
            openai.api_key = api_key
            self.model = "gpt-4"

            # Define available robot actions
            self.available_actions = {
                "move_to_location": {
                    "description": "Move robot to a specific location",
                    "parameters": {
                        "location": {"type": "string", "description": "Target location"},
                        "speed": {"type": "number", "default": 0.5}
                    }
                },
                "detect_objects": {
                    "description": "Detect objects in the environment",
                    "parameters": {
                        "target_objects": {"type": "array", "items": {"type": "string"}},
                        "max_objects": {"type": "integer", "default": 10}
                    }
                },
                "navigate_to_object": {
                    "description": "Navigate to a specific object",
                    "parameters": {
                        "object_id": {"type": "string", "description": "ID of target object"},
                        "approach_distance": {"type": "number", "default": 0.5}
                    }
                },
                "pick_up_object": {
                    "description": "Pick up an object",
                    "parameters": {
                        "object_id": {"type": "string", "description": "ID of object to pick up"},
                        "arm": {"type": "string", "default": "right"}
                    }
                },
                "place_object": {
                    "description": "Place object at location",
                    "parameters": {
                        "location": {"type": "string", "description": "Placement location"},
                        "arm": {"type": "string", "default": "right"}
                    }
                },
                "speak": {
                    "description": "Make robot speak",
                    "parameters": {
                        "text": {"type": "string", "description": "Text to speak"}
                    }
                }
            }

        def generate_plan(self, command: VoiceCommand, world_state: Dict[str, Any]) -> List[RobotAction]:
            """Generate execution plan from voice command"""
            prompt = self._create_planning_prompt(command, world_state)

            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    functions=self._get_function_definitions(),
                    function_call="auto"
                )

                return self._parse_response(response)
            except Exception as e:
                print(f"Error generating plan: {e}")
                return self._create_fallback_plan(command)

        def _create_planning_prompt(self, command: VoiceCommand, world_state: Dict[str, Any]) -> str:
            """Create planning prompt for LLM"""
            return f"""
            Given the following voice command: "{command.original_text}"

            Command intent: {command.intent}
            Command entities: {command.entities}

            Current world state:
            - Robot position: {world_state.get('robot_position', 'unknown')}
            - Known objects: {list(world_state.get('objects', {}).keys())}
            - Available locations: {world_state.get('locations', [])}
            - Robot capabilities: {world_state.get('capabilities', {})}

            Please decompose this command into a sequence of specific robot actions.
            Consider spatial relationships, object properties, and task dependencies.

            Respond with the action sequence in JSON format.
            """

        def _get_function_definitions(self) -> List[Dict]:
            """Get function definitions for LLM"""
            functions = []
            for action_name, action_info in self.available_actions.items():
                function_def = {
                    "name": action_name,
                    "description": action_info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": action_info["parameters"],
                        "required": []
                    }
                }

                # Determine required parameters
                for param_name, param_info in action_info["parameters"].items():
                    if "default" not in param_info:
                        function_def["parameters"]["required"].append(param_name)

                functions.append(function_def)

            return functions

        def _parse_response(self, response) -> List[RobotAction]:
            """Parse LLM response into robot actions"""
            actions = []

            if hasattr(response.choices[0].message, 'function_call'):
                call = response.choices[0].message.function_call
                action = self._create_action_from_function_call(call)
                if action:
                    actions.append(action)
            elif hasattr(response.choices[0].message, 'tool_calls'):
                for tool_call in response.choices[0].message.tool_calls:
                    action = self._create_action_from_function_call(tool_call.function)
                    if action:
                        actions.append(action)

            return actions

        def _create_action_from_function_call(self, function_call) -> Optional[RobotAction]:
            """Create RobotAction from function call"""
            try:
                action_name = function_call.name
                arguments = json.loads(function_call.arguments)

                action_type_map = {
                    "move_to_location": ActionType.NAVIGATION,
                    "detect_objects": ActionType.PERCEPTION,
                    "navigate_to_object": ActionType.NAVIGATION,
                    "pick_up_object": ActionType.MANIPULATION,
                    "place_object": ActionType.MANIPULATION,
                    "speak": ActionType.INTERACTION
                }

                action_type = action_type_map.get(action_name, ActionType.NAVIGATION)

                return RobotAction(
                    action_type=action_type,
                    action_name=action_name,
                    parameters=arguments,
                    description=f"{action_name} with params: {arguments}"
                )
            except Exception as e:
                print(f"Error parsing function call: {e}")
                return None
    ```

    ### 3. ROS 2 Integration Layer

    The ROS 2 integration layer handles communication between components:

    ```python
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from geometry_msgs.msg import Twist, Pose
    from sensor_msgs.msg import Image, CameraInfo
    from builtin_interfaces.msg import Duration
    from rclpy.action import ActionClient
    from nav2_msgs.action import NavigateToPose
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    class AutonomousHumanoidNode(Node):
        def __init__(self):
            super().__init__('autonomous_humanoid_node')

            # Initialize cognitive components
            self.voice_processor = VoiceCommandProcessor()
            self.cognitive_planner = CognitivePlanner(api_key="your-api-key")

            # Publishers and subscribers
            self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
            self.joint_trajectory_pub = self.create_publisher(
                JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10
            )
            self.speech_pub = self.create_publisher(String, '/robot_speech', 10)

            # Subscribers for sensor data
            self.camera_sub = self.create_subscription(
                Image, '/camera/rgb/image_raw', self.camera_callback, 10
            )
            self.imu_sub = self.create_subscription(
                Imu, '/imu/data', self.imu_callback, 10
            )

            # Action clients
            self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

            # World state
            self.world_state = {
                "robot_position": (0, 0, 0),
                "objects": {},
                "locations": ["kitchen", "living_room", "bedroom", "office"],
                "capabilities": {
                    "navigation": {"max_speed": 0.5, "min_turn_radius": 0.3},
                    "manipulation": {"max_payload": 2.0, "reach_distance": 1.2}
                }
            }

            # Execution state
            self.current_plan = []
            self.plan_index = 0
            self.is_executing = False

            # Timer for continuous processing
            self.process_timer = self.create_timer(1.0, self.process_commands)

            self.get_logger().info("Autonomous Humanoid Node initialized")

        def process_commands(self):
            """Main processing loop for voice commands"""
            if not self.is_executing:
                # Record and process voice command
                try:
                    command_text = self.voice_processor.record_voice_command(duration=3)
                    if command_text.strip():
                        self.get_logger().info(f"Recognized: {command_text}")

                        # Process command
                        voice_command = self.voice_processor.process_command(command_text)
                        if voice_command:
                            # Generate plan
                            plan = self.cognitive_planner.generate_plan(voice_command, self.world_state)
                            self.execute_plan(plan)
                except Exception as e:
                    self.get_logger().error(f"Error processing command: {e}")

        def execute_plan(self, plan: List[RobotAction]):
            """Execute a sequence of robot actions"""
            if not plan:
                return

            self.current_plan = plan
            self.plan_index = 0
            self.is_executing = True

            self.get_logger().info(f"Starting execution of plan with {len(plan)} actions")
            self.execute_next_action()

        def execute_next_action(self):
            """Execute the next action in the plan"""
            if (not self.current_plan or
                self.plan_index >= len(self.current_plan) or
                not self.is_executing):
                # Plan completed
                self.is_executing = False
                self.current_plan = []
                self.plan_index = 0

                # Announce completion
                completion_msg = String()
                completion_msg.data = "Task completed successfully"
                self.speech_pub.publish(completion_msg)
                return

            action = self.current_plan[self.plan_index]
            self.get_logger().info(f"Executing action {self.plan_index + 1}: {action.action_name}")

            # Execute based on action type
            if action.action_type == ActionType.NAVIGATION:
                self.execute_navigation_action(action)
            elif action.action_type == ActionType.MANIPULATION:
                self.execute_manipulation_action(action)
            elif action.action_type == ActionType.PERCEPTION:
                self.execute_perception_action(action)
            elif action.action_type == ActionType.INTERACTION:
                self.execute_interaction_action(action)

        def execute_navigation_action(self, action):
            """Execute navigation action"""
            if action.action_name == "move_to_location":
                location = action.parameters.get("location", "unknown")

                # Look up coordinates for named location
                target_pose = self.get_location_pose(location)
                if target_pose:
                    self.send_navigation_goal(target_pose)
                else:
                    self.get_logger().error(f"Unknown location: {location}")
                    self.plan_index += 1
                    if self.is_executing:
                        self.execute_next_action()

        def get_location_pose(self, location_name: str) -> Optional[Pose]:
            """Get pose for named location"""
            location_map = {
                "kitchen": Pose(position=Point(x=3.0, y=1.0, z=0.0)),
                "living_room": Pose(position=Point(x=1.0, y=2.0, z=0.0)),
                "bedroom": Pose(position=Point(x=4.0, y=3.0, z=0.0)),
                "office": Pose(position=Point(x=2.0, y=4.0, z=0.0)),
            }
            return location_map.get(location_name)

        def send_navigation_goal(self, pose: Pose):
            """Send navigation goal to Nav2"""
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose.header.frame_id = 'map'
            goal_msg.pose.pose = pose

            self.nav_client.wait_for_server()
            future = self.nav_client.send_goal_async(goal_msg)
            future.add_done_callback(self.navigation_result_callback)

        def navigation_result_callback(self, future):
            """Handle navigation result"""
            result = future.result()
            if result.status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info('Navigation succeeded')
                self.plan_index += 1
                if self.is_executing:
                    self.execute_next_action()
            else:
                self.get_logger().error('Navigation failed')
                self.is_executing = False

        def execute_manipulation_action(self, action):
            """Execute manipulation action"""
            if action.action_name == "pick_up_object":
                # Implementation would interface with manipulation controllers
                object_id = action.parameters.get("object_id", "unknown")
                self.get_logger().info(f"Attempting to pick up {object_id}")

                # Simulate manipulation completion
                self.plan_index += 1
                if self.is_executing:
                    self.execute_next_action()

        def execute_perception_action(self, action):
            """Execute perception action"""
            if action.action_name == "detect_objects":
                target_objects = action.parameters.get("target_objects", [])
                self.get_logger().info(f"Detecting objects: {target_objects}")

                # In real implementation, this would trigger object detection
                # For now, simulate detection
                detected_objects = self.simulate_object_detection(target_objects)

                # Update world state with detected objects
                for obj in detected_objects:
                    self.world_state["objects"][obj["id"]] = obj["properties"]

                self.plan_index += 1
                if self.is_executing:
                    self.execute_next_action()

        def execute_interaction_action(self, action):
            """Execute interaction action"""
            if action.action_name == "speak":
                text = action.parameters.get("text", "")
                self.get_logger().info(f"Robot says: {text}")

                speech_msg = String()
                speech_msg.data = text
                self.speech_pub.publish(speech_msg)

            self.plan_index += 1
            if self.is_executing:
                self.execute_next_action()

        def camera_callback(self, msg):
            """Handle camera data for perception"""
            # Process camera data for object detection
            pass

        def imu_callback(self, msg):
            """Handle IMU data for balance and orientation"""
            # Update robot orientation in world state
            self.world_state["robot_orientation"] = msg.orientation
    ```

    ### 4. Simulation Environment Setup

    The simulation environment provides a safe testing ground:

    ```xml
    <!-- autonomous_humanoid.world -->
    <?xml version="1.0" ?>
    <sdf version="1.7">
      <world name="autonomous_humanoid_world">
        <!-- Physics -->
        <physics name="humanoid_physics" type="ode">
          <max_step_size>0.001</max_step_size>
          <real_time_update_rate>1000.0</real_time_update_rate>
          <gravity>0 0 -9.8</gravity>
          <ode>
            <solver>
              <type>quick</type>
              <iters>100</iters>
              <sor>1.0</sor>
            </solver>
            <constraints>
              <cfm>1e-6</cfm>
              <erp>0.05</erp>
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
                  <size>20 20</size>
                </plane>
              </geometry>
            </collision>
            <visual name="visual">
              <geometry>
                <plane>
                  <normal>0 0 1</normal>
                  <size>20 20</size>
                </plane>
              </geometry>
              <material>
                <ambient>0.7 0.7 0.7 1</ambient>
                <diffuse>0.7 0.7 0.7 1</diffuse>
              </material>
            </visual>
          </link>
        </model>

        <!-- Rooms -->
        <model name="kitchen_area">
          <static>true</static>
          <link name="kitchen_floor">
            <collision name="collision">
              <geometry>
                <box>
                  <size>4 4 0.1</size>
                </box>
              </geometry>
              <pose>2 2 0.05 0 0 0</pose>
            </collision>
            <visual name="visual">
              <geometry>
                <box>
                  <size>4 4 0.1</size>
                </box>
              </geometry>
              <pose>2 2 0.05 0 0 0</pose>
              <material>
                <ambient>0.8 0.6 0.2 1</ambient>
                <diffuse>0.8 0.6 0.2 1</diffuse>
              </material>
            </visual>
          </link>
        </model>

        <model name="living_room_area">
          <static>true</static>
          <link name="living_room_floor">
            <collision name="collision">
              <geometry>
                <box>
                  <size>4 4 0.1</size>
                </box>
              </geometry>
              <pose>2 -2 0.05 0 0 0</pose>
            </collision>
            <visual name="visual">
              <geometry>
                <box>
                  <size>4 4 0.1</size>
                </box>
              </geometry>
              <pose>2 -2 0.05 0 0 0</pose>
              <material>
                <ambient>0.2 0.6 0.8 1</ambient>
                <diffuse>0.2 0.6 0.8 1</diffuse>
              </material>
            </visual>
          </link>
        </model>

        <!-- Objects -->
        <model name="red_cup">
          <pose>2.5 2.5 0.5 0 0 0</pose>
          <link name="cup_link">
            <collision name="collision">
              <geometry>
                <cylinder>
                  <radius>0.05</radius>
                  <length>0.1</length>
                </cylinder>
              </geometry>
            </collision>
            <visual name="visual">
              <geometry>
                <cylinder>
                  <radius>0.05</radius>
                  <length>0.1</length>
                </cylinder>
              </geometry>
              <material>
                <ambient>0.8 0.2 0.2 1</ambient>
                <diffuse>0.8 0.2 0.2 1</diffuse>
              </material>
            </visual>
            <inertial>
              <mass>0.1</mass>
              <inertia>
                <ixx>0.001</ixx>
                <ixy>0</ixy>
                <ixz>0</ixz>
                <iyy>0.001</iyy>
                <iyz>0</iyz>
                <izz>0.001</izz>
              </inertia>
            </inertial>
          </link>
        </model>

        <!-- Include humanoid robot -->
        <include>
          <uri>model://simple_humanoid</uri>
          <pose>0 0 1 0 0 0</pose>
        </include>
      </world>
    </sdf>
    ```

    ## Implementation Phases

    ### Phase 1: Basic Voice Command Recognition (Week 1)
    - Implement Whisper-based speech recognition
    - Create basic command processing
    - Test with simple navigation commands
    - Integrate with ROS 2 messaging

    ### Phase 2: Cognitive Planning Integration (Week 2)
    - Integrate LLM for task decomposition
    - Implement plan validation and safety checks
    - Test with complex multi-step commands
    - Add error handling and recovery

    ### Phase 3: Navigation and Perception (Week 3)
    - Integrate Nav2 for path planning
    - Implement object detection and recognition
    - Test navigation in simulated environment
    - Add obstacle avoidance

    ### Phase 4: Manipulation and Integration (Week 4)
    - Implement object manipulation capabilities
    - Integrate all components into complete system
    - Test end-to-end functionality
    - Optimize performance and reliability

    ## Testing Scenarios

    ### Basic Navigation Test
    ```
    Command: "Go to the kitchen"
    Expected: Robot navigates to kitchen area
    ```

    ### Object Interaction Test
    ```
    Command: "Find the red cup and bring it to me"
    Expected: Robot detects red cup, navigates to it, picks it up, brings to user
    ```

    ### Complex Task Test
    ```
    Command: "Go to the living room, find the book on the table, and bring it to the kitchen"
    Expected: Multi-step execution with navigation, object detection, manipulation, and navigation
    ```

    ### Error Handling Test
    ```
    Command: "Go to the moon"
    Expected: Robot responds appropriately to impossible command
    ```

    ## Evaluation Criteria

    ### Functionality (40%)
    - Voice command recognition accuracy (>80%)
    - Successful task completion rate (>70%)
    - Proper error handling and recovery
    - Safety constraint adherence

    ### Integration (30%)
    - Seamless component interaction
    - Robust ROS 2 communication
    - Proper state management
    - Real-time performance

    ### Innovation (20%)
    - Creative problem-solving approaches
    - Efficient resource utilization
    - Novel interaction methods
    - Advanced planning techniques

    ### Documentation (10%)
    - Clear code documentation
    - Comprehensive testing results
    - Performance analysis
    - Future improvement suggestions

    ## Advanced Features (Optional)

    ### Multi-Modal Interaction
    - Combine speech with gesture recognition
    - Visual feedback through robot expressions
    - Haptic feedback for manipulation tasks

    ### Learning and Adaptation
    - Remember user preferences
    - Adapt to environment changes
    - Learn from interaction patterns

    ### Social Interaction
    - Natural conversation flow
    - Context-aware responses
    - Emotional intelligence simulation

    ## Deployment Considerations

    ### Simulation to Real World Transfer
    - Domain randomization for robust perception
    - Sim-to-real transfer techniques
    - Reality gap minimization strategies

    ### Safety and Ethics
    - Emergency stop mechanisms
    - Privacy considerations for voice data
    - Ethical AI usage guidelines

    ### Scalability
    - Multi-robot coordination
    - Cloud-based processing options
    - Distributed system architecture

    ## Conclusion

    The Autonomous Humanoid capstone project represents the integration of all Physical AI and humanoid robotics concepts covered in this course. Successfully completing this project demonstrates mastery of:

    - ROS 2 for robot communication and control
    - Simulation environments for testing and development
    - AI systems for perception and decision-making
    - Natural language processing for human-robot interaction
    - System integration and real-time operation

    This project serves as a foundation for advanced research and development in Physical AI and humanoid robotics, preparing students for cutting-edge work in this rapidly evolving field.
  </div>
  <div className="urdu">
    # کیپ اسٹون پروجیکٹ: خود مختار ہیومنائیڈ

    ## تعارف

    خود مختار ہیومنائیڈ کیپ اسٹون پروجیکٹ اس کورس میں سیکھے گئے تمام تصورات کا نقطہ عروج ہے۔ یہ جامع پروجیکٹ تمام چار ماڈیولز - روبوٹک نروس سسٹم (ROS 2)، ڈیجیٹل ٹوئن (Gazebo اور Unity)، AI-Robot Brain (NVIDIA Isaac™)، اور Vision-Language-Action (VLA) - کو مربوط کرتا ہے تاکہ ایک مکمل خود مختار ہیومنائیڈ روبوٹ بنایا جا سکے جو صوتی احکامات وصول کرنے، راستوں کی منصوبہ بندی کرنے، رکاوٹوں سے بچتے ہوئے نیویگیٹ کرنے، کمپیوٹر ویژن کا استعمال کرتے ہوئے اشیاء کی شناخت کرنے، اور ان کو مناسب طریقے سے ہیرا پھیری کرنے کے قابل ہو۔

    یہ پروجیکٹ Physical AI کی صلاحیتوں کے مکمل اسپیکٹرم کا مظاہرہ کرتا ہے، نچلی سطح کے کنٹرول سے لے کر اعلیٰ سطحی علمی منصوبہ بندی تک، اور ایک حقیقت پسندانہ منظرنامہ فراہم کرتا ہے جو تمام کورس اجزاء کے انضمام کو ظاہر کرتا ہے۔

    ## پروجیکٹ کا جائزہ

    ### پروجیکٹ کا مقصد

    ایک خود مختار ہیومنائیڈ روبوٹ سسٹم تیار کریں جو درج ذیل کام کر سکے:
    1.  OpenAI Whisper کا استعمال کرتے ہوئے صوتی احکامات وصول اور سمجھ سکے۔
    2.  LLMs کے ساتھ علمی منصوبہ بندی (cognitive planning) کا استعمال کرتے ہوئے نیویگیشن راستوں کی منصوبہ بندی کر سکے۔
    3.  رکاوٹوں سے بچتے ہوئے ماحول میں نیویگیٹ کر سکے۔
    4.  کمپیوٹر ویژن کا استعمال کرتے ہوئے اشیاء کی شناخت اور ان کا پتہ لگا سکے۔
    5.  کام مکمل کرنے کے لیے اشیاء کو ہیرا پھیری کر سکے۔
    6.  تقریر کے ذریعے انسانوں کے ساتھ قدرتی طور پر بات چیت کر سکے۔

    ### سسٹم آرکیٹیکچر

    خود مختار ہیومنائیڈ سسٹم متعدد ٹیکنالوجیز کو مربوط کرتا ہے:

    ```
    صارف کا صوتی کمانڈ → Whisper ASR → LLM علمی منصوبہ بندی → ROS 2 کنٹرول → جسمانی/ورچوئل روبوٹ
           ↓                    ↓                ↓                    ↓              ↓
       قدرتی زبان       تقریر سے متن        کام کی تقسیم       عمل کا نفاذ     ماحول
       کی سمجھ          کی پروسیسنگ        عمل کی ترتیب         انٹرفیس           تعامل
    ```

    ## تکنیکی تقاضے

    ### ہارڈویئر کے تقاضے

    پروجیکٹ کو سیمولیشن یا فزیکل ہارڈویئر پر لاگو کیا جا سکتا ہے:

    #### سیمولیشن ماحول
    *   **کم از کم**: RTX 4070 Ti جس میں 12GB VRAM ہو
    *   **تجویز کردہ**: RTX 3090/4090 جس میں 24GB VRAM ہو
    *   **CPU**: Intel Core i7 (13th Gen+) یا AMD Ryzen 9
    *   **RAM**: 64GB DDR5
    *   **OS**: Ubuntu 22.04 LTS

    #### جسمانی نفاذ (اختیاری)
    *   **روبوٹ پلیٹ فارم**: Unitree H1 Humanoid یا G1 Humanoid
    *   **ایج کمپیوٹنگ**: NVIDIA Jetson Orin Nano (8GB) یا Orin NX (16GB)
    *   **سینسرز**: Intel RealSense D435i، IMU، ReSpeaker مائیکروفون سرنی

    ### سافٹ ویئر اسٹیک

    *   **ROS 2**: روبوٹ آپریٹنگ سسٹم Humble Hawksbill
    *   **Isaac Sim**: فوٹو ریئلسٹک سیمولیشن اور مصنوعی ڈیٹا کے لیے
    *   **Isaac ROS**: ہارڈویئر سے تیز رفتار پرسیپشن اور نیویگیشن
    *   **Gazebo**: فزکس سیمولیشن ماحول
    *   **OpenAI Whisper**: تقریر کی شناخت
    *   **Large Language Model**: GPT-4، Claude، یا علمی منصوبہ بندی کے لیے مساوی
    *   **کمپیوٹر ویژن**: Isaac ROS Stereo DNN یا کسٹم ماڈلز

    ## سسٹم کے اجزاء

    ### 1. وائس کمانڈ پروسیسنگ سسٹم

    وائس کمانڈ پروسیسنگ سسٹم قدرتی زبان کے ان پٹ کو ہینڈل کرتا ہے:

    ```python
    import whisper
    import torch
    import pyaudio
    import numpy as np
    from typing import Dict, List, Optional
    from dataclasses import dataclass

    @dataclass
    class VoiceCommand:
        """ایک پروسیس شدہ وائس کمانڈ کی نمائندگی کرتا ہے"""
        original_text: str
        processed_text: str
        intent: str
        entities: Dict[str, str]
        confidence: float

    class VoiceCommandProcessor:
        def __init__(self):
            """وائس کمانڈ پروسیسنگ سسٹم کو شروع کریں"""
            # Whisper ماڈل لوڈ کریں
            self.whisper_model = whisper.load_model("base")

            # آڈیو ریکارڈنگ شروع کریں
            self.audio = pyaudio.PyAudio()
            self.rate = 16000
            self.chunk = 1024
            self.format = pyaudio.paInt16
            self.channels = 1

            # ارادے کی شناخت کے لیے کمانڈ کے نمونے
            self.command_patterns = {
                "navigation": [
                    r"move to (.+)",
                    r"go to (.+)",
                    r"navigate to (.+)",
                    r"walk to (.+)"
                ],
                "object_interaction": [
                    r"pick up (.+)",
                    r"grasp (.+)",
                    r"take (.+)",
                    r"get (.+)"
                ],
                "object_placement": [
                    r"put (.+) on (.+)",
                    r"place (.+) on (.+)",
                    r"move (.+) to (.+)"
                ],
                "object_search": [
                    r"find (.+)",
                    r"locate (.+)",
                    r"where is (.+)",
                    r"search for (.+)"
                ]
            }

        def record_voice_command(self, duration: int = 5) -> str:
            """مائیکروفون سے وائس کمانڈ ریکارڈ کریں"""
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )

            frames = []
            for _ in range(0, int(self.rate / self.chunk * duration)):
                data = stream.read(self.chunk)
                frames.append(data)

            stream.stop_stream()
            stream.close()

            # numpy array میں تبدیل کریں
            audio_data = b''.join(frames)
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_np = audio_np.astype(np.float32) / 32768.0

            # Whisper کا استعمال کرتے ہوئے ٹرانسکرائب کریں
            result = self.whisper_model.transcribe(audio_np, fp16=False)
            return result["text"].strip()

        def process_command(self, text: str) -> Optional[VoiceCommand]:
            """ٹیکسٹ کمانڈ پر کارروائی کریں اور ارادے اور اکائیوں کو نکالیں"""
            text_lower = text.lower()

            # ارادے کی شناخت کریں
            intent = "unknown"
            entities = {}

            for intent_type, patterns in self.command_patterns.items():
                for pattern in patterns:
                    import re
                    match = re.search(pattern, text_lower)
                    if match:
                        intent = intent_type
                        # پیٹرن گروپس کی بنیاد پر اکائیوں کو نکالیں
                        groups = match.groups()
                        if intent_type == "navigation":
                            entities["destination"] = groups[0]
                        elif intent_type == "object_interaction":
                            entities["object"] = groups[0]
                        elif intent_type == "object_placement":
                            entities["object"] = groups[0]
                            entities["destination"] = groups[1]
                        elif intent_type == "object_search":
                            entities["object"] = groups[0]
                        break
                if intent != "unknown":
                    break

            return VoiceCommand(
                original_text=text,
                processed_text=text_lower,
                intent=intent,
                entities=entities,
                confidence=0.9  # پلیس ہولڈر
            )
    ```

    ### 2. علمی منصوبہ بندی کا نظام

    علمی منصوبہ بندی کا نظام اعلیٰ سطحی احکامات کو قابل عمل ایکشنز میں تبدیل کرتا ہے:

    ```python
    import openai
    import json
    from typing import Dict, List, Any
    from dataclasses import dataclass
    from enum import Enum

    class ActionType(Enum):
        NAVIGATION = "navigation"
        MANIPULATION = "manipulation"
        PERCEPTION = "perception"
        INTERACTION = "interaction"

    @dataclass
    class RobotAction:
        """ایک ہی روبوٹ ایکشن کی نمائندگی کرتا ہے"""
        action_type: ActionType
        action_name: str
        parameters: Dict[str, Any]
        description: str
        priority: int = 1

    class CognitivePlanner:
        def __init__(self, api_key: str):
            """علمی منصوبہ بندی کے نظام کو شروع کریں"""
            openai.api_key = api_key
            self.model = "gpt-4"

            # دستیاب روبوٹ ایکشنز کی وضاحت کریں
            self.available_actions = {
                "move_to_location": {
                    "description": "روبوٹ کو ایک مخصوص مقام پر منتقل کریں",
                    "parameters": {
                        "location": {"type": "string", "description": "ہدف کا مقام"},
                        "speed": {"type": "number", "default": 0.5}
                    }
                },
                "detect_objects": {
                    "description": "ماحول میں اشیاء کا پتہ لگائیں",
                    "parameters": {
                        "target_objects": {"type": "array", "items": {"type": "string"}},
                        "max_objects": {"type": "integer", "default": 10}
                    }
                },
                "navigate_to_object": {
                    "description": "ایک مخصوص آبجیکٹ پر جائیں",
                    "parameters": {
                        "object_id": {"type": "string", "description": "ہدف آبجیکٹ کی ID"},
                        "approach_distance": {"type": "number", "default": 0.5}
                    }
                },
                "pick_up_object": {
                    "description": "ایک آبجیکٹ اٹھائیں",
                    "parameters": {
                        "object_id": {"type": "string", "description": "اٹھانے والے آبجیکٹ کی ID"},
                        "arm": {"type": "string", "default": "right"}
                    }
                },
                "place_object": {
                    "description": "آبجیکٹ کو مقام پر رکھیں",
                    "parameters": {
                        "location": {"type": "string", "description": "پلیسمینٹ کا مقام"},
                        "arm": {"type": "string", "default": "right"}
                    }
                },
                "speak": {
                    "description": "روبوٹ کو بات کروائیں",
                    "parameters": {
                        "text": {"type": "string", "description": "بولنے کا متن"}
                    }
                }
            }

        def generate_plan(self, command: VoiceCommand, world_state: Dict[str, Any]) -> List[RobotAction]:
            """وائس کمانڈ سے عمل درآمد کا منصوبہ بنائیں"""
            prompt = self._create_planning_prompt(command, world_state)

            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    functions=self._get_function_definitions(),
                    function_call="auto"
                )

                return self._parse_response(response)
            except Exception as e:
                print(f"منصوبہ بنانے میں غلطی: {e}")
                return self._create_fallback_plan(command)

        def _create_planning_prompt(self, command: VoiceCommand, world_state: Dict[str, Any]) -> str:
            """LLM کے لیے منصوبہ بندی کا پرامپٹ بنائیں"""
            return f"""
            درج ذیل صوتی کمانڈ کو دیکھتے ہوئے: "{command.original_text}"

            کمانڈ کا ارادہ: {command.intent}
            کمانڈ کی اکائیاں: {command.entities}

            موجودہ دنیا کی حالت:
            - روبوٹ کی پوزیشن: {world_state.get('robot_position', 'unknown')}
            - معلوم اشیاء: {list(world_state.get('objects', {}).keys())}
            - دستیاب مقامات: {world_state.get('locations', [])}
            - روبوٹ کی صلاحیتیں: {world_state.get('capabilities', {})}

            براہ کرم اس کمانڈ کو مخصوص روبوٹ ایکشنز کے سلسلے میں توڑ دیں۔
            مقامی تعلقات، آبجیکٹ کی خصوصیات، اور ٹاسک کے انحصار پر غور کریں۔

            JSON فارمیٹ میں ایکشن کا سلسلہ جواب دیں۔
            """

        def _get_function_definitions(self) -> List[Dict]:
            """LLM کے لیے فنکشن کی تعریفیں حاصل کریں"""
            functions = []
            for action_name, action_info in self.available_actions.items():
                function_def = {
                    "name": action_name,
                    "description": action_info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": action_info["parameters"],
                        "required": []
                    }
                }

                # مطلوبہ پیرامیٹرز کا تعین کریں
                for param_name, param_info in action_info["parameters"].items():
                    if "default" not in param_info:
                        function_def["parameters"]["required"].append(param_name)

                functions.append(function_def)

            return functions

        def _parse_response(self, response) -> List[RobotAction]:
            """LLM جواب کو روبوٹ ایکشنز میں پارس کریں"""
            actions = []

            if hasattr(response.choices[0].message, 'function_call'):
                call = response.choices[0].message.function_call
                action = self._create_action_from_function_call(call)
                if action:
                    actions.append(action)
            elif hasattr(response.choices[0].message, 'tool_calls'):
                for tool_call in response.choices[0].message.tool_calls:
                    action = self._create_action_from_function_call(tool_call.function)
                    if action:
                        actions.append(action)

            return actions

        def _create_action_from_function_call(self, function_call) -> Optional[RobotAction]:
            """فنکشن کال سے RobotAction بنائیں"""
            try:
                action_name = function_call.name
                arguments = json.loads(function_call.arguments)

                action_type_map = {
                    "move_to_location": ActionType.NAVIGATION,
                    "detect_objects": ActionType.PERCEPTION,
                    "navigate_to_object": ActionType.NAVIGATION,
                    "pick_up_object": ActionType.MANIPULATION,
                    "place_object": ActionType.MANIPULATION,
                    "speak": ActionType.INTERACTION
                }

                action_type = action_type_map.get(action_name, ActionType.NAVIGATION)

                return RobotAction(
                    action_type=action_type,
                    action_name=action_name,
                    parameters=arguments,
                    description=f"{action_name} params کے ساتھ: {arguments}"
                )
            except Exception as e:
                print(f"فنکشن کال کو پارس کرنے میں غلطی: {e}")
                return None
    ```

    ### 3. ROS 2 انٹیگریشن لیئر

    ROS 2 انٹیگریشن لیئر اجزاء کے درمیان مواصلات کو ہینڈل کرتی ہے:

    ```python
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from geometry_msgs.msg import Twist, Pose
    from sensor_msgs.msg import Image, CameraInfo
    from builtin_interfaces.msg import Duration
    from rclpy.action import ActionClient
    from nav2_msgs.action import NavigateToPose
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    class AutonomousHumanoidNode(Node):
        def __init__(self):
            super().__init__('autonomous_humanoid_node')

            # علمی اجزاء کو شروع کریں
            self.voice_processor = VoiceCommandProcessor()
            self.cognitive_planner = CognitivePlanner(api_key="آپ کی API key")

            # پبلشرز اور سبسکرائبرز
            self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
            self.joint_trajectory_pub = self.create_publisher(
                JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10
            )
            self.speech_pub = self.create_publisher(String, '/robot_speech', 10)

            # سینسر ڈیٹا کے لیے سبسکرائبرز
            self.camera_sub = self.create_subscription(
                Image, '/camera/rgb/image_raw', self.camera_callback, 10
            )
            self.imu_sub = self.create_subscription(
                Imu, '/imu/data', self.imu_callback, 10
            )

            # ایکشن کلائنٹس
            self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

            # دنیا کی حالت
            self.world_state = {
                "robot_position": (0, 0, 0),
                "objects": {},
                "locations": ["باورچی خانہ", "بیٹھک", "سونے کا کمرہ", "دفتر"],
                "capabilities": {
                    "navigation": {"زیادہ سے زیادہ رفتار": 0.5, "کم از کم موڑ کا رداس": 0.3},
                    "manipulation": {"زیادہ سے زیادہ پے لوڈ": 2.0, "پہنچنے کا فاصلہ": 1.2}
                }
            }

            # عمل درآمد کی حالت
            self.current_plan = []
            self.plan_index = 0
            self.is_executing = False

            # مسلسل پروسیسنگ کے لیے ٹائمر
            self.process_timer = self.create_timer(1.0, self.process_commands)

            self.get_logger().info("خود مختار ہیومنائیڈ نوڈ شروع ہو گیا")

        def process_commands(self):
            """وائس کمانڈز کے لیے مرکزی پروسیسنگ لوپ"""
            if not self.is_executing:
                # وائس کمانڈ ریکارڈ اور پروسیس کریں
                try:
                    command_text = self.voice_processor.record_voice_command(duration=3)
                    if command_text.strip():
                        self.get_logger().info(f"تسلیم شدہ: {command_text}")

                        # کمانڈ پروسیس کریں
                        voice_command = self.voice_processor.process_command(command_text)
                        if voice_command:
                            # منصوبہ بنائیں
                            plan = self.cognitive_planner.generate_plan(voice_command, self.world_state)
                            self.execute_plan(plan)
                except Exception as e:
                    self.get_logger().error(f"کمانڈ پروسیس کرنے میں غلطی: {e}")

        def execute_plan(self, plan: List[RobotAction]):
            """روبوٹ ایکشنز کے سلسلے پر عمل کریں"""
            if not plan:
                return

            self.current_plan = plan
            self.plan_index = 0
            self.is_executing = True

            self.get_logger().info(f"{len(plan)} ایکشنز کے ساتھ منصوبے پر عمل درآمد شروع کر رہا ہے")
            self.execute_next_action()

        def execute_next_action(self):
            """منصوبے میں اگلے ایکشن پر عمل کریں"""
            if (not self.current_plan or
                self.plan_index >= len(self.current_plan) or
                not self.is_executing):
                # منصوبہ مکمل
                self.is_executing = False
                self.current_plan = []
                self.plan_index = 0

                # تکمیل کا اعلان کریں
                completion_msg = String()
                completion_msg.data = "کام کامیابی سے مکمل ہو گیا"
                self.speech_pub.publish(completion_msg)
                return

            action = self.current_plan[self.plan_index]
            self.get_logger().info(f"ایکشن {self.plan_index + 1} پر عمل کر رہا ہے: {action.action_name}")

            # ایکشن کی قسم کی بنیاد پر عمل کریں
            if action.action_type == ActionType.NAVIGATION:
                self.execute_navigation_action(action)
            elif action.action_type == ActionType.MANIPULATION:
                self.execute_manipulation_action(action)
            elif action.action_type == ActionType.PERCEPTION:
                self.execute_perception_action(action)
            elif action.action_type == ActionType.INTERACTION:
                self.execute_interaction_action(action)

        def execute_navigation_action(self, action):
            """نیویگیشن ایکشن پر عمل کریں"""
            if action.action_name == "move_to_location":
                location = action.parameters.get("location", "unknown")

                # نامزد مقام کے لیے کوآرڈینیٹس تلاش کریں
                target_pose = self.get_location_pose(location)
                if target_pose:
                    self.send_navigation_goal(target_pose)
                else:
                    self.get_logger().error(f"نامعلوم مقام: {location}")
                    self.plan_index += 1
                    if self.is_executing:
                        self.execute_next_action()

        def get_location_pose(self, location_name: str) -> Optional[Pose]:
            """نامزد مقام کے لیے پوز حاصل کریں"""
            location_map = {
                "kitchen": Pose(position=Point(x=3.0, y=1.0, z=0.0)),
                "living_room": Pose(position=Point(x=1.0, y=2.0, z=0.0)),
                "bedroom": Pose(position=Point(x=4.0, y=3.0, z=0.0)),
                "office": Pose(position=Point(x=2.0, y=4.0, z=0.0)),
            }
            return location_map.get(location_name)

        def send_navigation_goal(self, pose: Pose):
            """Nav2 کو نیویگیشن گول بھیجیں"""
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose.header.frame_id = 'map'
            goal_msg.pose.pose = pose

            self.nav_client.wait_for_server()
            future = self.nav_client.send_goal_async(goal_msg)
            future.add_done_callback(self.navigation_result_callback)

        def navigation_result_callback(self, future):
            """نیویگیشن نتیجہ کو ہینڈل کریں"""
            result = future.result()
            if result.status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info('نیویگیشن کامیاب رہی')
                self.plan_index += 1
                if self.is_executing:
                    self.execute_next_action()
            else:
                self.get_logger().error(f'نیویگیشن ناکام رہی: {result.status} اسٹیٹس کے ساتھ')
                self.is_executing = False

        def execute_manipulation_action(self, action):
            """مینیپولیشن ایکشن پر عمل کریں"""
            if action.action_name == "pick_up_object":
                # نفاذ مینیپولیشن کنٹرولرز کے ساتھ انٹرفیس کرے گا
                object_id = action.parameters.get("object_id", "unknown")
                self.get_logger().info(f"{object_id} اٹھانے کی کوشش کر رہا ہے")

                # مینیپولیشن کی تکمیل کو سیمولیٹ کریں
                self.plan_index += 1
                if self.is_executing:
                    self.execute_next_action()

        def execute_perception_action(self, action):
            """پرسیپشن ایکشن پر عمل کریں"""
            if action.action_name == "detect_objects":
                target_objects = action.parameters.get("target_objects", [])
                self.get_logger().info(f"اشیاء کا پتہ لگا رہا ہے: {target_objects}")

                # حقیقی نفاذ میں، یہ آبجیکٹ ڈیٹیکشن کو متحرک کرے گا
                # فی الحال، ڈیٹیکشن کو سیمولیٹ کریں
                detected_objects = self.simulate_object_detection(target_objects)

                # پتہ لگائی گئی اشیاء کے ساتھ دنیا کی حالت کو اپ ڈیٹ کریں
                for obj in detected_objects:
                    self.world_state["objects"][obj["id"]] = obj["properties"]

                self.plan_index += 1
                if self.is_executing:
                    self.execute_next_action()

        def execute_interaction_action(self, action):
            """تعامل ایکشن پر عمل کریں"""
            if action.action_name == "speak":
                text = action.parameters.get("text", "")
                self.get_logger().info(f"روبوٹ کہتا ہے: {text}")

                speech_msg = String()
                speech_msg.data = text
                self.speech_pub.publish(speech_msg)

            self.plan_index += 1
            if self.is_executing:
                self.execute_next_action()

        def camera_callback(self, msg):
            """پرسیپشن کے لیے کیمرہ ڈیٹا کو ہینڈل کریں"""
            # آبجیکٹ ڈیٹیکشن کے لیے کیمرہ ڈیٹا پر کارروائی کریں
            pass

        def imu_callback(self, msg):
            """بیلنس اور اورینٹیشن کے لیے IMU ڈیٹا کو ہینڈل کریں"""
            # دنیا کی حالت میں روبوٹ اورینٹیشن کو اپ ڈیٹ کریں
            self.world_state["robot_orientation"] = msg.orientation
    ```

    ### 4. سیمولیشن ماحولیاتی سیٹ اپ

    سیمولیشن ماحول ایک محفوظ جانچ کا میدان فراہم کرتا ہے:

    ```xml
    <!-- autonomous_humanoid.world -->
    <?xml version="1.0" ?>
    <sdf version="1.7">
      <world name="autonomous_humanoid_world">
        <!-- فزکس -->
        <physics name="humanoid_physics" type="ode">
          <max_step_size>0.001</max_step_size>
          <real_time_update_rate>1000.0</real_time_update_rate>
          <gravity>0 0 -9.8</gravity>
          <ode>
            <solver>
              <type>quick</type>
              <iters>100</iters>
              <sor>1.0</sor>
            </solver>
            <constraints>
              <cfm>1e-6</cfm>
              <erp>0.05</erp>
            </constraints>
          </ode>
        </physics>

        <!-- روشنی -->
        <light name="sun" type="directional">
          <cast_shadows>true</cast_shadows>
          <pose>0 0 10 0 0 0</pose>
          <diffuse>0.8 0.8 0.8 1</diffuse>
          <specular>0.2 0.2 0.2 1</specular>
          <direction>-0.3 0.3 -1</direction>
        </light>

        <!-- زمینی ہوائی جہاز -->
        <model name="ground_plane">
          <static>true</static>
          <link name="link">
            <collision name="collision">
              <geometry>
                <plane>
                  <normal>0 0 1</normal>
                  <size>20 20</size>
                </plane>
              </geometry>
            </collision>
            <visual name="visual">
              <geometry>
                <plane>
                  <normal>0 0 1</normal>
                  <size>20 20</size>
                </plane>
              </geometry>
              <material>
                <ambient>0.7 0.7 0.7 1</ambient>
                <diffuse>0.7 0.7 0.7 1</diffuse>
              </material>
            </visual>
          </link>
        </model>

        <!-- کمرے -->
        <model name="kitchen_area">
          <static>true</static>
          <link name="kitchen_floor">
            <collision name="collision">
              <geometry>
                <box>
                  <size>4 4 0.1</size>
                </box>
              </geometry>
              <pose>2 2 0.05 0 0 0</pose>
            </collision>
            <visual name="visual">
              <geometry>
                <box>
                  <size>4 4 0.1</size>
                </box>
              </geometry>
              <pose>2 2 0.05 0 0 0</pose>
              <material>
                <ambient>0.8 0.6 0.2 1</ambient>
                <diffuse>0.8 0.6 0.2 1</diffuse>
              </material>
            </visual>
          </link>
        </model>

        <model name="living_room_area">
          <static>true</static>
          <link name="living_room_floor">
            <collision name="collision">
              <geometry>
                <box>
                  <size>4 4 0.1</size>
                </box>
              </geometry>
              <pose>2 -2 0.05 0 0 0</pose>
            </collision>
            <visual name="visual">
              <geometry>
                <box>
                  <size>4 4 0.1</size>
                </box>
              </geometry>
              <pose>2 -2 0.05 0 0 0</pose>
              <material>
                <ambient>0.2 0.6 0.8 1</ambient>
                <diffuse>0.2 0.6 0.8 1</diffuse>
              </material>
            </visual>
          </link>
        </model>

        <!-- اشیاء -->
        <model name="red_cup">
          <pose>2.5 2.5 0.5 0 0 0</pose>
          <link name="cup_link">
            <collision name="collision">
              <geometry>
                <cylinder>
                  <radius>0.05</radius>
                  <length>0.1</length>
                </cylinder>
              </geometry>
            </collision>
            <visual name="visual">
              <geometry>
                <cylinder>
                  <radius>0.05</radius>
                  <length>0.1</length>
                </cylinder>
              </geometry>
              <material>
                <ambient>0.8 0.2 0.2 1</ambient>
                <diffuse>0.8 0.2 0.2 1</diffuse>
              </material>
            </visual>
            <inertial>
              <mass>0.1</mass>
              <inertia>
                <ixx>0.001</ixx>
                <ixy>0</ixy>
                <ixz>0</ixz>
                <iyy>0.001</iyy>
                <iyz>0</iyz>
                <izz>0.001</izz>
              </inertia>
            </inertial>
          </link>
        </model>

        <!-- ہیومنائیڈ روبوٹ شامل کریں -->
        <include>
          <uri>model://simple_humanoid</uri>
          <pose>0 0 1 0 0 0</pose>
        </include>
      </world>
    </sdf>
    ```

    ## نفاذ کے مراحل

    ### مرحلہ 1: بنیادی صوتی کمانڈ کی شناخت (ہفتہ 1)
    *   Whisper پر مبنی تقریر کی شناخت نافذ کریں۔
    *   بنیادی کمانڈ پروسیسنگ بنائیں۔
    *   سادہ نیویگیشن کمانڈز کے ساتھ جانچ کریں۔
    *   ROS 2 پیغام رسانی کے ساتھ مربوط کریں۔

    ### مرحلہ 2: علمی منصوبہ بندی کا انضمام (ہفتہ 2)
    *   ٹاسک ڈیکمپوزیشن کے لیے LLM کو مربوط کریں۔
    *   منصوبے کی توثیق اور حفاظتی جانچ نافذ کریں۔
    *   پیچیدہ کثیر مرحلہ کمانڈز کے ساتھ جانچ کریں۔
    *   غلطی کو سنبھالنا اور بازیابی شامل کریں۔

    ### مرحلہ 3: نیویگیشن اور پرسیپشن (ہفتہ 3)
    *   راستے کی منصوبہ بندی کے لیے Nav2 کو مربوط کریں۔
    *   آبجیکٹ ڈیٹیکشن اور شناخت نافذ کریں۔
    *   سیمولیٹڈ ماحول میں نیویگیشن کی جانچ کریں۔
    *   رکاوٹ سے بچاؤ شامل کریں۔

    ### مرحلہ 4: ہیرا پھیری اور انضمام (ہفتہ 4)
    *   آبجیکٹ ہیرا پھیری کی صلاحیتیں نافذ کریں۔
    *   تمام اجزاء کو مکمل نظام میں مربوط کریں۔
    *   اختتام سے اختتام تک کی فعالیت کی جانچ کریں۔
    *   کارکردگی اور وشوسنییتا کو بہتر بنائیں۔

    ## جانچ کے منظرنامے

    ### بنیادی نیویگیشن ٹیسٹ
    ```
    کمانڈ: "باورچی خانے میں جاؤ"
    متوقع: روبوٹ باورچی خانے کے علاقے میں نیویگیٹ کرتا ہے
    ```

    ### آبجیکٹ انٹریکشن ٹیسٹ
    ```
    کمانڈ: "سرخ کپ تلاش کرو اور میرے پاس لاؤ"
    متوقع: روبوٹ سرخ کپ کا پتہ لگاتا ہے، اس پر جاتا ہے، اسے اٹھاتا ہے، صارف کے پاس لاتا ہے
    ```

    ### پیچیدہ ٹاسک ٹیسٹ
    ```
    کمانڈ: "بیٹھک میں جاؤ، میز پر کتاب تلاش کرو، اور اسے باورچی خانے میں لاؤ"
    متوقع: نیویگیشن، آبجیکٹ ڈیٹیکشن، ہیرا پھیری، اور نیویگیشن کے ساتھ کثیر مرحلہ عمل درآمد
    ```

    ### غلطی کو سنبھالنے کا ٹیسٹ
    ```
    کمانڈ: "چاند پر جاؤ"
    متوقع: روبوٹ ناممکن کمانڈ کا مناسب جواب دیتا ہے
    ```

    ## تشخیص کے معیار

    ### فعالیت (40%)
    *   صوتی کمانڈ کی شناخت کی درستگی (>80%)
    *   کامیاب کام کی تکمیل کی شرح (>70%)
    *   مناسب غلطی کو سنبھالنا اور بازیابی
    *   حفاظتی رکاوٹ کی پابندی

    ### انضمام (30%)
    *   بغیر کسی رکاوٹ کے اجزاء کا تعامل
    *   مضبوط ROS 2 مواصلات
    *   مناسب حالت کا انتظام
    *   ریئل ٹائم کارکردگی

    ### جدت (20%)
    *   تخلیقی مسئلہ حل کرنے کے طریقے
    *   موثر وسائل کا استعمال
    *   ناول تعامل کے طریقے
    *   جدید منصوبہ بندی کی تکنیکیں

    ### دستاویزی (10%)
    *   واضح کوڈ کی دستاویزات
    *   جامع جانچ کے نتائج
    *   کارکردگی کا تجزیہ
    *   مستقبل کی بہتری کی تجاویز

    ## جدید خصوصیات (اختیاری)

    ### ملٹی موڈل انٹریکشن
    *   تقریر کو اشارے کی شناخت کے ساتھ جوڑیں
    *   روبوٹ کے تاثرات کے ذریعے بصری فیڈ بیک
    *   ہیرا پھیری کے کاموں کے لیے ہیپٹک فیڈ بیک

    ### سیکھنا اور موافقت
    *   صارف کی ترجیحات کو یاد رکھیں
    *   ماحولیاتی تبدیلیوں کے مطابق ڈھالیں
    *   تعامل کے نمونوں سے سیکھیں

    ### سماجی تعامل
    *   قدرتی گفتگو کا بہاؤ
    *   سیاق و سباق سے آگاہ جوابات
    *   جذباتی ذہانت کی سیمولیشن

    ## تعیناتی کے تحفظات

    ### سیمولیشن سے حقیقی دنیا میں منتقلی
    *   مضبوط پرسیپشن کے لیے ڈومین رینڈمائزیشن
    *   Sim-to-real منتقلی کی تکنیکیں
    *   حقیقت کے فرق کو کم سے کم کرنے کی حکمت عملی

    ### حفاظت اور اخلاقیات
    *   ہنگامی اسٹاپ میکانزم
    *   صوتی ڈیٹا کے لیے رازداری کے تحفظات
    *   اخلاقی AI استعمال کے رہنما اصول

    ### توسیع پذیری
    *   ملٹی روبوٹ کوآرڈینیشن
    *   کلاؤڈ پر مبنی پروسیسنگ کے اختیارات
    *   موزوں نظام کا آرکیٹیکچر

    ## نتیجہ

    خود مختار ہیومنائیڈ کیپ اسٹون پروجیکٹ اس کورس میں شامل تمام Physical AI اور ہیومنائیڈ روبوٹکس تصورات کے انضمام کی نمائندگی کرتا ہے۔ اس پروجیکٹ کو کامیابی سے مکمل کرنا اس کی مہارت کا مظاہرہ کرتا ہے:

    *   روبوٹ مواصلات اور کنٹرول کے لیے ROS 2
    *   جانچ اور ترقی کے لیے سیمولیشن ماحول
    *   پرسیپشن اور فیصلہ سازی کے لیے AI سسٹمز
    *   انسانی روبوٹ تعامل کے لیے قدرتی زبان کی پروسیسنگ
    *   سسٹم انضمام اور ریئل ٹائم آپریشن

    یہ پروجیکٹ Physical AI اور ہیومنائیڈ روبوٹکس میں جدید تحقیق اور ترقی کی بنیاد کے طور پر کام کرتا ہے، جو طلباء کو اس تیزی سے ترقی پذیر میدان میں جدید کام کے لیے تیار کرتا ہے۔
  </div>
</BilingualChapter>
