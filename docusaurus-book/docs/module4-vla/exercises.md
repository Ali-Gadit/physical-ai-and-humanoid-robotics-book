---
id: exercises
title: "Module 4 Practical Exercises"
sidebar_position: 5
---

# Module 4 Practical Exercises

## Overview

This section contains hands-on exercises to reinforce your understanding of Vision-Language-Action (VLA) systems, including voice-to-action capabilities using OpenAI Whisper and cognitive planning using Large Language Models to translate natural language into ROS 2 actions. These exercises will help you gain practical experience with the integration of vision, language, and action components for humanoid robot control.

## Exercise 1: Whisper-Based Voice Recognition Setup

### Objective
Set up and configure OpenAI Whisper for voice command recognition in a humanoid robot system.

### Instructions
1. Install OpenAI Whisper and required dependencies
2. Configure audio input for voice capture
3. Test Whisper with various voice commands
4. Integrate Whisper with ROS 2 messaging system
5. Validate recognition accuracy and performance

### Required Components
- OpenAI Whisper installation
- Audio recording setup
- ROS 2 integration
- Command validation system

### Code Template
```python
import whisper
import torch
import pyaudio
import numpy as np
from std_msgs.msg import String
import rclpy
from rclpy.node import Node

class WhisperVoiceNode(Node):
    def __init__(self):
        super().__init__('whisper_voice_node')

        # Initialize Whisper model
        self.get_logger().info("Loading Whisper model...")
        self.model = whisper.load_model("base")

        # Audio configuration
        self.rate = 16000
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Publisher for recognized text
        self.recognized_pub = self.create_publisher(String, '/recognized_speech', 10)

        # Timer for continuous listening
        self.listen_timer = self.create_timer(5.0, self.listen_and_recognize)

        self.get_logger().info("Whisper Voice Node initialized")

    def record_audio(self, duration=5):
        """Record audio from microphone"""
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        frames = []
        for i in range(0, int(self.rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        # Convert to numpy array
        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        audio_np = audio_np.astype(np.float32) / 32768.0

        return audio_np

    def listen_and_recognize(self):
        """Listen for voice and recognize speech"""
        try:
            # Record audio
            self.get_logger().info("Recording audio...")
            audio_data = self.record_audio(duration=3)

            # Transcribe using Whisper
            self.get_logger().info("Transcribing audio...")
            audio_tensor = torch.from_numpy(audio_data)
            result = self.model.transcribe(audio_tensor, fp16=False)

            recognized_text = result["text"].strip()
            self.get_logger().info(f"Recognized: {recognized_text}")

            # Publish recognized text
            msg = String()
            msg.data = recognized_text
            self.recognized_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Error in voice recognition: {e}")

    def destroy_node(self):
        """Clean up resources"""
        self.audio.terminate()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = WhisperVoiceNode()

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
- Whisper model loads successfully
- Audio recording works properly
- Speech recognition produces text output
- ROS 2 messages published correctly

### Evaluation Criteria
- Whisper installation and configuration
- Audio recording functionality
- Recognition accuracy
- ROS 2 integration

## Exercise 2: Cognitive Planning with LLMs

### Objective
Implement cognitive planning using a Large Language Model to translate natural language commands into robot action sequences.

### Instructions
1. Set up LLM API access (OpenAI GPT, Anthropic Claude, etc.)
2. Create a function to process natural language commands
3. Implement task decomposition logic
4. Generate executable action sequences
5. Integrate with ROS 2 action execution

### Code Template
```python
import openai
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class CognitivePlanningNode(Node):
    def __init__(self, api_key):
        super().__init__('cognitive_planning_node')

        # Set up LLM
        openai.api_key = api_key
        self.model = "gpt-4"

        # Publishers and subscribers
        self.command_sub = self.create_subscription(
            String,
            '/natural_language_command',
            self.command_callback,
            10
        )

        self.action_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Available robot actions
        self.robot_actions = {
            "move_forward": {
                "description": "Move robot forward",
                "parameters": {"distance": "float", "speed": "float"}
            },
            "turn_left": {
                "description": "Turn robot left",
                "parameters": {"angle": "float", "speed": "float"}
            },
            "turn_right": {
                "description": "Turn robot right",
                "parameters": {"angle": "float", "speed": "float"}
            },
            "stop": {
                "description": "Stop robot movement",
                "parameters": {}
            }
        }

        self.get_logger().info("Cognitive Planning Node initialized")

    def command_callback(self, msg):
        """Process natural language command"""
        command = msg.data
        self.get_logger().info(f"Received command: {command}")

        # Generate plan using LLM
        plan = self.generate_plan(command)

        # Execute plan
        self.execute_plan(plan)

    def generate_plan(self, command):
        """Generate action plan using LLM"""
        prompt = f"""
        Given the natural language command: "{command}"

        Available robot actions: {json.dumps(self.robot_actions)}

        Please decompose this command into a sequence of specific robot actions.
        Each action should be one of the available actions with appropriate parameters.

        Respond with the action sequence in JSON format:
        {{
            "actions": [
                {{
                    "action": "action_name",
                    "parameters": {{"param1": "value1", "param2": "value2"}}
                }}
            ]
        }}
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a robot task planner. Generate executable action sequences from natural language commands."},
                    {"role": "user", "content": prompt}
                ]
            )

            result = json.loads(response.choices[0].message.content)
            return result.get("actions", [])

        except Exception as e:
            self.get_logger().error(f"Error generating plan: {e}")
            return [{"action": "stop", "parameters": {}}]

    def execute_plan(self, plan):
        """Execute the generated action plan"""
        self.get_logger().info(f"Executing plan with {len(plan)} actions")

        for action in plan:
            self.execute_single_action(action)

    def execute_single_action(self, action):
        """Execute a single action"""
        action_name = action["action"]
        parameters = action.get("parameters", {})

        self.get_logger().info(f"Executing: {action_name} with {parameters}")

        if action_name == "move_forward":
            self.move_forward(parameters.get("distance", 1.0), parameters.get("speed", 0.5))
        elif action_name == "turn_left":
            self.turn_left(parameters.get("angle", 90.0), parameters.get("speed", 0.3))
        elif action_name == "turn_right":
            self.turn_right(parameters.get("angle", 90.0), parameters.get("speed", 0.3))
        elif action_name == "stop":
            self.stop_robot()

    def move_forward(self, distance, speed):
        """Move robot forward"""
        cmd = Twist()
        cmd.linear.x = speed if distance > 0 else -speed
        cmd.angular.z = 0.0

        # Publish for duration based on distance and speed
        duration = abs(distance) / abs(speed) if speed != 0 else 0
        self.publish_for_duration(cmd, duration)

    def turn_left(self, angle, speed):
        """Turn robot left"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = speed

        # Convert angle to time (simplified)
        duration = abs(angle) / 180.0 * 3.14 / speed if speed != 0 else 0
        self.publish_for_duration(cmd, duration)

    def turn_right(self, angle, speed):
        """Turn robot right"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = -speed

        duration = abs(angle) / 180.0 * 3.14 / speed if speed != 0 else 0
        self.publish_for_duration(cmd, duration)

    def stop_robot(self):
        """Stop robot movement"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.action_pub.publish(cmd)

    def publish_for_duration(self, cmd, duration):
        """Publish command for specified duration"""
        import time
        start_time = time.time()
        while time.time() - start_time < duration:
            self.action_pub.publish(cmd)
            # In real ROS 2, use proper rate control

def main(args=None, api_key="your-api-key-here"):
    rclpy.init(args=args)
    node = CognitivePlanningNode(api_key)

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
- LLM processes natural language commands
- Action sequences generated correctly
- Robot actions executed via ROS 2
- Proper task decomposition

## Exercise 3: Voice-to-Action Integration

### Objective
Integrate voice recognition with cognitive planning to create a complete voice-controlled robot system.

### Instructions
1. Combine Whisper voice recognition with LLM cognitive planning
2. Create a pipeline from voice input to robot action
3. Implement error handling for recognition failures
4. Add safety checks and validation
5. Test with various voice commands

### Code Template
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import whisper
import openai
import pyaudio
import numpy as np
import threading
import queue

class VoiceToActionNode(Node):
    def __init__(self, api_key):
        super().__init__('voice_to_action_node')

        # Initialize components
        self.whisper_model = whisper.load_model("base")
        openai.api_key = api_key
        self.llm_model = "gpt-4"

        # Audio configuration
        self.audio = pyaudio.PyAudio()
        self.rate = 16000
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1

        # Publishers and subscribers
        self.action_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/voice_status', 10)

        # Audio processing queue
        self.audio_queue = queue.Queue()
        self.processing_active = True

        # Start audio recording thread
        self.recording_thread = threading.Thread(target=self.record_audio_continuous)
        self.recording_thread.daemon = True
        self.recording_thread.start()

        # Timer for processing audio
        self.process_timer = self.create_timer(2.0, self.process_audio)

        # Robot actions definition
        self.robot_actions = {
            "move_forward": {"linear": (0.5, 0.0), "duration": 2.0},
            "move_backward": {"linear": (-0.5, 0.0), "duration": 2.0},
            "turn_left": {"angular": 0.5, "duration": 1.0},
            "turn_right": {"angular": -0.5, "duration": 1.0},
            "stop": {"linear": (0.0, 0.0), "duration": 0.1}
        }

        self.get_logger().info("Voice-to-Action Node initialized")

    def record_audio_continuous(self):
        """Continuously record audio in a separate thread"""
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        frames = []
        while self.processing_active:
            data = stream.read(self.chunk)
            frames.append(data)

            # Process every ~3 seconds of audio
            if len(frames) * self.chunk >= self.rate * 3:
                audio_data = b''.join(frames)
                self.audio_queue.put(audio_data)
                frames = []

        stream.stop_stream()
        stream.close()

    def process_audio(self):
        """Process audio from queue"""
        try:
            while True:  # Process all available audio
                audio_data = self.audio_queue.get_nowait()

                # Convert to numpy array
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                audio_np = audio_np.astype(np.float32) / 32768.0

                # Transcribe
                result = self.whisper_model.transcribe(audio_np, fp16=False)
                recognized_text = result["text"].strip()

                if recognized_text:
                    self.get_logger().info(f"Recognized: {recognized_text}")

                    # Process with cognitive planner
                    self.process_command_with_llm(recognized_text)

        except queue.Empty:
            pass  # No more audio to process

    def process_command_with_llm(self, command):
        """Process command using LLM cognitive planning"""
        if not command.strip():
            return

        # Publish status
        status_msg = String()
        status_msg.data = f"Processing: {command}"
        self.status_pub.publish(status_msg)

        # Create LLM prompt for action planning
        prompt = f"""
        Natural language command: "{command}"

        Available robot actions:
        - move_forward: Move robot forward
        - move_backward: Move robot backward
        - turn_left: Turn robot left
        - turn_right: Turn robot right
        - stop: Stop robot movement

        Please convert this command to the most appropriate robot action.
        Respond with just the action name, nothing else.
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a robot command interpreter. Respond with only the action name from the available actions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            action_name = response.choices[0].message.content.strip().lower()

            # Validate action
            if action_name in self.robot_actions:
                self.execute_action(action_name)
                self.get_logger().info(f"Executed action: {action_name}")
            else:
                self.get_logger().warn(f"Unrecognized action: {action_name}")
                self.execute_action("stop")  # Safety fallback

        except Exception as e:
            self.get_logger().error(f"Error processing command with LLM: {e}")
            self.execute_action("stop")  # Safety fallback

    def execute_action(self, action_name):
        """Execute a robot action"""
        if action_name not in self.robot_actions:
            self.get_logger().error(f"Unknown action: {action_name}")
            return

        action_def = self.robot_actions[action_name]

        cmd = Twist()
        if "linear" in action_def:
            linear_speed, angular_speed = action_def["linear"]
            cmd.linear.x = float(linear_speed)
            cmd.angular.z = float(angular_speed)
        elif "angular" in action_def:
            cmd.angular.z = float(action_def["angular"])

        # Publish command for specified duration
        duration = action_def["duration"]
        self.publish_command_for_duration(cmd, duration)

    def publish_command_for_duration(self, cmd, duration):
        """Publish command for specified duration"""
        import time
        start_time = time.time()

        while time.time() - start_time < duration:
            self.action_pub.publish(cmd)
            time.sleep(0.1)  # 10Hz update rate

        # Stop robot after action
        stop_cmd = Twist()
        self.action_pub.publish(stop_cmd)

    def destroy_node(self):
        """Clean up resources"""
        self.processing_active = False
        self.audio.terminate()
        super().destroy_node()

def main(args=None, api_key="your-api-key-here"):
    rclpy.init(args=args)
    node = VoiceToActionNode(api_key)

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
- Continuous audio recording and processing
- Voice commands recognized and processed
- Appropriate robot actions executed
- Safety fallbacks implemented

## Exercise 4: VLA System Integration and Testing

### Objective
Create a complete Vision-Language-Action system that integrates voice recognition, cognitive planning, and robot execution in a simulated environment.

### Instructions
1. Set up a complete VLA system with all components
2. Integrate with Gazebo simulation environment
3. Test with various complex commands
4. Implement error handling and recovery
5. Validate system performance and safety

### Launch File Template
```xml
<?xml version="1.0"?>
<launch>
  <!-- Gazebo simulation -->
  <include file="$(find-pkg-share gazebo_ros)/launch/gazebo.launch.py">
    <arg name="world" value="$(find-pkg-share vla_exercises)/worlds/vla_test.world"/>
  </include>

  <!-- Robot state publisher -->
  <node pkg="robot_state_publisher" exec="robot_state_publisher" name="robot_state_publisher">
    <param name="robot_description" value="$(find-pkg-share vla_exercises)/urdf/test_robot.urdf"/>
  </node>

  <!-- Voice recognition node -->
  <node pkg="vla_exercises" exec="whisper_voice_node" name="whisper_voice" output="screen">
    <param name="model_size" value="base"/>
  </node>

  <!-- Cognitive planning node -->
  <node pkg="vla_exercises" exec="cognitive_planning_node" name="cognitive_planner" output="screen">
    <param name="llm_model" value="gpt-4"/>
  </node>

  <!-- Voice-to-action integration node -->
  <node pkg="vla_exercises" exec="voice_to_action_node" name="vla_system" output="screen">
    <param name="api_key" value="your-api-key"/>
  </node>

  <!-- Navigation stack -->
  <include file="$(find-pkg-share nav2_bringup)/launch/navigation_launch.py">
    <arg name="params_file" value="$(find-pkg-share vla_exercises)/config/vla_nav_params.yaml"/>
  </include>

  <!-- RViz for visualization -->
  <node pkg="rviz2" exec="rviz2" name="rviz2" args="-d $(find-pkg-share vla_exercises)/rviz/vla_config.rviz"/>
</launch>
```

### Testing Scenarios
```python
# Test scenarios for VLA system
test_scenarios = [
    {
        "command": "move forward",
        "expected": "Robot moves forward",
        "validation": "Check robot position change"
    },
    {
        "command": "turn left",
        "expected": "Robot turns left",
        "validation": "Check robot orientation change"
    },
    {
        "command": "go to kitchen",
        "expected": "Robot navigates to kitchen area",
        "validation": "Check robot reaches kitchen location"
    },
    {
        "command": "find the red ball",
        "expected": "Robot searches for red ball",
        "validation": "Check if ball is detected in camera feed"
    },
    {
        "command": "stop",
        "expected": "Robot stops all movement",
        "validation": "Check robot velocity is zero"
    }
]

def run_vla_tests():
    """Run comprehensive VLA system tests"""
    import time

    for i, scenario in enumerate(test_scenarios):
        print(f"Running test {i+1}: {scenario['command']}")

        # In simulation, send the command to the system
        # This would typically involve publishing to the command topic
        print(f"  Expected: {scenario['expected']}")
        print(f"  Validation: {scenario['validation']}")

        # Wait for execution
        time.sleep(5)  # Adjust based on expected execution time

        print(f"  Test {i+1} completed\n")

if __name__ == "__main__":
    run_vla_tests()
```

### Performance Metrics
```python
class VLASystemMetrics:
    def __init__(self):
        self.response_times = []
        self.recognition_accuracy = []
        self.execution_success = []
        self.safety_violations = []

    def record_response_time(self, time_ms):
        """Record system response time"""
        self.response_times.append(time_ms)

    def record_recognition_accuracy(self, accuracy):
        """Record speech recognition accuracy"""
        self.recognition_accuracy.append(accuracy)

    def record_execution_success(self, success):
        """Record action execution success"""
        self.execution_success.append(success)

    def record_safety_violation(self, violation_type):
        """Record safety violation"""
        self.safety_violations.append(violation_type)

    def get_performance_report(self):
        """Generate performance report"""
        if not self.response_times:
            return "No data collected"

        avg_response = sum(self.response_times) / len(self.response_times)
        avg_accuracy = sum(self.recognition_accuracy) / len(self.recognition_accuracy) if self.recognition_accuracy else 0
        success_rate = sum(self.execution_success) / len(self.execution_success) if self.execution_success else 0

        return f"""
        VLA System Performance Report:
        - Average Response Time: {avg_response:.2f}ms
        - Average Recognition Accuracy: {avg_accuracy:.2f}%
        - Execution Success Rate: {success_rate:.2f}%
        - Safety Violations: {len(self.safety_violations)}
        """
```

### Expected Output
- Complete VLA system running in simulation
- Successful execution of test scenarios
- Performance metrics collected and reported
- Safety constraints enforced

## Exercise 5: Autonomous Humanoid Capstone Integration

### Objective
Integrate all VLA components into the complete Autonomous Humanoid capstone system and test end-to-end functionality.

### Instructions
1. Combine all previous exercises into a complete system
2. Implement the Autonomous Humanoid capstone project
3. Test with complex multi-step commands
4. Validate system robustness and reliability
5. Document lessons learned and improvements

### Complete System Integration
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import Image, Imu
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from builtin_interfaces.msg import Duration
import whisper
import openai
import pyaudio
import numpy as np

class AutonomousHumanoidSystem(Node):
    def __init__(self, api_key):
        super().__init__('autonomous_humanoid_system')

        # Initialize all components
        self.whisper_model = whisper.load_model("base")
        openai.api_key = api_key
        self.llm_model = "gpt-4"

        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.rate = 16000
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.speech_pub = self.create_publisher(String, '/robot_speech', 10)
        self.status_pub = self.create_publisher(String, '/system_status', 10)

        # Subscribers
        self.camera_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.camera_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)

        # Action clients
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # System state
        self.world_state = {
            "robot_position": (0, 0, 0),
            "robot_orientation": 0.0,
            "objects": {},
            "locations": ["kitchen", "living_room", "bedroom", "office"]
        }

        # Execution state
        self.current_plan = []
        self.plan_index = 0
        self.is_executing = False
        self.listening_for_command = True

        # Timers
        self.voice_timer = self.create_timer(5.0, self.check_for_voice_command)
        self.status_timer = self.create_timer(1.0, self.publish_system_status)

        self.get_logger().info("Autonomous Humanoid System initialized")

    def check_for_voice_command(self):
        """Check for and process voice commands"""
        if not self.listening_for_command:
            return

        try:
            # Record voice command
            command_text = self.record_voice_command(duration=3)
            if command_text.strip():
                self.get_logger().info(f"Heard: {command_text}")

                # Process with cognitive planning
                plan = self.cognitive_plan_command(command_text)
                if plan:
                    self.execute_plan(plan)
                else:
                    self.speak_response("I didn't understand that command.")

        except Exception as e:
            self.get_logger().error(f"Error in voice processing: {e}")

    def record_voice_command(self, duration=3):
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

    def cognitive_plan_command(self, command_text):
        """Generate execution plan using cognitive planning"""
        prompt = f"""
        Natural language command: "{command_text}"

        Current world state:
        - Robot position: {self.world_state['robot_position']}
        - Known locations: {self.world_state['locations']}
        - Known objects: {list(self.world_state['objects'].keys())}

        Available robot capabilities:
        - Navigation: move to locations, avoid obstacles
        - Manipulation: pick up and place objects
        - Perception: detect objects in environment
        - Interaction: speak responses

        Please decompose this command into a sequence of specific robot actions.
        Consider the current world state and robot capabilities.

        Respond with the action sequence in JSON format:
        {{
            "actions": [
                {{
                    "action": "action_type",
                    "parameters": {{"param1": "value1"}}
                }}
            ]
        }}
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an advanced robot task planner. Generate detailed action sequences for humanoid robots."},
                    {"role": "user", "content": prompt}
                ]
            )

            result = response.choices[0].message.content
            # Extract JSON from response (in real implementation, use proper JSON parsing)
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
                return plan_data.get("actions", [])
            else:
                return []

        except Exception as e:
            self.get_logger().error(f"Error in cognitive planning: {e}")
            return []

    def execute_plan(self, plan):
        """Execute the generated plan"""
        if not plan:
            self.speak_response("I couldn't create a plan for that command.")
            return

        self.current_plan = plan
        self.plan_index = 0
        self.is_executing = True
        self.listening_for_command = False  # Pause listening during execution

        self.get_logger().info(f"Executing plan with {len(plan)} actions")
        self.speak_response("Starting to execute your command.")

        self.execute_next_action()

    def execute_next_action(self):
        """Execute the next action in the plan"""
        if (not self.current_plan or
            self.plan_index >= len(self.current_plan) or
            not self.is_executing):
            # Plan completed
            self.is_executing = False
            self.listening_for_command = True
            self.current_plan = []
            self.plan_index = 0

            self.speak_response("Task completed successfully.")
            return

        action = self.current_plan[self.plan_index]
        action_type = action.get("action", "")
        parameters = action.get("parameters", {})

        self.get_logger().info(f"Executing action {self.plan_index + 1}: {action_type}")

        # Execute based on action type
        if action_type == "navigate_to_location":
            self.execute_navigation_action(parameters)
        elif action_type == "detect_objects":
            self.execute_detection_action(parameters)
        elif action_type == "speak":
            self.execute_speak_action(parameters)
        else:
            self.get_logger().warn(f"Unknown action type: {action_type}")
            self.plan_index += 1
            if self.is_executing:
                self.execute_next_action()

    def execute_navigation_action(self, parameters):
        """Execute navigation action"""
        location = parameters.get("location", "unknown")

        # In a real system, you would look up coordinates for the location
        # For this exercise, we'll use a simple mapping
        location_map = {
            "kitchen": (2.0, 1.0, 0.0),
            "living_room": (1.0, 2.0, 0.0),
            "bedroom": (4.0, 3.0, 0.0),
            "office": (2.0, 4.0, 0.0)
        }

        if location in location_map:
            target_pos = location_map[location]
            target_pose = Pose()
            target_pose.position.x = target_pos[0]
            target_pose.position.y = target_pos[1]
            target_pose.position.z = target_pos[2]
            target_pose.orientation.w = 1.0  # Default orientation

            self.send_navigation_goal(target_pose)
        else:
            self.get_logger().error(f"Unknown location: {location}")
            self.plan_index += 1
            if self.is_executing:
                self.execute_next_action()

    def send_navigation_goal(self, pose):
        """Send navigation goal to Nav2"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose = pose

        self.nav_client.wait_for_server()
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.navigation_result_callback)

    def navigation_result_callback(self, future):
        """Handle navigation result"""
        try:
            goal_result = future.result()

            if goal_result.status == 3:  # STATUS_SUCCEEDED (would normally use GoalStatus.STATUS_SUCCEEDED)
                self.get_logger().info('Navigation succeeded')
                self.plan_index += 1
                if self.is_executing:
                    self.execute_next_action()
            else:
                self.get_logger().error('Navigation failed')
                self.is_executing = False
                self.listening_for_command = True
                self.speak_response("Navigation failed. I couldn't reach the destination.")
        except Exception as e:
            self.get_logger().error(f'Error in navigation result callback: {e}')
            self.is_executing = False
            self.listening_for_command = True

    def execute_detection_action(self, parameters):
        """Execute object detection action"""
        target_objects = parameters.get("target_objects", ["object"])
        self.get_logger().info(f"Detecting objects: {target_objects}")

        # In a real system, this would trigger computer vision modules
        # For this exercise, we'll simulate detection
        detected = f"Detected: {', '.join(target_objects)}"
        self.speak_response(detected)

        # Update world state with detected objects
        for obj in target_objects:
            self.world_state["objects"][obj] = {"position": (0, 0, 0), "detected": True}

        self.plan_index += 1
        if self.is_executing:
            self.execute_next_action()

    def execute_speak_action(self, parameters):
        """Execute speech action"""
        text = parameters.get("text", "Hello")
        self.speak_response(text)

        self.plan_index += 1
        if self.is_executing:
            self.execute_next_action()

    def speak_response(self, text):
        """Make robot speak"""
        self.get_logger().info(f"Robot says: {text}")
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)

    def publish_system_status(self):
        """Publish system status"""
        status_msg = String()
        status_msg.data = f"Executing: {self.is_executing}, Listening: {self.listening_for_command}, Actions in plan: {len(self.current_plan) if self.current_plan else 0}"
        self.status_pub.publish(status_msg)

    def camera_callback(self, msg):
        """Handle camera data"""
        # Process camera feed for object detection
        pass

    def imu_callback(self, msg):
        """Handle IMU data"""
        # Update robot orientation
        self.world_state["robot_orientation"] = msg.orientation.z

    def destroy_node(self):
        """Clean up resources"""
        self.audio.terminate()
        super().destroy_node()

def main(args=None, api_key="your-api-key-here"):
    rclpy.init(args=args)
    node = AutonomousHumanoidSystem(api_key)

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
- Complete autonomous humanoid system operational
- Voice commands processed and executed
- Multi-step tasks completed successfully
- System status monitoring active

## Assessment Questions

1. How does the integration of Whisper with cognitive planning enhance robot capabilities?

2. What are the main challenges in translating natural language to executable robot actions?

3. How do you ensure safety when using LLMs for robot control?

4. What performance metrics are important for VLA systems?

5. How would you extend this system for real-world deployment?

## Submission Requirements

For each exercise, submit:
- Source code for all implemented components
- Configuration files and launch files
- Test results and performance metrics
- Screenshots of successful execution
- A detailed report documenting your implementation, challenges faced, and solutions implemented

## Evaluation Rubric

- **Functionality** (40%): Systems work as expected and meet requirements
- **Integration** (25%): Components work together seamlessly
- **Innovation** (20%): Creative solutions and advanced features
- **Documentation** (15%): Clear explanations and proper documentation

Complete all exercises to gain comprehensive experience with Vision-Language-Action systems for Physical AI and humanoid robotics applications.