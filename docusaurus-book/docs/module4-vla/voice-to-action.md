---
id: voice-to-action
title: "Voice-to-Action: Using OpenAI Whisper for Voice Commands"
sidebar_position: 2
---

# Voice-to-Action: Using OpenAI Whisper for Voice Commands

## Introduction

Voice-to-Action systems enable natural human-robot interaction by allowing users to control robots through spoken commands. OpenAI Whisper, a state-of-the-art automatic speech recognition (ASR) system, provides the foundation for robust voice command recognition in humanoid robots. This technology enables robots to understand and respond to natural language, making them more accessible and intuitive to use.

For humanoid robots operating in human-centric environments, voice interfaces are particularly valuable as they enable hands-free interaction and natural communication patterns that humans are accustomed to.

## Understanding OpenAI Whisper

### Architecture and Capabilities

OpenAI Whisper is a transformer-based neural network trained on a large dataset of multilingual and multitask supervised data. Its key capabilities include:

- **Multilingual Support**: Recognition in 99+ languages
- **Robust Performance**: Works well in noisy environments
- **Speaker Identification**: Can distinguish between different speakers
- **Timestamping**: Provides timing information for speech segments
- **Punctuation and Capitalization**: Outputs properly formatted text

### Whisper Models

Whisper comes in several sizes with different performance characteristics:

- **tiny**: ~39M parameters, fastest but least accurate
- **base**: ~74M parameters, good balance of speed and accuracy
- **small**: ~244M parameters, better accuracy
- **medium**: ~769M parameters, high accuracy
- **large**: ~1550M parameters, highest accuracy, slowest

For humanoid robots, the choice depends on hardware capabilities and real-time requirements.

## Installing Whisper for Robotics

### Prerequisites

```bash
# Install required dependencies
pip install torch torchvision torchaudio
pip install openai-whisper
pip install pyaudio  # For audio input
pip install sounddevice  # Alternative audio library
pip install numpy
pip install transformers  # If using additional NLP processing
```

### Hardware Requirements

- **Microphone**: USB microphone or array for voice capture
- **Processing Power**: GPU recommended for real-time processing
- **Memory**: 2GB+ RAM for base model, 8GB+ for large model

## Basic Whisper Implementation

### Simple Speech Recognition

```python
import whisper
import torch
import pyaudio
import wave
import numpy as np
import time

class WhisperVoiceToAction:
    def __init__(self, model_size="base", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize Whisper voice-to-action system

        Args:
            model_size: Size of Whisper model (tiny, base, small, medium, large)
            device: Device to run model on (cuda, cpu)
        """
        print(f"Loading Whisper {model_size} model on {device}...")
        self.model = whisper.load_model(model_size).to(device)
        self.device = device

        # Audio configuration
        self.rate = 16000  # Sample rate
        self.chunk = 1024  # Buffer size
        self.format = pyaudio.paInt16
        self.channels = 1

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        print("Whisper voice-to-action system initialized successfully!")

    def record_audio(self, duration=5):
        """
        Record audio from microphone

        Args:
            duration: Recording duration in seconds

        Returns:
            Audio data as numpy array
        """
        print(f"Recording for {duration} seconds...")

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

        print("Recording complete!")

        stream.stop_stream()
        stream.close()

        # Convert to numpy array
        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        # Normalize to [-1, 1]
        audio_np = audio_np.astype(np.float32) / 32768.0

        return audio_np

    def transcribe_audio(self, audio_data):
        """
        Transcribe audio using Whisper

        Args:
            audio_data: Audio data as numpy array

        Returns:
            Transcribed text
        """
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_data).to(self.device)

        # Transcribe
        result = self.model.transcribe(audio_tensor, fp16=False if self.device == "cpu" else True)

        return result["text"].strip()

    def continuous_listening(self, callback_func=None):
        """
        Continuously listen for voice commands

        Args:
            callback_func: Function to call with transcribed text
        """
        print("Starting continuous listening...")
        print("Say 'quit' to exit")

        while True:
            try:
                # Record audio
                audio_data = self.record_audio(duration=3)  # 3-second recordings

                # Transcribe
                text = self.transcribe_audio(audio_data)

                if text:
                    print(f"Recognized: {text}")

                    if callback_func:
                        callback_func(text)

                    # Exit condition
                    if "quit" in text.lower():
                        break

            except KeyboardInterrupt:
                print("\nStopping...")
                break
            except Exception as e:
                print(f"Error during recognition: {e}")

    def close(self):
        """Clean up resources"""
        self.audio.terminate()

# Example usage
def command_callback(text):
    """Callback function to process recognized commands"""
    print(f"Processing command: {text}")

    # Here you would implement command processing logic
    # For example, mapping commands to robot actions
    if "move" in text.lower():
        print("Robot should move!")
    elif "stop" in text.lower():
        print("Robot should stop!")
    elif "hello" in text.lower():
        print("Robot should greet!")

def main():
    # Initialize voice-to-action system
    vta = WhisperVoiceToAction(model_size="base")

    try:
        # Start continuous listening
        vta.continuous_listening(callback_func=command_callback)
    finally:
        vta.close()

if __name__ == "__main__":
    main()
```

## Advanced Voice Command Processing

### Command Recognition and Parsing

```python
import re
from typing import Dict, List, Optional
import json

class VoiceCommandProcessor:
    def __init__(self):
        """Initialize command processor with predefined command patterns"""
        self.command_patterns = {
            # Navigation commands
            "move_forward": [r"move forward", r"go forward", r"move ahead", r"go ahead"],
            "move_backward": [r"move backward", r"go backward", r"move back", r"go back"],
            "turn_left": [r"turn left", r"turn to the left", r"rotate left"],
            "turn_right": [r"turn right", r"turn to the right", r"rotate right"],
            "stop": [r"stop", r"halt", r"freeze", r"pause"],

            # Manipulation commands
            "pick_up": [r"pick up", r"grasp", r"grab", r"take"],
            "put_down": [r"put down", r"release", r"drop", r"place"],

            # Interaction commands
            "greet": [r"hello", r"hi", r"greet", r"say hello"],
            "introduce": [r"introduce yourself", r"who are you", r"what are you"],
            "follow": [r"follow me", r"come with me", r"follow"],

            # Complex commands with parameters
            "navigate_to": [r"go to (.+)", r"move to (.+)", r"navigate to (.+)"],
            "find_object": [r"find (.+)", r"locate (.+)", r"search for (.+)"]
        }

        # Object recognition patterns
        self.object_patterns = [
            r"(\w+) (table|chair|cup|bottle|box|door|window)",
            r"(red|blue|green|yellow|big|small) (\w+)",
            r"(\w+) (left|right|front|back)"
        ]

    def extract_command(self, text: str) -> Dict:
        """
        Extract command and parameters from text

        Args:
            text: Recognized speech text

        Returns:
            Dictionary with command type and parameters
        """
        text_lower = text.lower().strip()

        for command_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    # Extract parameters if present
                    params = match.groups() if match.groups() else []

                    return {
                        "command": command_type,
                        "original_text": text,
                        "parameters": list(params),
                        "confidence": 0.9  # Placeholder confidence
                    }

        # If no specific command found, return as general command
        return {
            "command": "unknown",
            "original_text": text,
            "parameters": [],
            "confidence": 0.5
        }

    def process_command(self, text: str) -> Optional[Dict]:
        """
        Process voice command and return executable action

        Args:
            text: Recognized speech text

        Returns:
            Action dictionary or None if command not recognized
        """
        command_info = self.extract_command(text)

        if command_info["command"] == "unknown":
            return None

        # Convert command to robot action
        action = self.command_to_action(command_info)
        return action

    def command_to_action(self, command_info: Dict) -> Dict:
        """
        Convert command information to robot action

        Args:
            command_info: Command information from extract_command

        Returns:
            Robot action dictionary
        """
        action_map = {
            "move_forward": {
                "type": "navigation",
                "action": "move_straight",
                "parameters": {"distance": 1.0, "speed": 0.5}
            },
            "move_backward": {
                "type": "navigation",
                "action": "move_straight",
                "parameters": {"distance": -1.0, "speed": 0.5}
            },
            "turn_left": {
                "type": "navigation",
                "action": "rotate",
                "parameters": {"angle": -90.0, "speed": 0.3}
            },
            "turn_right": {
                "type": "navigation",
                "action": "rotate",
                "parameters": {"angle": 90.0, "speed": 0.3}
            },
            "stop": {
                "type": "control",
                "action": "stop",
                "parameters": {}
            },
            "greet": {
                "type": "interaction",
                "action": "speak",
                "parameters": {"text": "Hello! How can I help you?"}
            },
            "navigate_to": {
                "type": "navigation",
                "action": "navigate_to_location",
                "parameters": {"location": command_info["parameters"][0] if command_info["parameters"] else "unknown"}
            }
        }

        return action_map.get(command_info["command"], {
            "type": "unknown",
            "action": "unknown",
            "parameters": {}
        })

# Integration with Whisper system
class AdvancedWhisperVoiceToAction(WhisperVoiceToAction):
    def __init__(self, model_size="base", device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_size, device)
        self.command_processor = VoiceCommandProcessor()
        self.command_history = []

    def process_voice_command(self, text: str) -> Optional[Dict]:
        """
        Process voice command and return executable action

        Args:
            text: Recognized speech text

        Returns:
            Robot action dictionary or None
        """
        action = self.command_processor.process_command(text)

        if action:
            # Add to command history
            self.command_history.append({
                "timestamp": time.time(),
                "original_text": text,
                "action": action
            })

            print(f"Command processed: {action}")
            return action

        print(f"Command not recognized: {text}")
        return None

    def continuous_listening_with_processing(self):
        """Continuously listen and process commands"""
        print("Starting continuous listening with command processing...")
        print("Say 'quit' to exit")

        while True:
            try:
                # Record audio
                audio_data = self.record_audio(duration=3)

                # Transcribe
                text = self.transcribe_audio(audio_data)

                if text:
                    print(f"Recognized: {text}")

                    # Process command
                    action = self.process_voice_command(text)

                    if action:
                        # Execute action (in real implementation, this would send to robot)
                        self.execute_action(action)

                    # Exit condition
                    if "quit" in text.lower():
                        break

            except KeyboardInterrupt:
                print("\nStopping...")
                break
            except Exception as e:
                print(f"Error during recognition: {e}")

    def execute_action(self, action: Dict):
        """
        Execute the robot action

        Args:
            action: Action dictionary from command processing
        """
        print(f"Executing action: {action}")
        # In real implementation, this would send commands to robot
        # via ROS 2 or other communication interface

        # Example action execution
        if action["type"] == "navigation":
            if action["action"] == "move_straight":
                distance = action["parameters"]["distance"]
                print(f"Moving straight {distance} meters")
            elif action["action"] == "rotate":
                angle = action["parameters"]["angle"]
                print(f"Rotating {angle} degrees")
        elif action["type"] == "control":
            if action["action"] == "stop":
                print("Stopping robot")
        elif action["type"] == "interaction":
            if action["action"] == "speak":
                text = action["parameters"]["text"]
                print(f"Speaking: {text}")

def main():
    # Initialize advanced voice-to-action system
    vta = AdvancedWhisperVoiceToAction(model_size="base")

    try:
        # Start continuous listening with processing
        vta.continuous_listening_with_processing()
    finally:
        vta.close()

if __name__ == "__main__":
    main()
```

## Integration with ROS 2

### Voice Command ROS 2 Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import AudioData
from builtin_interfaces.msg import Duration
import whisper
import torch
import pyaudio
import numpy as np
import threading
import queue

class VoiceCommandNode(Node):
    def __init__(self):
        super().__init__('voice_command_node')

        # Initialize Whisper model
        self.get_logger().info("Loading Whisper model...")
        self.model = whisper.load_model("base")

        # Publishers for robot commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.speech_pub = self.create_publisher(String, '/robot_speech', 10)

        # Subscribers for audio input (if using external audio source)
        self.audio_sub = self.create_subscription(
            AudioData,
            '/audio_input',
            self.audio_callback,
            10
        )

        # Publisher for recognized text
        self.recognized_pub = self.create_publisher(String, '/recognized_speech', 10)

        # Audio configuration
        self.rate = 16000
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Command processor
        self.command_processor = VoiceCommandProcessor()

        # Start audio recording thread
        self.recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = threading.Thread(target=self.record_audio_thread)
        self.recording_thread.daemon = True

        # Timer for processing audio
        self.process_timer = self.create_timer(3.0, self.process_audio)

        self.get_logger().info("Voice Command Node initialized")

    def start_recording(self):
        """Start audio recording"""
        self.recording = True
        self.recording_thread.start()

    def stop_recording(self):
        """Stop audio recording"""
        self.recording = False

    def record_audio_thread(self):
        """Audio recording thread"""
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        frames = []
        while self.recording:
            data = stream.read(self.chunk)
            frames.append(data)

            if len(frames) * self.chunk >= self.rate * 3:  # 3 seconds of audio
                # Convert to numpy array and add to queue
                audio_data = b''.join(frames)
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                audio_np = audio_np.astype(np.float32) / 32768.0

                self.audio_queue.put(audio_np)
                frames = []  # Reset frames

        stream.stop_stream()
        stream.close()

    def audio_callback(self, msg):
        """Callback for audio data from external source"""
        # Convert audio data to numpy array
        audio_np = np.frombuffer(msg.data, dtype=np.int16)
        audio_np = audio_np.astype(np.float32) / 32768.0

        # Add to processing queue
        self.audio_queue.put(audio_np)

    def process_audio(self):
        """Process audio from queue"""
        try:
            while True:  # Process all available audio in queue
                audio_data = self.audio_queue.get_nowait()

                # Transcribe audio
                text = self.transcribe_audio(audio_data)

                if text.strip():
                    self.get_logger().info(f"Recognized: {text}")

                    # Publish recognized text
                    recognized_msg = String()
                    recognized_msg.data = text
                    self.recognized_pub.publish(recognized_msg)

                    # Process command
                    self.process_command(text)

        except queue.Empty:
            pass  # No more audio to process

    def transcribe_audio(self, audio_data):
        """Transcribe audio using Whisper"""
        try:
            audio_tensor = torch.from_numpy(audio_data)
            result = self.model.transcribe(audio_tensor, fp16=False)
            return result["text"].strip()
        except Exception as e:
            self.get_logger().error(f"Transcription error: {e}")
            return ""

    def process_command(self, text):
        """Process recognized command"""
        action = self.command_processor.process_command(text)

        if action:
            self.execute_action(action)
        else:
            self.get_logger().info(f"Command not recognized: {text}")

    def execute_action(self, action):
        """Execute robot action based on command"""
        self.get_logger().info(f"Executing action: {action}")

        if action["type"] == "navigation":
            if action["action"] == "move_straight":
                self.move_straight(action["parameters"]["distance"])
            elif action["action"] == "rotate":
                self.rotate(action["parameters"]["angle"])
        elif action["type"] == "control":
            if action["action"] == "stop":
                self.stop_robot()
        elif action["type"] == "interaction":
            if action["action"] == "speak":
                self.speak(action["parameters"]["text"])

    def move_straight(self, distance):
        """Move robot straight for specified distance"""
        cmd = Twist()
        cmd.linear.x = 0.5 if distance > 0 else -0.5  # Adjust speed as needed
        cmd.angular.z = 0.0

        # Publish command for duration based on distance
        duration = abs(distance) / 0.5  # Assuming 0.5 m/s speed
        self.publish_for_duration(cmd, duration)

    def rotate(self, angle):
        """Rotate robot by specified angle"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.5 if angle > 0 else -0.5  # Adjust angular speed as needed

        # Convert angle to time (assuming constant angular velocity)
        duration = abs(angle) / 180.0 * 3.14  # Simplified calculation
        self.publish_for_duration(cmd, duration)

    def stop_robot(self):
        """Stop robot movement"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

    def speak(self, text):
        """Make robot speak text"""
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)

    def publish_for_duration(self, cmd, duration):
        """Publish command for specified duration"""
        start_time = self.get_clock().now()

        # Create duration object
        duration_msg = Duration()
        duration_msg.sec = int(duration)
        duration_msg.nanosec = int((duration - int(duration)) * 1e9)

        # Calculate end time properly
        end_time = start_time + Duration(seconds=duration_msg.sec, nanoseconds=duration_msg.nanosec)

        while self.get_clock().now() < end_time:
            self.cmd_vel_pub.publish(cmd)
            # Use a proper rate instead of rclpy.spin_once
            time.sleep(0.1)  # Sleep for 100ms

def main(args=None):
    rclpy.init(args=args)
    node = VoiceCommandNode()

    try:
        node.start_recording()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_recording()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Voice Processing Features

### Voice Activity Detection

```python
import webrtcvad
import collections

class VoiceActivityDetector:
    def __init__(self, sample_rate=16000, frame_duration_ms=30, mode=3):
        """
        Initialize Voice Activity Detector

        Args:
            sample_rate: Audio sample rate (8000, 16000, 32000, or 48000)
            frame_duration_ms: Frame duration in milliseconds (10, 20, or 30)
            mode: Aggressiveness mode (0-3, higher means more aggressive)
        """
        self.vad = webrtcvad.Vad(mode)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # 2 bytes per sample

        # Buffer for voice activity detection
        self.ring_buffer = collections.deque(maxlen=30)  # 30 frames = 900ms
        self.triggered = False
        self.vad_frames = []

    def is_speech(self, audio_data):
        """
        Detect if audio contains speech

        Args:
            audio_data: Audio data as bytes

        Returns:
            Boolean indicating if speech is detected
        """
        # Convert to frames
        frames = self._bytes_to_frames(audio_data)

        speech_frames = 0
        total_frames = len(frames)

        for frame in frames:
            if self.vad.is_speech(frame, self.sample_rate):
                speech_frames += 1

        # Consider as speech if more than 20% of frames contain speech
        speech_ratio = speech_frames / total_frames if total_frames > 0 else 0
        return speech_ratio > 0.2

    def _bytes_to_frames(self, audio_data):
        """Convert audio bytes to frames for VAD"""
        frames = []
        for i in range(0, len(audio_data), self.frame_size):
            frame = audio_data[i:i + self.frame_size]
            if len(frame) == self.frame_size:
                frames.append(frame)
        return frames

# Enhanced voice-to-action with VAD
class VADEnhancedVoiceToAction(AdvancedWhisperVoiceToAction):
    def __init__(self, model_size="base", device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_size, device)
        self.vad_detector = VoiceActivityDetector()

        # Voice activity state
        self.listening_for_speech = False
        self.speech_buffer = b""
        self.max_silence_frames = 10  # Max silence frames before stopping
        self.silence_counter = 0

    def record_with_vad(self, max_duration=10):
        """
        Record audio with voice activity detection

        Args:
            max_duration: Maximum recording duration in seconds

        Returns:
            Audio data containing speech
        """
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        frames = []
        silence_frames = 0
        max_frames = int(max_duration * self.rate / self.chunk)
        frame_count = 0

        print("Listening with VAD...")

        while frame_count < max_frames:
            data = stream.read(self.chunk)
            frames.append(data)
            frame_count += 1

            # Check for voice activity
            if self.vad_detector.is_speech(data):
                silence_frames = 0  # Reset silence counter
                print(".", end="", flush=True)  # Visual feedback
            else:
                silence_frames += 1
                print(" ", end="", flush=True)  # Visual feedback

            # If we have enough silence, stop recording
            if silence_frames > 15 and len(frames) > 10:  # At least 10 frames recorded
                print(f"\nSilence detected, stopping after {silence_frames} silent frames")
                break

        print(f"\nRecording complete: {len(frames)} frames")

        stream.stop_stream()
        stream.close()

        # Convert to numpy array
        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        audio_np = audio_np.astype(np.float32) / 32768.0

        return audio_np
```

## Performance Optimization

### Model Optimization for Real-time Processing

```python
class OptimizedWhisperProcessor:
    def __init__(self, model_size="base"):
        """Initialize optimized Whisper processor"""
        # Load model with optimizations
        self.model = whisper.load_model(model_size, device="cuda" if torch.cuda.is_available() else "cpu")

        # Enable optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # Pre-allocate tensors for better performance
        self.max_audio_length = 32000 * 30  # 30 seconds at 16kHz
        self.audio_buffer = torch.zeros(self.max_audio_length, dtype=torch.float32)

    def transcribe_optimized(self, audio_data):
        """Optimized transcription with pre-allocated tensors"""
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_data).to(self.model.device)

        # Pad or trim to standard length for consistent performance
        if len(audio_tensor) < self.max_audio_length:
            padded = torch.zeros(self.max_audio_length, dtype=audio_tensor.dtype, device=audio_tensor.device)
            padded[:len(audio_tensor)] = audio_tensor
        else:
            padded = audio_tensor[:self.max_audio_length]

        # Transcribe
        result = self.model.transcribe(padded, fp16=False if self.model.device.type == "cpu" else True)

        return result["text"].strip()
```

## Best Practices for Voice-to-Action

### 1. Error Handling and Robustness

```python
class RobustVoiceToAction:
    def __init__(self):
        self.retry_count = 3
        self.confidence_threshold = 0.7
        self.timeout_seconds = 5.0

    def robust_transcribe(self, audio_data):
        """Transcribe with error handling and retries"""
        for attempt in range(self.retry_count):
            try:
                text = self.transcribe_audio(audio_data)

                # Check if transcription is reasonable
                if self.is_reasonable_transcription(text):
                    return text
                else:
                    print(f"Attempt {attempt + 1}: Transcription seems unreasonable: '{text}'")

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")

            if attempt < self.retry_count - 1:
                time.sleep(0.5)  # Brief pause before retry

        return None  # Failed after all retries

    def is_reasonable_transcription(self, text):
        """Check if transcription is reasonable"""
        if not text or len(text.strip()) < 2:
            return False

        # Check for common Whisper errors
        if text.strip().lower() in ["", "thank you", "thanks", "bye", "goodbye"]:
            return False

        # Check for excessive punctuation or special characters
        special_chars = sum(1 for c in text if c in "!@#$%^&*()_+-=[]{}|;:,.<>?")
        if special_chars > len(text) * 0.3:  # More than 30% special characters
            return False

        return True
```

### 2. Context-Aware Processing

```python
class ContextAwareVoiceProcessor:
    def __init__(self):
        self.context = {
            "location": "unknown",
            "time_of_day": "unknown",
            "robot_state": "idle",
            "last_command": None,
            "conversation_history": []
        }

    def process_command_with_context(self, text):
        """Process command considering current context"""
        # Update context based on command
        if "move" in text.lower() or "go" in text.lower():
            self.context["robot_state"] = "navigating"
        elif "stop" in text.lower():
            self.context["robot_state"] = "stopped"

        # Process command
        action = self.command_processor.process_command(text)

        # Add context to action if needed
        if action:
            action["context"] = self.context.copy()

        # Update conversation history
        self.context["conversation_history"].append({
            "timestamp": time.time(),
            "command": text,
            "action": action
        })

        # Keep only recent history
        if len(self.context["conversation_history"]) > 10:
            self.context["conversation_history"] = self.context["conversation_history"][-10:]

        return action
```

## Troubleshooting Common Issues

### Audio Quality Issues
- **Background noise**: Use noise suppression or better microphones
- **Clipping**: Reduce microphone gain or use automatic gain control
- **Low volume**: Increase microphone sensitivity or use directional microphones

### Recognition Accuracy
- **Model selection**: Choose appropriate model size for your needs
- **Audio preprocessing**: Apply noise reduction and normalization
- **Language settings**: Specify the correct language for Whisper

### Performance Issues
- **Latency**: Optimize model loading and use smaller models if needed
- **Memory usage**: Use CPU offloading for large models
- **Real-time constraints**: Implement streaming recognition for continuous input

## Hands-on Exercise

Create a complete Voice-to-Action system that includes:

1. Whisper-based speech recognition with appropriate model selection
2. Voice Activity Detection to reduce unnecessary processing
3. Command parsing and mapping to robot actions
4. ROS 2 integration for controlling a simulated robot
5. Context-aware processing to improve command understanding
6. Error handling and robustness features

This exercise will give you hands-on experience with implementing voice interfaces for humanoid robots and understanding the challenges involved in natural human-robot interaction.