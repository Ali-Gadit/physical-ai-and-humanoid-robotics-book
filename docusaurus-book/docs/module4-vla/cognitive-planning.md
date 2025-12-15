---
id: cognitive-planning
title: "Cognitive Planning: Using LLMs to Translate Natural Language into ROS 2 Actions"
sidebar_position: 3
---

import BilingualChapter from '@site/src/components/BilingualChapter';

<BilingualChapter>
  <div className="english">
    # Cognitive Planning: Using LLMs to Translate Natural Language into ROS 2 Actions

    ## Introduction

    Cognitive planning represents the intelligence layer of Vision-Language-Action (VLA) systems, where Large Language Models (LLMs) serve as the reasoning engine that translates high-level natural language commands into executable sequences of robotic actions. This capability enables humanoid robots to understand complex, context-dependent instructions and break them down into specific, actionable steps that can be executed through ROS 2 interfaces.

    The cognitive planning system bridges the gap between human intent and robot execution, enabling sophisticated human-robot interaction where users can issue commands in natural language like "Please clean the table in the kitchen and then bring me a cup of water" without needing to specify low-level robotic actions.

    ## Understanding Cognitive Planning in Robotics

    ### The Planning Pipeline

    Cognitive planning in robotics follows a multi-step process:

    ```
    Natural Language Command → LLM Interpretation → Task Decomposition → Action Sequencing → ROS 2 Execution
    ```

    ### Key Components

    1. **Language Understanding**: Interpreting the user's intent and command structure
    2. **World Modeling**: Understanding the current state of the environment
    3. **Task Decomposition**: Breaking complex tasks into simpler subtasks
    4. **Action Sequencing**: Ordering actions based on dependencies and constraints
    5. **Execution Monitoring**: Ensuring plan execution proceeds as expected

    ## Large Language Models for Robotics

    ### Choosing the Right LLM

    For cognitive planning in robotics, consider these factors:

    - **Reasoning Capabilities**: Ability to decompose complex tasks
    - **Context Understanding**: Understanding spatial and temporal relationships
    - **Tool Usage**: Ability to interface with external tools and APIs
    - **Safety Constraints**: Built-in safety mechanisms for robot control
    - **Latency Requirements**: Response time for real-time interaction

    ### Popular LLM Options

    1. **OpenAI GPT-4**: Excellent reasoning and tool usage capabilities
    2. **Anthropic Claude**: Strong safety and reasoning features
    3. **Open Source Models**: Llama 2/3, Mistral, etc. for local deployment
    4. **Specialized Models**: Models fine-tuned for robotics applications

    ## Implementing Cognitive Planning Systems

    ### Basic Cognitive Planning Architecture

    ```python
    import openai
    import json
    import time
    from typing import Dict, List, Any, Optional
    from dataclasses import dataclass
    from enum import Enum

    class RobotActionType(Enum):
        NAVIGATION = "navigation"
        MANIPULATION = "manipulation"
        PERCEPTION = "perception"
        INTERACTION = "interaction"
        CONTROL = "control"

    @dataclass
    class RobotAction:
        """Represents a single robot action"""
        action_type: RobotActionType
        action_name: str
        parameters: Dict[str, Any]
        description: str
        estimated_duration: float = 1.0  # seconds

    class WorldState:
        """Represents the current state of the world/environment"""
        def __init__(self):
            self.objects = {}  # object_id -> position, properties
            self.robot_position = (0, 0, 0)
            self.robot_orientation = 0.0
            self.robot_state = "idle"  # idle, moving, manipulating, etc.
            self.environment_map = {}  # locations and their properties
            self.time_of_day = "unknown"
            self.last_updated = time.time()

        def update_object_position(self, obj_id: str, position: tuple):
            """Update the position of an object in the world"""
            if obj_id not in self.objects:
                self.objects[obj_id] = {}
            self.objects[obj_id]["position"] = position
            self.objects[obj_id]["last_seen"] = time.time()

        def get_object_position(self, obj_id: str) -> Optional[tuple]:
            """Get the position of an object"""
            obj_data = self.objects.get(obj_id, {})
            return obj_data.get("position")

        def update_robot_state(self, position: tuple, orientation: float, state: str):
            """Update robot state"""
            self.robot_position = position
            self.robot_orientation = orientation
            self.robot_state = state
            self.last_updated = time.time()

    class CognitivePlanner:
        """Main cognitive planning system"""
        def __init__(self, api_key: str = None, model: str = "gpt-4"):
            self.model = model
            self.api_key = api_key
            if api_key:
                openai.api_key = api_key

            self.world_state = WorldState()
            self.action_history = []

            # Define available robot actions
            self.available_actions = {
                "move_to_location": {
                    "description": "Move robot to a specific location",
                    "parameters": {
                        "location": {"type": "string", "description": "Target location name or coordinates"},
                        "speed": {"type": "number", "description": "Movement speed (0.1-1.0)", "default": 0.5}
                    }
                },
                "pick_up_object": {
                    "description": "Pick up an object from the environment",
                    "parameters": {
                        "object_id": {"type": "string", "description": "ID of the object to pick up"},
                        "arm": {"type": "string", "description": "Which arm to use", "enum": ["left", "right"], "default": "right"}
                    }
                },
                "place_object": {
                    "description": "Place an object at a specific location",
                    "parameters": {
                        "location": {"type": "string", "description": "Target location for placement"},
                        "arm": {"type": "string", "description": "Which arm is holding the object", "enum": ["left", "right"], "default": "right"}
                    }
                },
                "grasp_object": {
                    "description": "Grasp an object with specified force",
                    "parameters": {
                        "object_id": {"type": "string", "description": "ID of the object to grasp"},
                        "force": {"type": "number", "description": "Grasping force (0.0-1.0)", "default": 0.5}
                    }
                },
                "release_object": {
                    "description": "Release a grasped object",
                    "parameters": {
                        "arm": {"type": "string", "description": "Which arm to release", "enum": ["left", "right"]}
                    }
                },
                "detect_objects": {
                    "description": "Detect objects in the current environment",
                    "parameters": {
                        "target_objects": {"type": "array", "items": {"type": "string"}, "description": "Specific objects to look for"},
                        "max_objects": {"type": "integer", "description": "Maximum number of objects to detect", "default": 10}
                    }
                },
                "navigate_to_object": {
                    "description": "Navigate to a specific object",
                    "parameters": {
                        "object_id": {"type": "string", "description": "ID of the target object"},
                        "approach_distance": {"type": "number", "description": "Distance to approach (meters)", "default": 0.5}
                    }
                },
                "speak": {
                    "description": "Make the robot speak",
                    "parameters": {
                        "text": {"type": "string", "description": "Text to speak"},
                        "language": {"type": "string", "description": "Language code", "default": "en"}
                    }
                }
            }

        def plan_from_command(self, command: str) -> List[RobotAction]:
            """
            Generate a plan from a natural language command

            Args:
                command: Natural language command from user

            Returns:
                List of RobotAction objects representing the plan
            """
            # Create a prompt for the LLM
            prompt = self._create_planning_prompt(command)

            try:
                # Call the LLM to generate the plan
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    functions=self._get_available_functions(),
                    function_call="auto"
                )

                # Parse the response
                plan = self._parse_llm_response(response)
                return plan

            except Exception as e:
                print(f"Error generating plan: {e}")
                return self._create_fallback_plan(command)

        def _create_planning_prompt(self, command: str) -> str:
            """Create a prompt for the LLM planning task"""
            prompt = f"""
            Given the following natural language command: "{command}"

            Current world state:
            - Robot position: {self.world_state.robot_position}
            - Robot state: {self.world_state.robot_state}
            - Known objects: {list(self.world_state.objects.keys())}
            - Environment: {list(self.world_state.environment_map.keys())}

            Please decompose this command into a sequence of specific robot actions.
            Each action should be one of the available functions.
            Consider the spatial relationships and task dependencies.

            Respond with the sequence of actions in JSON format.
            """
            return prompt

        def _get_system_prompt(self) -> str:
            """Get the system prompt for the LLM"""
            return """
            You are an expert robot task planner. Your job is to decompose natural language commands into specific, executable robot actions.
            Consider the following:
            1. Spatial relationships and navigation requirements
            2. Object manipulation needs
            3. Sequential dependencies between actions
            4. Safety constraints
            5. Efficiency of the action sequence

            Always respond with valid JSON containing the action sequence.
            """

        def _get_available_functions(self) -> List[Dict]:
            """Get the list of available functions for the LLM"""
            functions = []
            for action_name, action_info in self.available_actions.items():
                function_def = {
                    "name": action_name,
                    "description": action_info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": action_info["parameters"],
                        "required": []  # Will be filled based on parameters
                    }
                }

                # Determine required parameters (those without defaults)
                for param_name, param_info in action_info["parameters"].items():
                    if "default" not in param_info:
                        function_def["parameters"]["required"].append(param_name)

                functions.append(function_def)

            return functions

        def _parse_llm_response(self, response) -> List[RobotAction]:
            """Parse the LLM response into RobotAction objects"""
            plan = []

            # Extract function calls from response
            if hasattr(response.choices[0].message, 'function_call'):
                # Handle single function call
                call = response.choices[0].message.function_call
                action = self._convert_function_call_to_action(call)
                if action:
                    plan.append(action)
            elif hasattr(response.choices[0].message, 'tool_calls'):
                # Handle multiple tool calls (newer API)
                for tool_call in response.choices[0].message.tool_calls:
                    action = self._convert_function_call_to_action(tool_call.function)
                    if action:
                        plan.append(action)

            return plan

        def _convert_function_call_to_action(self, function_call) -> Optional[RobotAction]:
            """Convert LLM function call to RobotAction"""
            try:
                action_name = function_call.name
                arguments = json.loads(function_call.arguments)

                # Map to RobotActionType
                action_type_map = {
                    "move_to_location": RobotActionType.NAVIGATION,
                    "pick_up_object": RobotActionType.MANIPULATION,
                    "place_object": RobotActionType.MANIPULATION,
                    "grasp_object": RobotActionType.MANIPULATION,
                    "release_object": RobotActionType.MANIPULATION,
                    "detect_objects": RobotActionType.PERCEPTION,
                    "navigate_to_object": RobotActionType.NAVIGATION,
                    "speak": RobotActionType.INTERACTION
                }

                action_type = action_type_map.get(action_name, RobotActionType.CONTROL)

                # Create action description
                description = f"{action_name} with parameters: {arguments}"

                return RobotAction(
                    action_type=action_type,
                    action_name=action_name,
                    parameters=arguments,
                    description=description
                )
            except Exception as e:
                print(f"Error converting function call to action: {e}")
                return None

        def _create_fallback_plan(self, command: str) -> List[RobotAction]:
            """Create a fallback plan if LLM fails"""
            print(f"Using fallback plan for command: {command}")

            # Simple fallback based on command keywords
            if "move" in command.lower() or "go" in command.lower():
                return [RobotAction(
                    action_type=RobotActionType.NAVIGATION,
                    action_name="move_to_location",
                    parameters={"location": "default"},
                    description="Fallback movement action"
                )]
            elif "pick" in command.lower() or "grasp" in command.lower():
                return [RobotAction(
                    action_type=RobotActionType.MANIPULATION,
                    action_name="pick_up_object",
                    parameters={"object_id": "unknown"},
                    description="Fallback pick-up action"
                )]
            else:
                return [RobotAction(
                    action_type=RobotActionType.INTERACTION,
                    action_name="speak",
                    parameters={"text": "I don't understand the command. Can you please rephrase?"},
                    description="Fallback communication action"
                )]

        def update_world_state(self, state_update: Dict[str, Any]):
            """Update the world state with new information"""
            for key, value in state_update.items():
                if hasattr(self.world_state, key):
                    setattr(self.world_state, key, value)
            self.world_state.last_updated = time.time()

    # Example usage
def main():
    # Initialize cognitive planner (you would need to provide your OpenAI API key)
    planner = CognitivePlanner(api_key="your-api-key-here")

    # Example commands
    commands = [
        "Go to the kitchen and bring me a cup",
        "Find the red ball and pick it up",
        "Move to the table and place the cup there"
    ]

    for command in commands:
        print(f"\nCommand: {command}")
        plan = planner.plan_from_command(command)

        print("Generated plan:")
        for i, action in enumerate(plan):
            print(f"  {i+1}. {action.description}")

if __name__ == "__main__":
    main()
    ```

    ## Advanced Cognitive Planning with Context

    ### Context-Aware Planning

    ```python
    class ContextAwarePlanner(CognitivePlanner):
        def __init__(self, api_key: str = None, model: str = "gpt-4"):
            super().__init__(api_key, model)

            # Additional context information
            self.conversation_history = []
            self.user_preferences = {}
            self.task_context = {}  # Context for current task
            self.safety_constraints = []

        def plan_with_context(self, command: str, context: Dict[str, Any] = None) -> List[RobotAction]:
            """
            Generate a plan considering additional context information

            Args:
                command: Natural language command
                context: Additional context information

            Returns:
                List of RobotAction objects
            """
            # Merge provided context with internal context
            full_context = self._merge_context(context or {})

            # Create enhanced prompt with context
            prompt = self._create_contextual_planning_prompt(command, full_context)

            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_contextual_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    functions=self._get_available_functions(),
                    function_call="auto",
                    temperature=0.3  # Lower temperature for more consistent planning
                )

                plan = self._parse_llm_response(response)

                # Apply safety constraints to the plan
                safe_plan = self._apply_safety_constraints(plan)

                return safe_plan

            except Exception as e:
                print(f"Error generating contextual plan: {e}")
                return self._create_fallback_plan(command)

        def _merge_context(self, external_context: Dict[str, Any]) -> Dict[str, Any]:
            """Merge external context with internal context"""
            full_context = {
                "world_state": self.world_state.__dict__.copy(),
                "conversation_history": self.conversation_history[-5:],  # Last 5 exchanges
                "user_preferences": self.user_preferences,
                "task_context": self.task_context,
                "safety_constraints": self.safety_constraints,
                "current_time": time.time(),
                "robot_capabilities": self._get_robot_capabilities()
            }

            # Update with external context
            full_context.update(external_context)

            return full_context

        def _create_contextual_planning_prompt(self, command: str, context: Dict[str, Any]) -> str:
            """Create a planning prompt with rich context"""
            prompt = f"""
            Natural Language Command: "{command}"

            CONTEXT INFORMATION:
            - Current Time: {context.get('current_time', 'unknown')}
            - Robot Capabilities: {context.get('robot_capabilities', {})}
            - Safety Constraints: {context.get('safety_constraints', [])}

            WORLD STATE:
            - Robot Position: {context['world_state'].get('robot_position', 'unknown')}
            - Robot State: {context['world_state'].get('robot_state', 'unknown')}
            - Known Objects: {list(context['world_state']['objects'].keys()) if context['world_state'].get('objects') else []}
            - Environment Map: {list(context['world_state']['environment_map'].keys()) if context['world_state'].get('environment_map') else []}

            USER PREFERENCES:
            - {context.get('user_preferences', {})}

            PREVIOUS INTERACTIONS:
            - {context.get('conversation_history', [])}

            TASK-SPECIFIC CONTEXT:
            - {context.get('task_context', {})}

            PLANNING REQUIREMENTS:
            1. Consider all safety constraints when generating the plan
            2. Take into account user preferences and previous interactions
            3. Ensure actions are feasible given robot capabilities
            4. Maintain spatial and temporal coherence
            5. Handle potential ambiguities by asking for clarification if needed

            Generate a detailed sequence of actions to fulfill the command.
            Each action should be specific and executable.
            If clarification is needed, include a 'request_clarification' action.
            """
            return prompt

        def _get_contextual_system_prompt(self) -> str:
            """Get system prompt for contextual planning"""
            return """
            You are an advanced robot task planner with access to rich contextual information.
            Your role is to generate safe, efficient, and contextually appropriate action sequences.

            When planning, consider:
            1. Safety constraints and avoid hazardous actions
            2. User preferences and past interactions
            3. Current world state and spatial relationships
            4. Robot capabilities and limitations
            5. Task-specific context and requirements

            Always prioritize safety and user preferences.
            If the command is ambiguous or requires clarification, generate a clarification request.
            """

        def _get_robot_capabilities(self) -> Dict[str, Any]:
            """Get robot capabilities information"""
            return {
                "navigation": {
                    "max_speed": 0.5,  # m/s
                    "min_turn_radius": 0.3,  # meters
                    "max_gradient": 15.0  # degrees
                },
                "manipulation": {
                    "max_payload": 2.0,  # kg
                    "reach_distance": 1.2,  # meters
                    "precision": "medium"
                },
                "perception": {
                    "camera_range": 5.0,  # meters
                    "object_detection_accuracy": 0.85
                },
                "interaction": {
                    "speech_synthesis": True,
                    "language_support": ["en", "es", "fr"]
                }
            }

        def _apply_safety_constraints(self, plan: List[RobotAction]) -> List[RobotAction]:
            """Apply safety constraints to the plan"""
            safe_plan = []

            for action in plan:
                if self._is_action_safe(action):
                    safe_plan.append(action)
                else:
                    print(f"Action filtered out for safety: {action.description}")
                    # Add safety action instead
                    safety_action = RobotAction(
                        action_type=RobotActionType.INTERACTION,
                        action_name="speak",
                        parameters={"text": "I cannot perform this action for safety reasons."},
                        description="Safety warning"
                    )
                    safe_plan.append(safety_action)

            return safe_plan

        def _is_action_safe(self, action: RobotAction) -> bool:
            """Check if an action is safe to execute"""
            # Check against safety constraints
            for constraint in self.safety_constraints:
                if constraint.get("type") == "forbidden_action":
                    if action.action_name in constraint.get("actions", []):
                        return False
                elif constraint.get("type") == "spatial_constraint":
                    # Check if action violates spatial constraints
                    if action.action_name == "move_to_location":
                        target = action.parameters.get("location")
                        forbidden_areas = constraint.get("forbidden_areas", [])
                        if target in forbidden_areas:
                            return False

            return True

        def add_safety_constraint(self, constraint: Dict[str, Any]):
            """Add a safety constraint"""
            self.safety_constraints.append(constraint)

        def update_user_preferences(self, preferences: Dict[str, Any]):
            """Update user preferences"""
            self.user_preferences.update(preferences)
    ```

    ## Integration with ROS 2

    ### ROS 2 Cognitive Planning Node

    ```python
    import rclpy
    import json
    from rclpy.node import Node
    from std_msgs.msg import String
    from geometry_msgs.msg import Twist, Pose, Point
    from action_msgs.msg import GoalStatus
    from rclpy.action import ActionClient
    from nav2_msgs.action import NavigateToPose
    from builtin_interfaces.msg import Time

    class CognitivePlanningNode(Node):
        def __init__(self):
            super().__init__('cognitive_planning_node')

            # Initialize cognitive planner
            self.planner = ContextAwarePlanner()

            # Subscribers for commands and world state updates
            self.command_sub = self.create_subscription(
                String,
                '/natural_language_command',
                self.command_callback,
                10
            )

            self.world_state_sub = self.create_subscription(
                String,  # In practice, you'd use a custom message type
                '/world_state_update',
                self.world_state_callback,
                10
            )

            # Publishers for robot commands and status
            self.action_pub = self.create_publisher(String, '/robot_action', 10)
            self.status_pub = self.create_publisher(String, '/planning_status', 10)

            # Action clients for navigation
            self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

            # Planning state
            self.current_plan = []
            self.plan_index = 0
            self.is_executing = False

            self.get_logger().info('Cognitive Planning Node initialized')

        def command_callback(self, msg):
            """Handle natural language command"""
            command = msg.data
            self.get_logger().info(f'Received command: {command}')

            # Generate plan
            self.current_plan = self.planner.plan_with_context(command)
            self.plan_index = 0
            self.is_executing = True

            # Publish planning status
            status_msg = String()
            status_msg.data = f'Generated plan with {len(self.current_plan)} actions'
            self.status_pub.publish(status_msg)

            # Start executing plan
            self.execute_next_action()

        def world_state_callback(self, msg):
            """Handle world state updates"""
            try:
                state_update = json.loads(msg.data)
                self.planner.update_world_state(state_update)
                self.get_logger().info('World state updated')
            except json.JSONDecodeError:
                self.get_logger().error('Invalid world state JSON')

        def execute_next_action(self):
            """Execute the next action in the plan"""
            if not self.current_plan or self.plan_index >= len(self.current_plan) or not self.is_executing:
                # Plan completed
                self.is_executing = False
                self.plan_index = 0
                self.current_plan = []

                status_msg = String()
                status_msg.data = 'Plan completed'
                self.status_pub.publish(status_msg)
                return

            action = self.current_plan[self.plan_index]
            self.get_logger().info(f'Executing action {self.plan_index + 1}: {action.action_name}')

            # Execute based on action type
            if action.action_type == RobotActionType.NAVIGATION:
                self.execute_navigation_action(action)
            elif action.action_type == RobotActionType.MANIPULATION:
                self.execute_manipulation_action(action)
            elif action.action_type == RobotActionType.PERCEPTION:
                self.execute_perception_action(action)
            elif action.action_type == RobotActionType.INTERACTION:
                self.execute_interaction_action(action)
            else:
                self.execute_generic_action(action)

        def execute_navigation_action(self, action):
            """Execute navigation action"""
            if action.action_name == "move_to_location":
                location = action.parameters.get("location", "unknown")

                # In a real implementation, you would look up the coordinates
                # for the named location from a map
                target_pose = self.get_location_pose(location)

                if target_pose:
                    self.send_navigation_goal(target_pose)
                else:
                    self.get_logger().error(f'Unknown location: {location}')
                    self.plan_index += 1
                    if self.is_executing:
                        self.execute_next_action()

        def get_location_pose(self, location_name: str) -> Optional[Pose]:
            """Get pose for a named location"""
            # In practice, this would look up coordinates from a map
            # For now, return a default pose or None
            location_map = {
                "kitchen": Pose(position=Point(x=3.0, y=1.0, z=0.0)),
                "living_room": Pose(position=Point(x=1.0, y=2.0, z=0.0)),
                "bedroom": Pose(position=Point(x=4.0, y=3.0, z=0.0)),
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
            try:
                goal_result = future.result()

                if goal_result.status == GoalStatus.STATUS_SUCCEEDED:
                    self.get_logger().info('Navigation succeeded')
                    self.plan_index += 1
                    if self.is_executing:
                        self.execute_next_action()
                else:
                    self.get_logger().error(f'Navigation failed with status: {goal_result.status}')
                    # Stop execution or handle failure
                    self.is_executing = False
            except Exception as e:
                self.get_logger().error(f'Error in navigation result callback: {e}')
                self.is_executing = False

        def execute_manipulation_action(self, action):
            """Execute manipulation action"""
            # In a real implementation, this would interface with manipulation controllers
            action_msg = String()
            action_msg.data = f"manipulation:{action.action_name}:{json.dumps(action.parameters)}"
            self.action_pub.publish(action_msg)

            # Simulate action completion
            self.get_logger().info(f'Manipulation action sent: {action.action_name}')
            self.plan_index += 1
            if self.is_executing:
                self.execute_next_action()

        def execute_perception_action(self, action):
            """Execute perception action"""
            # In a real implementation, this would trigger perception modules
            action_msg = String()
            action_msg.data = f"perception:{action.action_name}:{json.dumps(action.parameters)}"
            self.action_pub.publish(action_msg)

            # Simulate action completion
            self.get_logger().info(f'Perception action sent: {action.action_name}')
            self.plan_index += 1
            if self.is_executing:
                self.execute_next_action()

        def execute_interaction_action(self, action):
            """Execute interaction action"""
            if action.action_name == "speak":
                text = action.parameters.get("text", "")
                self.get_logger().info(f'Robot says: {text}')

                # Publish speech command
                action_msg = String()
                action_msg.data = f"interaction:speak:{text}"
                self.action_pub.publish(action_msg)

            self.plan_index += 1
            if self.is_executing:
                self.execute_next_action()

        def execute_generic_action(self, action):
            """Execute generic action"""
            action_msg = String()
            action_msg.data = f"generic:{action.action_name}:{json.dumps(action.parameters)}"
            self.action_pub.publish(action_msg)

            self.get_logger().info(f'Generic action sent: {action.action_name}')
            self.plan_index += 1
            if self.is_executing:
                self.execute_next_action()

def main(args=None):
    rclpy.init(args=args)
    node = CognitivePlanningNode()

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

    ## Advanced Planning Techniques

    ### Hierarchical Task Network (HTN) Planning

    ```python
    class HTNPlanner:
        """Hierarchical Task Network planner for complex task decomposition"""

        def __init__(self):
            self.tasks = {}
            self.methods = {}
            self.primitive_actions = set()

            # Define basic tasks and methods
            self._define_basic_tasks()

        def _define_basic_tasks(self):
            """Define basic tasks and their decomposition methods"""
            # High-level tasks
            self.tasks = {
                "clean_room": {"description": "Clean the entire room"},
                "serve_drink": {"description": "Serve a drink to a person"},
                "organize_objects": {"description": "Organize objects in an area"}
            }

            # Methods for decomposing tasks
            self.methods = {
                "clean_room": [
                    {
                        "name": "clean_room_method_1",
                        "preconditions": [],
                        "decomposition": [
                            {"task": "detect_objects", "parameters": {"target_objects": ["trash"]}},
                            {"task": "pick_up_object", "parameters": {"object_type": "trash"}},
                            {"task": "move_to_location", "parameters": {"location": "trash_bin"}},
                            {"task": "release_object", "parameters": {}}
                        ]
                    }
                ],
                "serve_drink": [
                    {
                        "name": "serve_drink_method_1",
                        "preconditions": [],
                        "decomposition": [
                            {"task": "detect_objects", "parameters": {"target_objects": ["cup"]}},
                            {"task": "pick_up_object", "parameters": {"object_id": "cup"}},
                            {"task": "move_to_location", "parameters": {"location": "water_source"}},
                            {"task": "fill_container", "parameters": {"container_id": "cup"}},
                            {"task": "move_to_location", "parameters": {"location": "person_location"}},
                            {"task": "place_object", "parameters": {"location": "table"}}
                        ]
                    }
                ]
            }

            # Primitive actions (leaf nodes in the hierarchy)
            self.primitive_actions = {
                "move_to_location", "pick_up_object", "place_object",
                "release_object", "detect_objects", "fill_container"
            }

        def decompose_task(self, task_name: str, parameters: Dict[str, Any] = None) -> List[RobotAction]:
            """Decompose a high-level task into primitive actions"""
            if task_name in self.primitive_actions:
                # This is already a primitive action
                return [RobotAction(
                    action_type=self._get_action_type(task_name),
                    action_name=task_name,
                    parameters=parameters or {},
                    description=f"Primitive action: {task_name}"
                )]

            if task_name not in self.methods:
                raise ValueError(f"No methods defined for task: {task_name}")

            # Use the first available method (in practice, you'd have selection logic)
            method = self.methods[task_name][0]

            plan = []
            for subtask in method["decomposition"]:
                subtask_name = subtask["task"]
                subtask_params = {**subtask.get("parameters", {}), **(parameters or {})}

                subplan = self.decompose_task(subtask_name, subtask_params)
                plan.extend(subplan)

            return plan

        def _get_action_type(self, action_name: str) -> RobotActionType:
            """Map action name to action type"""
            navigation_actions = {"move_to_location", "navigate_to_object"}
            manipulation_actions = {"pick_up_object", "place_object", "grasp_object", "release_object", "fill_container"}
            perception_actions = {"detect_objects"}
            interaction_actions = {"speak"}

            if action_name in navigation_actions:
                return RobotActionType.NAVIGATION
            elif action_name in manipulation_actions:
                return RobotActionType.MANIPULATION
            elif action_name in perception_actions:
                return RobotActionType.PERCEPTION
            elif action_name in interaction_actions:
                return RobotActionType.INTERACTION
            else:
                return RobotActionType.CONTROL

    # Integration with cognitive planner
    class AdvancedCognitivePlanner(ContextAwarePlanner):
        def __init__(self, api_key: str = None, model: str = "gpt-4"):
            super().__init__(api_key, model)
            self.hierarchical_planner = HTNPlanner()

        def plan_with_hierarchical_decomposition(self, command: str) -> List[RobotAction]:
            """Plan using hierarchical task decomposition"""
            # First, try to identify if this is a high-level task
            high_level_task = self._identify_high_level_task(command)

            if high_level_task:
                # Use hierarchical planner for decomposition
                try:
                    plan = self.hierarchical_planner.decompose_task(
                        high_level_task["task_name"],
                        high_level_task.get("parameters", {})
                    )
                    return plan
                except ValueError:
                    # Fall back to LLM planning if hierarchical method not available
                    pass

            # Fall back to LLM-based planning
            return self.plan_with_context(command)

        def _identify_high_level_task(self, command: str) -> Optional[Dict[str, Any]]:
            """Identify if command corresponds to a high-level task"""
            command_lower = command.lower()

            # Simple keyword-based identification (in practice, use more sophisticated NLU)
            if "clean" in command_lower:
                return {"task_name": "clean_room", "parameters": {}}
            elif any(word in command_lower for word in ["serve", "bring", "get"]):
                if any(word in command_lower for word in ["water", "drink", "beverage"]):
                    return {"task_name": "serve_drink", "parameters": {}}

            return None
    ```

    ## Safety and Validation

    ### Plan Validation and Safety Checking

    ```python
    class PlanValidator:
        """Validate plans for safety and feasibility"""

        def __init__(self):
            self.safety_rules = []
            self.feasibility_constraints = []
            self._setup_default_rules()

        def _setup_default_rules(self):
            """Set up default safety and feasibility rules"""
            # Safety rules
            self.safety_rules = [
                self._no_navigation_to_forbidden_areas,
                self._no_manipulation_of_heavy_objects,
                self._no_high_speed_navigation_in_crowded_areas
            ]

            # Feasibility constraints
            self.feasibility_constraints = [
                self._check_robot_reachability,
                self._check_payload_limits,
                self._check_battery_level
            ]

        def validate_plan(self, plan: List[RobotAction], context: Dict[str, Any]) -> Dict[str, Any]:
            """
            Validate a plan for safety and feasibility

            Returns:
                Dictionary with validation results
            """
            results = {
                "is_valid": True,
                "safety_issues": [],
                "feasibility_issues": [],
                "warnings": [],
                "modified_plan": plan.copy()
            }

            # Check safety
            for rule in self.safety_rules:
                rule_result = rule(plan, context)
                if not rule_result["is_safe"]:
                    results["is_valid"] = False
                    results["safety_issues"].extend(rule_result["issues"])

            # Check feasibility
            for constraint in self.feasibility_constraints:
                constraint_result = constraint(plan, context)
                if not constraint_result["is_feasible"]:
                    results["is_valid"] = False
                    results["feasibility_issues"].extend(constraint_result["issues"])

            # Apply modifications if needed
            if results["safety_issues"] or results["feasibility_issues"]:
                results["modified_plan"] = self._modify_plan_for_safety(
                    plan, results["safety_issues"], results["feasibility_issues"]
                )

            return results

        def _no_navigation_to_forbidden_areas(self, plan: List[RobotAction], context: Dict[str, Any]) -> Dict[str, Any]:
            """Check that navigation doesn't go to forbidden areas"""
            issues = []

            forbidden_areas = context.get("safety_constraints", {}).get("forbidden_areas", [])

            for i, action in enumerate(plan):
                if (action.action_name == "move_to_location" and
                    action.parameters.get("location") in forbidden_areas):
                    issues.append(f"Action {i}: Navigation to forbidden area '{action.parameters['location']}'")

            return {
                "is_safe": len(issues) == 0,
                "issues": issues
            }

        def _no_manipulation_of_heavy_objects(self, plan: List[RobotAction], context: Dict[str, Any]) -> Dict[str, Any]:
            """Check that robot doesn't try to manipulate objects beyond its payload capacity"""
            issues = []
            max_payload = context.get("robot_capabilities", {}).get("manipulation", {}).get("max_payload", 2.0)

            for i, action in enumerate(plan):
                if action.action_name == "pick_up_object":
                    # In a real system, you'd look up the object's weight
                    object_weight = self._get_object_weight(action.parameters.get("object_id", ""))
                    if object_weight > max_payload:
                        issues.append(f"Action {i}: Attempting to pick up object weighing {object_weight}kg, "
                                    f"exceeds payload capacity of {max_payload}kg")

            return {
                "is_safe": len(issues) == 0,
                "issues": issues
            }

        def _check_robot_reachability(self, plan: List[RobotAction], context: Dict[str, Any]) -> Dict[str, Any]:
            """Check that manipulation targets are within robot reach"""
            issues = []
            max_reach = context.get("robot_capabilities", {}).get("manipulation", {}).get("reach_distance", 1.2)

            # This would require spatial reasoning to check if targets are reachable
            # For now, we'll simulate with a simple check
            for i, action in enumerate(plan):
                if action.action_name == "navigate_to_object":
                    # In a real system, you'd calculate the distance to the object
                    distance = self._estimate_distance_to_object(action.parameters.get("object_id"))
                    if distance > max_reach:
                        issues.append(f"Action {i}: Object out of reach (distance: {distance:.2f}m, max: {max_reach}m)")

            return {
                "is_feasible": len(issues) == 0,
                "issues": issues
            }

        def _modify_plan_for_safety(self, plan: List[RobotAction], safety_issues: List[str],
                                  feasibility_issues: List[str]) -> List[RobotAction]:
            """Modify plan to address safety and feasibility issues"""
            modified_plan = plan.copy()

            # Simple modification approach: replace unsafe actions with safe alternatives
            # In practice, you'd have more sophisticated replanning

            for issue in safety_issues + feasibility_issues:
                # Parse issue to identify problematic actions and modify them
                pass  # Implementation would depend on issue format

            return modified_plan

        def _get_object_weight(self, object_id: str) -> float:
            """Get the weight of an object (simulated)"""
            # In a real system, this would query a knowledge base
            # For simulation, return random weights
            import random
            return random.uniform(0.1, 5.0)  # 0.1 to 5.0 kg

        def _estimate_distance_to_object(self, object_id: str) -> float:
            """Estimate distance to an object (simulated)"""
            # In a real system, this would use spatial reasoning
            import random
            return random.uniform(0.5, 3.0)  # 0.5 to 3.0 meters
    ```

    ## Performance Optimization

    ### Caching and Optimization Techniques

    ```python
    import functools
    import hashlib
    from typing import Tuple

    class OptimizedCognitivePlanner(AdvancedCognitivePlanner):
        def __init__(self, api_key: str = None, model: str = "gpt-4"):
            super().__init__(api_key, model)

            # Plan caching
            self.plan_cache = {}
            self.cache_size_limit = 100

            # Performance monitoring
            self.plan_generation_times = []

        def plan_from_command(self, command: str, use_cache: bool = True) -> List[RobotAction]:
            """Generate plan with caching and performance monitoring"""
            start_time = time.time()

            # Create cache key
            cache_key = self._create_cache_key(command, self.world_state)

            if use_cache and cache_key in self.plan_cache:
                # Return cached plan
                cached_plan, generation_time = self.plan_cache[cache_key]
                self.get_logger().info(f"Retrieved plan from cache (generated in {generation_time:.2f}s)")
                self._update_performance_stats(time.time() - start_time, cached=True)
                return cached_plan

            # Generate new plan
            plan = super().plan_from_command(command)

            # Cache the new plan
            generation_time = time.time() - start_time
            self._cache_plan(cache_key, plan, generation_time)

            self._update_performance_stats(generation_time, cached=False)

            return plan

        def _create_cache_key(self, command: str, world_state) -> str:
            """Create a cache key for the command and world state"""
            state_str = str(sorted(world_state.__dict__.items()))
            combined = f"{command}|{state_str}"
            return hashlib.md5(combined.encode()).hexdigest()

        def _cache_plan(self, cache_key: str, plan: List[RobotAction], generation_time: float):
            """Cache a generated plan"""
            # Remove oldest entries if cache is full
            if len(self.plan_cache) >= self.cache_size_limit:
                # Remove oldest entry (in a real system, you might use LRU)
                oldest_key = next(iter(self.plan_cache))
                del self.plan_cache[oldest_key]

            self.plan_cache[cache_key] = (plan, generation_time)

        def _update_performance_stats(self, generation_time: float, cached: bool):
            """Update performance statistics"""
            if not cached:
                self.plan_generation_times.append(generation_time)

                # Keep only recent measurements (last 100)
                if len(self.plan_generation_times) > 100:
                    self.plan_generation_times = self.plan_generation_times[-100:]

        def get_performance_stats(self) -> Dict[str, float]:
            """Get performance statistics"""
            if not self.plan_generation_times:
                return {"avg_generation_time": 0.0, "min_time": 0.0, "max_time": 0.0}

            times = self.plan_generation_times
            return {
                "avg_generation_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "total_plans_generated": len(times),
                "cache_hit_rate": len([t for t in times if t < 0.1]) / len(times) if times else 0  # Approximation
            }
    ```

    ## Best Practices for Cognitive Planning

    ### 1. Error Handling and Recovery

    ```python
    class RobustCognitivePlanner(OptimizedCognitivePlanner):
        def __init__(self, api_key: str = None, model: str = "gpt-4"):
            super().__init__(api_key, model)
            self.max_replan_attempts = 3
            self.recovery_strategies = [
                self._simplified_plan_recovery,
                self._ask_for_clarification,
                self._fallback_to_safe_behavior
            ]

        def execute_plan_with_recovery(self, plan: List[RobotAction]) -> bool:
            """Execute plan with built-in recovery mechanisms"""
            for attempt in range(self.max_replan_attempts):
                try:
                    success = self._execute_plan_internal(plan)
                    if success:
                        return True
                except Exception as e:
                    self.get_logger().error(f"Plan execution failed on attempt {attempt + 1}: {e}")

                    if attempt < self.max_replan_attempts - 1:
                        # Try recovery strategy
                        plan = self._apply_recovery_strategy(plan, str(e))

            return False  # All attempts failed

        def _apply_recovery_strategy(self, failed_plan: List[RobotAction], error: str) -> List[RobotAction]:
            """Apply recovery strategy based on error type"""
            for strategy in self.recovery_strategies:
                new_plan = strategy(failed_plan, error)
                if new_plan:
                    return new_plan

            # If no recovery strategy works, return empty plan
            return []

        def _simplified_plan_recovery(self, failed_plan: List[RobotAction], error: str) -> Optional[List[RobotAction]]:
            """Try to create a simplified version of the plan"""
            # Implementation would depend on error type
            # For example, if navigation failed, try a simpler path
            pass

        def _ask_for_clarification(self, failed_plan: List[RobotAction], error: str) -> Optional[List[RobotAction]]:
            """Generate plan to ask for clarification"""
            return [RobotAction(
                action_type=RobotActionType.INTERACTION,
                action_name="speak",
                parameters={"text": "I encountered an issue. Could you please clarify or rephrase your command?"},
                description="Request clarification"
            )]

        def _fallback_to_safe_behavior(self, failed_plan: List[RobotAction], error: str) -> Optional[List[RobotAction]]:
            """Generate plan for safe behavior"""
            return [RobotAction(
                action_type=RobotActionType.CONTROL,
                action_name="stop",
                parameters={},
                description="Stop and wait for new command"
            )]
    ```

    ### 2. Context Management

    ```python
    class ContextManager:
        """Manage context for cognitive planning"""

        def __init__(self):
            self.long_term_memory = {}  # Persistent across sessions
            self.short_term_memory = {}  # For current interaction
            self.context_windows = {}   # Time-based context windows

        def update_context(self, key: str, value: Any, duration: Optional[float] = None):
            """Update context with optional duration"""
            if duration:
                # Store with expiration
                self.context_windows[key] = {
                    "value": value,
                    "expires": time.time() + duration
                }
            else:
                # Store permanently
                self.long_term_memory[key] = value

            # Always update short-term memory
            self.short_term_memory[key] = value

        def get_context(self, key: str) -> Optional[Any]:
            """Get context value, handling expiration"""
            # Check context windows first (they override other memories)
            if key in self.context_windows:
                window_data = self.context_windows[key]
                if time.time() < window_data["expires"]:
                    return window_data["value"]
                else:
                    # Remove expired context
                    del self.context_windows[key]

            # Check short-term memory
            if key in self.short_term_memory:
                return self.short_term_memory[key]

            # Check long-term memory
            return self.long_term_memory.get(key)

        def clear_expired_context(self):
            """Remove expired context windows"""
            current_time = time.time()
            expired_keys = [
                key for key, data in self.context_windows.items()
                if current_time >= data["expires"]
            ]

            for key in expired_keys:
                del self.context_windows[key]
    ```

    ## Troubleshooting Common Issues

    ### Planning Failures
    - **Ambiguous Commands**: Implement clarification requests
    - **Resource Constraints**: Add feasibility checking
    - **Safety Violations**: Implement safety constraints and validation
    - **Performance Issues**: Use caching and optimization techniques

    ### LLM Integration Issues
    - **API Limitations**: Handle rate limits and timeouts
    - **Cost Management**: Implement usage tracking and optimization
    - **Response Variability**: Add consistency checks and validation

    ## Hands-on Exercise

    Create a complete cognitive planning system that includes:

    1. LLM-based natural language understanding and task decomposition
    2. Context-aware planning with world state management
    3. Hierarchical task network for complex task decomposition
    4. Safety validation and feasibility checking
    5. ROS 2 integration for plan execution
    6. Performance optimization with caching
    7. Error handling and recovery mechanisms

    This exercise will give you hands-on experience with implementing cognitive planning systems for humanoid robots and understanding the challenges involved in translating natural language to executable robot actions.
  </div>
  <div className="urdu">
    # علمی منصوبہ بندی (Cognitive Planning): LLMs کا استعمال کرتے ہوئے قدرتی زبان کو ROS 2 ایکشنز میں تبدیل کرنا

    ## تعارف

    علمی منصوبہ بندی (Cognitive Planning) Vision-Language-Action (VLA) سسٹمز کی ذہانت کی تہہ کی نمائندگی کرتی ہے، جہاں Large Language Models (LLMs) استدلال کے انجن (reasoning engine) کے طور پر کام کرتے ہیں جو اعلیٰ سطحی قدرتی زبان کے احکامات کو روبوٹک ایکشنز کے قابل عمل سلسلے میں ترجمہ کرتے ہیں۔ یہ صلاحیت ہیومنائیڈ روبوٹس کو پیچیدہ، سیاق و سباق پر منحصر ہدایات کو سمجھنے اور انہیں مخصوص، قابل عمل اقدامات میں توڑنے کے قابل بناتی ہے جو ROS 2 انٹرفیس کے ذریعے انجام دیے جا سکتے ہیں۔

    علمی منصوبہ بندی کا نظام انسانی ارادے اور روبوٹ کے عمل درآمد کے درمیان فرق کو ختم کرتا ہے، جس سے جدید انسانی روبوٹ تعامل ممکن ہوتا ہے جہاں صارف قدرتی زبان میں احکامات جاری کر سکتے ہیں جیسے "باورچی خانے میں میز صاف کریں اور پھر میرے لیے ایک کپ پانی لائیں" بغیر نچلی سطح کے روبوٹک ایکشنز کی وضاحت کی ضرورت کے۔

    ## روبوٹکس میں علمی منصوبہ بندی کو سمجھنا

    ### پلاننگ پائپ لائن

    روبوٹکس میں علمی منصوبہ بندی ایک کثیر مرحلہ عمل کی پیروی کرتی ہے:

    ```
    قدرتی زبان کا حکم → LLM کی تشریح → ٹاسک ڈیکمپوزیشن → ایکشن سیکوینسنگ → ROS 2 ایگزیکیوشن
    ```

    ### اہم اجزاء

    1.  **زبان کی سمجھ**: صارف کے ارادے اور کمانڈ کی ساخت کی تشریح۔
    2.  **ورلڈ ماڈلنگ**: ماحول کی موجودہ حالت کو سمجھنا۔
    3.  **ٹاسک ڈیکمپوزیشن**: پیچیدہ کاموں کو سادہ ذیلی کاموں میں توڑنا۔
    4.  **ایکشن سیکوینسنگ**: انحصار اور رکاوٹوں کی بنیاد پر ایکشنز کو ترتیب دینا۔
    5.  **ایگزیکیوشن مانیٹرنگ**: اس بات کو یقینی بنانا کہ منصوبے پر عمل درآمد توقع کے مطابق ہو۔

    ## روبوٹکس کے لیے بڑے لینگویج ماڈلز (LLMs)

    ### صحیح LLM کا انتخاب

    روبوٹکس میں علمی منصوبہ بندی کے لیے، ان عوامل پر غور کریں:

    *   **استدلال کی صلاحیتیں**: پیچیدہ کاموں کو توڑنے کی صلاحیت۔
    *   **سیاق و سباق کی سمجھ**: مقامی اور وقتی تعلقات کو سمجھنا۔
    *   **ٹول کا استعمال**: بیرونی ٹولز اور APIs کے ساتھ انٹرفیس کرنے کی صلاحیت۔
    *   **حفاظتی رکاوٹیں**: روبوٹ کنٹرول کے لیے بلٹ ان حفاظتی میکانزم۔
    *   **تاخیر کے تقاضے**: ریئل ٹائم تعامل کے لیے ردعمل کا وقت۔

    ### مقبول LLM کے اختیارات

    1.  **OpenAI GPT-4**: بہترین استدلال اور ٹول کے استعمال کی صلاحیتیں۔
    2.  **Anthropic Claude**: مضبوط حفاظتی اور استدلال کی خصوصیات۔
    3.  **اوپن سورس ماڈلز**: Llama 2/3، Mistral، وغیرہ مقامی تعیناتی کے لیے۔
    4.  **خصوصی ماڈلز**: روبوٹکس ایپلی کیشنز کے لیے فائن ٹیونڈ ماڈلز۔

    ## علمی منصوبہ بندی کے نظام کا نفاذ

    ### بنیادی علمی منصوبہ بندی کا آرکیٹیکچر

    ```python
    import openai
    import json
    import time
    from typing import Dict, List, Any, Optional
    from dataclasses import dataclass
    from enum import Enum

    class RobotActionType(Enum):
        NAVIGATION = "navigation"
        MANIPULATION = "manipulation"
        PERCEPTION = "perception"
        INTERACTION = "interaction"
        CONTROL = "control"

    @dataclass
    class RobotAction:
        """ایک ہی روبوٹ ایکشن کی نمائندگی کرتا ہے"""
        action_type: RobotActionType
        action_name: str
        parameters: Dict[str, Any]
        description: str
        estimated_duration: float = 1.0  # سیکنڈ

    class WorldState:
        """دنیا/ماحول کی موجودہ حالت کی نمائندگی کرتا ہے"""
        def __init__(self):
            self.objects = {}  # object_id -> پوزیشن، خصوصیات
            self.robot_position = (0, 0, 0)
            self.robot_orientation = 0.0
            self.robot_state = "idle"  # idle, moving, manipulating, وغیرہ
            self.environment_map = {}  # مقامات اور ان کی خصوصیات
            self.time_of_day = "unknown"
            self.last_updated = time.time()

        def update_object_position(self, obj_id: str, position: tuple):
            """دنیا میں کسی آبجیکٹ کی پوزیشن کو اپ ڈیٹ کریں"""
            if obj_id not in self.objects:
                self.objects[obj_id] = {}
            self.objects[obj_id]["position"] = position
            self.objects[obj_id]["last_seen"] = time.time()

        def get_object_position(self, obj_id: str) -> Optional[tuple]:
            """کسی آبجیکٹ کی پوزیشن حاصل کریں"""
            obj_data = self.objects.get(obj_id, {})
            return obj_data.get("position")

        def update_robot_state(self, position: tuple, orientation: float, state: str):
            """روبوٹ کی حالت کو اپ ڈیٹ کریں"""
            self.robot_position = position
            self.robot_orientation = orientation
            self.robot_state = state
            self.last_updated = time.time()

    class CognitivePlanner:
        """مرکزی علمی منصوبہ بندی کا نظام"""
        def __init__(self, api_key: str = None, model: str = "gpt-4"):
            self.model = model
            self.api_key = api_key
            if api_key:
                openai.api_key = api_key

            self.world_state = WorldState()
            self.action_history = []

            # دستیاب روبوٹ ایکشنز کی وضاحت کریں
            self.available_actions = {
                "move_to_location": {
                    "description": "روبوٹ کو ایک مخصوص مقام پر منتقل کریں",
                    "parameters": {
                        "location": {"type": "string", "description": "ہدف کا مقام کا نام یا کوآرڈینیٹس"},
                        "speed": {"type": "number", "description": "حرکت کی رفتار (0.1-1.0)", "default": 0.5}
                    }
                },
                "pick_up_object": {
                    "description": "ماحول سے ایک آبجیکٹ اٹھائیں",
                    "parameters": {
                        "object_id": {"type": "string", "description": "اٹھانے والے آبجیکٹ کی ID"},
                        "arm": {"type": "string", "description": "کون سا بازو استعمال کرنا ہے", "enum": ["left", "right"], "default": "right"}
                    }
                },
                "place_object": {
                    "description": "ایک مخصوص مقام پر ایک آبجیکٹ رکھیں",
                    "parameters": {
                        "location": {"type": "string", "description": "پلیسمینٹ کے لیے ہدف کا مقام"},
                        "arm": {"type": "string", "description": "کون سا بازو آبجیکٹ کو تھامے ہوئے ہے", "enum": ["left", "right"], "default": "right"}
                    }
                },
                "grasp_object": {
                    "description": "مخصوص قوت کے ساتھ ایک آبجیکٹ کو پکڑیں",
                    "parameters": {
                        "object_id": {"type": "string", "description": "پکڑنے والے آبجیکٹ کی ID"},
                        "force": {"type": "number", "description": "پکڑنے والی قوت (0.0-1.0)", "default": 0.5}
                    }
                },
                "release_object": {
                    "description": "پکڑے ہوئے آبجیکٹ کو چھوڑیں",
                    "parameters": {
                        "arm": {"type": "string", "description": "کون سا بازو چھوڑنا ہے", "enum": ["left", "right"]}
                    }
                },
                "detect_objects": {
                    "description": "موجودہ ماحول میں اشیاء کا پتہ لگائیں",
                    "parameters": {
                        "target_objects": {"type": "array", "items": {"type": "string"}, "description": "تلاش کرنے کے لیے مخصوص اشیاء"},
                        "max_objects": {"type": "integer", "description": "پتہ لگانے کے لیے زیادہ سے زیادہ اشیاء کی تعداد", "default": 10}
                    }
                },
                "navigate_to_object": {
                    "description": "ایک مخصوص آبجیکٹ پر جائیں",
                    "parameters": {
                        "object_id": {"type": "string", "description": "ہدف آبجیکٹ کی ID"},
                        "approach_distance": {"type": "number", "description": "قریب آنے کا فاصلہ (میٹرز)", "default": 0.5}
                    }
                },
                "speak": {
                    "description": "روبوٹ کو بات کروائیں",
                    "parameters": {
                        "text": {"type": "string", "description": "بولنے کا متن"},
                        "language": {"type": "string", "description": "زبان کا کوڈ", "default": "en"}
                    }
                }
            }

        def plan_from_command(self, command: str) -> List[RobotAction]:
            """
            قدرتی زبان کے کمانڈ سے منصوبہ تیار کریں

            Args:
                command: صارف سے قدرتی زبان کا کمانڈ

            Returns:
                منصوبے کی نمائندگی کرنے والے RobotAction اشیاء کی فہرست
            """
            # LLM کے لیے ایک پرامپٹ بنائیں
            prompt = self._create_planning_prompt(command)

            try:
                # منصوبہ تیار کرنے کے لیے LLM کو کال کریں
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    functions=self._get_available_functions(),
                    function_call="auto"
                )

                # جواب کو پارس کریں
                plan = self._parse_llm_response(response)
                return plan

            except Exception as e:
                print(f"Error generating plan: {e}")
                return self._create_fallback_plan(command)

        def _create_planning_prompt(self, command: str) -> str:
            """LLM پلاننگ ٹاسک کے لیے ایک پرامپٹ بنائیں"""
            prompt = f"""
            درج ذیل قدرتی زبان کے کمانڈ کو دیکھتے ہوئے: "{command}"

            موجودہ دنیا کی حالت:
            - روبوٹ کی پوزیشن: {self.world_state.robot_position}
            - روبوٹ کی حالت: {self.world_state.robot_state}
            - معلوم اشیاء: {list(self.world_state.objects.keys())}
            - ماحول: {list(self.world_state.environment_map.keys())}

            براہ کرم اس کمانڈ کو مخصوص روبوٹ ایکشنز کے سلسلے میں توڑ دیں۔
            ہر ایکشن دستیاب فنکشنز میں سے ایک ہونا چاہیے۔
            مقامی تعلقات اور ٹاسک کے انحصار پر غور کریں۔

            JSON فارمیٹ میں ایکشنز کا سلسلہ جواب دیں۔
            """
            return prompt

        def _get_system_prompt(self) -> str:
            """LLM کے لیے سسٹم پرامپٹ حاصل کریں"""
            return """
            آپ ایک ماہر روبوٹ ٹاسک پلانر ہیں۔ آپ کا کام قدرتی زبان کے کمانڈز کو مخصوص، قابل عمل روبوٹ ایکشنز میں توڑنا ہے۔
            درج ذیل پر غور کریں:
            1. مقامی تعلقات اور نیویگیشن کی ضروریات
            2. آبجیکٹ مینیپولیشن کی ضروریات
            3. ایکشنز کے درمیان ترتیب وار انحصار
            4. حفاظتی رکاوٹیں
            5. ایکشن کے سلسلے کی کارکردگی

            ہمیشہ ایکشن کے سلسلے پر مشتمل درست JSON کے ساتھ جواب دیں۔
            """

        def _get_available_functions(self) -> List[Dict]:
            """LLM کے لیے دستیاب فنکشنز کی فہرست حاصل کریں"""
            functions = []
            for action_name, action_info in self.available_actions.items():
                function_def = {
                    "name": action_name,
                    "description": action_info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": action_info["parameters"],
                        "required": []  # پیرامیٹرز کی بنیاد پر بھرا جائے گا
                    }
                }

                # مطلوبہ پیرامیٹرز کا تعین کریں (جن میں ڈیفالٹ نہیں ہے)
                for param_name, param_info in action_info["parameters"].items():
                    if "default" not in param_info:
                        function_def["parameters"]["required"].append(param_name)

                functions.append(function_def)

            return functions

        def _parse_llm_response(self, response) -> List[RobotAction]:
            """LLM جواب کو RobotAction اشیاء میں پارس کریں"""
            plan = []

            # جواب سے فنکشن کالز نکالیں
            if hasattr(response.choices[0].message, 'function_call'):
                # سنگل فنکشن کال کو ہینڈل کریں
                call = response.choices[0].message.function_call
                action = self._convert_function_call_to_action(call)
                if action:
                    plan.append(action)
            elif hasattr(response.choices[0].message, 'tool_calls'):
                # متعدد ٹول کالز کو ہینڈل کریں (نئی API)
                for tool_call in response.choices[0].message.tool_calls:
                    action = self._convert_function_call_to_action(tool_call.function)
                    if action:
                        plan.append(action)

            return plan

        def _convert_function_call_to_action(self, function_call) -> Optional[RobotAction]:
            """LLM فنکشن کال کو RobotAction میں تبدیل کریں"""
            try:
                action_name = function_call.name
                arguments = json.loads(function_call.arguments)

                # RobotActionType پر نقشہ بنائیں
                action_type_map = {
                    "move_to_location": RobotActionType.NAVIGATION,
                    "pick_up_object": RobotActionType.MANIPULATION,
                    "place_object": RobotActionType.MANIPULATION,
                    "grasp_object": RobotActionType.MANIPULATION,
                    "release_object": RobotActionType.MANIPULATION,
                    "detect_objects": RobotActionType.PERCEPTION,
                    "navigate_to_object": RobotActionType.NAVIGATION,
                    "speak": RobotActionType.INTERACTION
                }

                action_type = action_type_map.get(action_name, RobotActionType.CONTROL)

                # ایکشن کی تفصیل بنائیں
                description = f"{action_name} with parameters: {arguments}"

                return RobotAction(
                    action_type=action_type,
                    action_name=action_name,
                    parameters=arguments,
                    description=description
                )
            except Exception as e:
                print(f"Error converting function call to action: {e}")
                return None

        def _create_fallback_plan(self, command: str) -> List[RobotAction]:
            """اگر LLM ناکام ہو جائے تو فال بیک منصوبہ بنائیں"""
            print(f"Using fallback plan for command: {command}")

            # کمانڈ کے مطلوبہ الفاظ کی بنیاد پر سادہ فال بیک
            if "move" in command.lower() or "go" in command.lower():
                return [RobotAction(
                    action_type=RobotActionType.NAVIGATION,
                    action_name="move_to_location",
                    parameters={"location": "default"},
                    description="فال بیک حرکت کا ایکشن"
                )]
            elif "pick" in command.lower() or "grasp" in command.lower():
                return [RobotAction(
                    action_type=RobotActionType.MANIPULATION,
                    action_name="pick_up_object",
                    parameters={"object_id": "unknown"},
                    description="فال بیک اٹھانے کا ایکشن"
                )]
            else:
                return [RobotAction(
                    action_type=RobotActionType.INTERACTION,
                    action_name="speak",
                    parameters={"text": "میں کمانڈ کو سمجھ نہیں پا رہا ہوں۔ کیا آپ براہ کرم دوبارہ الفاظ استعمال کر سکتے ہیں؟"},
                    description="فال بیک مواصلاتی ایکشن"
                )]

        def update_world_state(self, state_update: Dict[str, Any]):
            """نئی معلومات کے ساتھ دنیا کی حالت کو اپ ڈیٹ کریں"""
            for key, value in state_update.items():
                if hasattr(self.world_state, key):
                    setattr(self.world_state, key, value)
            self.world_state.last_updated = time.time()

    # استعمال کی مثال
def main():
    # علمی منصوبہ ساز کو شروع کریں (آپ کو اپنی OpenAI API کی فراہم کرنی ہوگی)
    planner = CognitivePlanner(api_key="your-api-key-here")

    # مثال کے کمانڈز
    commands = [
        "باورچی خانے میں جاؤ اور میرے لیے ایک کپ لاؤ",
        "سرخ گیند تلاش کرو اور اسے اٹھاؤ",
        "میز پر جاؤ اور کپ وہاں رکھو"
    ]

    for command in commands:
        print(f"\nCommand: {command}")
        plan = planner.plan_from_command(command)

        print("Generated plan:")
        for i, action in enumerate(plan):
            print(f"  {i+1}. {action.description}")

if __name__ == "__main__":
    main()
    ```

    ## سیاق و سباق کے ساتھ جدید علمی منصوبہ بندی

    ### سیاق و سباق سے آگاہ منصوبہ بندی

    ```python
    class ContextAwarePlanner(CognitivePlanner):
        def __init__(self, api_key: str = None, model: str = "gpt-4"):
            super().__init__(api_key, model)

            # اضافی سیاق و سباق کی معلومات
            self.conversation_history = []
            self.user_preferences = {}
            self.task_context = {}  # موجودہ ٹاسک کے لیے سیاق و سباق
            self.safety_constraints = []

        def plan_with_context(self, command: str, context: Dict[str, Any] = None) -> List[RobotAction]:
            """
            اضافی سیاق و سباق کی معلومات پر غور کرتے ہوئے منصوبہ تیار کریں

            Args:
                command: قدرتی زبان کا کمانڈ
                context: اضافی سیاق و سباق کی معلومات

            Returns:
                RobotAction اشیاء کی فہرست
            """
            # فراہم کردہ سیاق و سباق کو اندرونی سیاق و سباق کے ساتھ ضم کریں
            full_context = self._merge_context(context or {})

            # سیاق و سباق کے ساتھ بہتر پرامپٹ بنائیں
            prompt = self._create_contextual_planning_prompt(command, full_context)

            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_contextual_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    functions=self._get_available_functions(),
                    function_call="auto",
                    temperature=0.3  # زیادہ مستقل منصوبہ بندی کے لیے کم درجہ حرارت
                )

                plan = self._parse_llm_response(response)

                # منصوبے پر حفاظتی رکاوٹیں لاگو کریں
                safe_plan = self._apply_safety_constraints(plan)

                return safe_plan

            except Exception as e:
                print(f"Error generating contextual plan: {e}")
                return self._create_fallback_plan(command)

        def _merge_context(self, external_context: Dict[str, Any]) -> Dict[str, Any]:
            """بیرونی سیاق و سباق کو اندرونی سیاق و سباق کے ساتھ ضم کریں"""
            full_context = {
                "world_state": self.world_state.__dict__.copy(),
                "conversation_history": self.conversation_history[-5:],  # آخری 5 تبادلے
                "user_preferences": self.user_preferences,
                "task_context": self.task_context,
                "safety_constraints": self.safety_constraints,
                "current_time": time.time(),
                "robot_capabilities": self._get_robot_capabilities()
            }

            # بیرونی سیاق و سباق کے ساتھ اپ ڈیٹ کریں
            full_context.update(external_context)

            return full_context

        def _create_contextual_planning_prompt(self, command: str, context: Dict[str, Any]) -> str:
            """ایک امیر سیاق و سباق کے ساتھ منصوبہ بندی کا پرامپٹ بنائیں"""
            prompt = f"""
            قدرتی زبان کا کمانڈ: "{command}"

            سیاق و سباق کی معلومات:
            - موجودہ وقت: {context.get('current_time', 'unknown')}
            - روبوٹ کی صلاحیتیں: {context.get('robot_capabilities', {})}
            - حفاظتی رکاوٹیں: {context.get('safety_constraints', [])}

            دنیا کی حالت:
            - روبوٹ کی پوزیشن: {context['world_state'].get('robot_position', 'unknown')}
            - روبوٹ کی حالت: {context['world_state'].get('robot_state', 'unknown')}
            - معلوم اشیاء: {list(context['world_state']['objects'].keys()) if context['world_state'].get('objects') else []}
            - ماحولیاتی نقشہ: {list(context['world_state']['environment_map'].keys()) if context['world_state'].get('environment_map') else []}

            صارف کی ترجیحات:
            - {context.get('user_preferences', {})}

            پچھلے تعاملات:
            - {context.get('conversation_history', [])}

            ٹاسک کے لیے مخصوص سیاق و سباق:
            - {context.get('task_context', {})}

            منصوبہ بندی کی ضروریات:
            1. منصوبہ تیار کرتے وقت تمام حفاظتی رکاوٹوں پر غور کریں۔
            2. صارف کی ترجیحات اور پچھلے تعاملات کو مدنظر رکھیں۔
            3. یقینی بنائیں کہ روبوٹ کی صلاحیتوں کے پیش نظر ایکشنز قابل عمل ہیں۔
            4. مقامی اور وقتی ہم آہنگی برقرار رکھیں۔
            5. اگر وضاحت کی ضرورت ہو تو ممکنہ ابہامات کو ہینڈل کریں۔

            کمانڈ کو پورا کرنے کے لیے ایکشنز کا ایک تفصیلی سلسلہ تیار کریں۔
            ہر ایکشن مخصوص اور قابل عمل ہونا چاہیے۔
            اگر وضاحت کی ضرورت ہو، تو 'request_clarification' ایکشن شامل کریں۔
            """
            return prompt

        def _get_contextual_system_prompt(self) -> str:
            """سیاق و سباق کی منصوبہ بندی کے لیے سسٹم پرامپٹ حاصل کریں"""
            return """
            آپ ایک جدید روبوٹ ٹاسک پلانر ہیں جس کے پاس بھرپور سیاق و سباق کی معلومات تک رسائی ہے۔
            آپ کا کردار محفوظ، موثر، اور سیاق و سباق کے مطابق ایکشن کے سلسلے تیار کرنا ہے۔

            منصوبہ بندی کرتے وقت، درج ذیل پر غور کریں:
            1. حفاظتی رکاوٹیں اور خطرناک ایکشنز سے گریز کریں۔
            2. صارف کی ترجیحات اور ماضی کے تعاملات۔
            3. موجودہ دنیا کی حالت اور مقامی تعلقات۔
            4. روبوٹ کی صلاحیتیں اور حدود۔
            5. ٹاسک کے لیے مخصوص سیاق و سباق اور ضروریات۔

            ہمیشہ حفاظت اور صارف کی ترجیحات کو ترجیح دیں۔
            اگر کمانڈ مبہم ہے یا وضاحت کی ضرورت ہے، تو وضاحت کی درخواست کریں۔
            """

        def _get_robot_capabilities(self) -> Dict[str, Any]:
            """روبوٹ کی صلاحیتوں کی معلومات حاصل کریں"""
            return {
                "navigation": {
                    "max_speed": 0.5,  # m/s
                    "min_turn_radius": 0.3,  # میٹر
                    "max_gradient": 15.0  # ڈگری
                },
                "manipulation": {
                    "max_payload": 2.0,  # kg
                    "reach_distance": 1.2,  # میٹر
                    "precision": "medium"
                },
                "perception": {
                    "camera_range": 5.0,  # میٹر
                    "object_detection_accuracy": 0.85
                },
                "interaction": {
                    "speech_synthesis": True,
                    "language_support": ["en", "es", "fr"]
                }
            }

        def _apply_safety_constraints(self, plan: List[RobotAction]) -> List[RobotAction]:
            """منصوبے پر حفاظتی رکاوٹیں لاگو کریں"""
            safe_plan = []

            for action in plan:
                if self._is_action_safe(action):
                    safe_plan.append(action)
                else:
                    print(f"Action filtered out for safety: {action.description}")
                    # اس کے بجائے حفاظتی ایکشن شامل کریں
                    safety_action = RobotAction(
                        action_type=RobotActionType.INTERACTION,
                        action_name="speak",
                        parameters={"text": "میں حفاظتی وجوہات کی بنا پر یہ ایکشن انجام نہیں دے سکتا۔"},
                        description="حفاظتی وارننگ"
                    )
                    safe_plan.append(safety_action)

            return safe_plan

        def _is_action_safe(self, action: RobotAction) -> bool:
            """چیک کریں کہ آیا ایکشن انجام دینا محفوظ ہے"""
            # حفاظتی رکاوٹوں کے خلاف چیک کریں
            for constraint in self.safety_constraints:
                if constraint.get("type") == "forbidden_action":
                    if action.action_name in constraint.get("actions", []):
                        return False
                elif constraint.get("type") == "spatial_constraint":
                    # چیک کریں کہ آیا ایکشن مقامی رکاوٹوں کی خلاف ورزی کرتا ہے
                    if action.action_name == "move_to_location":
                        target = action.parameters.get("location")
                        forbidden_areas = constraint.get("forbidden_areas", [])
                        if target in forbidden_areas:
                            return False

            return True

        def add_safety_constraint(self, constraint: Dict[str, Any]):
            """ایک حفاظتی رکاوٹ شامل کریں"""
            self.safety_constraints.append(constraint)

        def update_user_preferences(self, preferences: Dict[str, Any]):
            """صارف کی ترجیحات کو اپ ڈیٹ کریں"""
            self.user_preferences.update(preferences)
    ```

    ## ROS 2 کے ساتھ انٹیگریشن

    ### ROS 2 علمی منصوبہ بندی کا نوڈ

    ```python
    import rclpy
    import json
    from rclpy.node import Node
    from std_msgs.msg import String
    from geometry_msgs.msg import Twist, Pose, Point
    from action_msgs.msg import GoalStatus
    from rclpy.action import ActionClient
    from nav2_msgs.action import NavigateToPose
    from builtin_interfaces.msg import Time

    class CognitivePlanningNode(Node):
        def __init__(self):
            super().__init__('cognitive_planning_node')

            # علمی منصوبہ ساز کو شروع کریں
            self.planner = ContextAwarePlanner()

            # کمانڈز اور دنیا کی حالت کی تازہ کاریوں کے لیے سبسکرائبرز
            self.command_sub = self.create_subscription(
                String,
                '/natural_language_command',
                self.command_callback,
                10
            )

            self.world_state_sub = self.create_subscription(
                String,  # عملی طور پر، آپ کسٹم میسج کی قسم استعمال کریں گے
                '/world_state_update',
                self.world_state_callback,
                10
            )

            # روبوٹ کمانڈز اور اسٹیٹس کے لیے پبلشرز
            self.action_pub = self.create_publisher(String, '/robot_action', 10)
            self.status_pub = self.create_publisher(String, '/planning_status', 10)

            # نیویگیشن کے لیے ایکشن کلائنٹس
            self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

            # منصوبہ بندی کی حالت
            self.current_plan = []
            self.plan_index = 0
            self.is_executing = False

            self.get_logger().info('Cognitive Planning Node initialized')

        def command_callback(self, msg):
            """قدرتی زبان کے کمانڈ کو ہینڈل کریں"""
            command = msg.data
            self.get_logger().info(f'Received command: {command}')

            # منصوبہ تیار کریں
            self.current_plan = self.planner.plan_with_context(command)
            self.plan_index = 0
            self.is_executing = True

            # منصوبہ بندی کی حالت شائع کریں
            status_msg = String()
            status_msg.data = f'Generated plan with {len(self.current_plan)} actions'
            self.status_pub.publish(status_msg)

            # منصوبہ پر عمل کرنا شروع کریں
            self.execute_next_action()

        def world_state_callback(self, msg):
            """دنیا کی حالت کی تازہ کاریوں کو ہینڈل کریں"""
            try:
                state_update = json.loads(msg.data)
                self.planner.update_world_state(state_update)
                self.get_logger().info('World state updated')
            except json.JSONDecodeError:
                self.get_logger().error('Invalid world state JSON')

        def execute_next_action(self):
            """منصوبے میں اگلے ایکشن پر عمل کریں"""
            if not self.current_plan or self.plan_index >= len(self.current_plan) or not self.is_executing:
                # منصوبہ مکمل
                self.is_executing = False
                self.plan_index = 0
                self.current_plan = []

                status_msg = String()
                status_msg.data = 'Plan completed'
                self.status_pub.publish(status_msg)
                return

            action = self.current_plan[self.plan_index]
            self.get_logger().info(f'Executing action {self.plan_index + 1}: {action.action_name}')

            # ایکشن کی قسم کی بنیاد پر عمل کریں
            if action.action_type == RobotActionType.NAVIGATION:
                self.execute_navigation_action(action)
            elif action.action_type == RobotActionType.MANIPULATION:
                self.execute_manipulation_action(action)
            elif action.action_type == RobotActionType.PERCEPTION:
                self.execute_perception_action(action)
            elif action.action_type == RobotActionType.INTERACTION:
                self.execute_interaction_action(action)
            else:
                self.execute_generic_action(action)

        def execute_navigation_action(self, action):
            """نیویگیشن ایکشن پر عمل کریں"""
            if action.action_name == "move_to_location":
                location = action.parameters.get("location", "unknown")

                # حقیقی نفاذ میں، آپ نامزد مقام کے لیے کوآرڈینیٹس تلاش کریں گے
                # نقشے سے
                target_pose = self.get_location_pose(location)

                if target_pose:
                    self.send_navigation_goal(target_pose)
                else:
                    self.get_logger().error(f'Unknown location: {location}')
                    self.plan_index += 1
                    if self.is_executing:
                        self.execute_next_action()

        def get_location_pose(self, location_name: str) -> Optional[Pose]:
            """نامزد مقام کے لیے پوز حاصل کریں"""
            # عملی طور پر، یہ نقشے سے کوآرڈینیٹس تلاش کرے گا
            # فی الحال، ایک ڈیفالٹ پوز یا None واپس کریں
            location_map = {
                "kitchen": Pose(position=Point(x=3.0, y=1.0, z=0.0)),
                "living_room": Pose(position=Point(x=1.0, y=2.0, z=0.0)),
                "bedroom": Pose(position=Point(x=4.0, y=3.0, z=0.0)),
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
            try:
                goal_result = future.result()

                if goal_result.status == GoalStatus.STATUS_SUCCEEDED:
                    self.get_logger().info('Navigation succeeded')
                    self.plan_index += 1
                    if self.is_executing:
                        self.execute_next_action()
                else:
                    self.get_logger().error(f'Navigation failed with status: {goal_result.status}')
                    # عمل درآمد بند کریں یا ناکامی کو ہینڈل کریں
                    self.is_executing = False
            except Exception as e:
                self.get_logger().error(f'Error in navigation result callback: {e}')
                self.is_executing = False

        def execute_manipulation_action(self, action):
            """مینیپولیشن ایکشن پر عمل کریں"""
            # حقیقی نفاذ میں، یہ مینیپولیشن کنٹرولرز کے ساتھ انٹرفیس کرے گا
            action_msg = String()
            action_msg.data = f"manipulation:{action.action_name}:{json.dumps(action.parameters)}"
            self.action_pub.publish(action_msg)

            # ایکشن کی تکمیل کو سیمولیٹ کریں
            self.get_logger().info(f'Manipulation action sent: {action.action_name}')
            self.plan_index += 1
            if self.is_executing:
                self.execute_next_action()

        def execute_perception_action(self, action):
            """پرسیپشن ایکشن پر عمل کریں"""
            # حقیقی نفاذ میں، یہ پرسیپشن ماڈیولز کو متحرک کرے گا
            action_msg = String()
            action_msg.data = f"perception:{action.action_name}:{json.dumps(action.parameters)}"
            self.action_pub.publish(action_msg)

            # ایکشن کی تکمیل کو سیمولیٹ کریں
            self.get_logger().info(f'Perception action sent: {action.action_name}')
            self.plan_index += 1
            if self.is_executing:
                self.execute_next_action()

        def execute_interaction_action(self, action):
            """تعامل ایکشن پر عمل کریں"""
            if action.action_name == "speak":
                text = action.parameters.get("text", "")
                self.get_logger().info(f'Robot says: {text}')

                # اسپیچ کمانڈ شائع کریں
                action_msg = String()
                action_msg.data = f"interaction:speak:{text}"
                self.action_pub.publish(action_msg)

            self.plan_index += 1
            if self.is_executing:
                self.execute_next_action()

        def execute_generic_action(self, action):
            """عام ایکشن پر عمل کریں"""
            action_msg = String()
            action_msg.data = f"generic:{action.action_name}:{json.dumps(action.parameters)}"
            self.action_pub.publish(action_msg)

            self.get_logger().info(f'Generic action sent: {action.action_name}')
            self.plan_index += 1
            if self.is_executing:
                self.execute_next_action()

def main(args=None):
    rclpy.init(args=args)
    node = CognitivePlanningNode()

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

    ## جدید منصوبہ بندی کی تکنیکیں

    ### ہائیرارکیکل ٹاسک نیٹ ورک (HTN) منصوبہ بندی

    ```python
    class HTNPlanner:
        """پیچیدہ ٹاسک ڈیکمپوزیشن کے لیے ہائیرارکیکل ٹاسک نیٹ ورک پلانر"""

        def __init__(self):
            self.tasks = {}
            self.methods = {}
            self.primitive_actions = set()

            # بنیادی ٹاسکس اور طریقے بیان کریں
            self._define_basic_tasks()

        def _define_basic_tasks(self):
            """بنیادی ٹاسکس اور ان کے ڈیکمپوزیشن کے طریقے بیان کریں"""
            # اعلیٰ سطحی ٹاسکس
            self.tasks = {
                "clean_room": {"description": "پورا کمرہ صاف کریں"},
                "serve_drink": {"description": "ایک شخص کو مشروب پیش کریں"},
                "organize_objects": {"description": "ایک علاقے میں اشیاء کو منظم کریں"}
            }

            # ٹاسکس کو توڑنے کے طریقے
            self.methods = {
                "clean_room": [
                    {
                        "name": "clean_room_method_1",
                        "preconditions": [],
                        "decomposition": [
                            {"task": "detect_objects", "parameters": {"target_objects": ["کوڑا کرکٹ"]}},
                            {"task": "pick_up_object", "parameters": {"object_type": "کوڑا کرکٹ"}},
                            {"task": "move_to_location", "parameters": {"location": "کوڑے دان"}},
                            {"task": "release_object", "parameters": {}}
                        ]
                    }
                ],
                "serve_drink": [
                    {
                        "name": "serve_drink_method_1",
                        "preconditions": [],
                        "decomposition": [
                            {"task": "detect_objects", "parameters": {"target_objects": ["کپ"]}},
                            {"task": "pick_up_object", "parameters": {"object_id": "کپ"}},
                            {"task": "move_to_location", "parameters": {"location": "پانی کا ذریعہ"}},
                            {"task": "fill_container", "parameters": {"container_id": "کپ"}},
                            {"task": "move_to_location", "parameters": {"location": "شخص کا مقام"}},
                            {"task": "place_object", "parameters": {"location": "میز"}}
                        ]
                    }
                ]
            }

            # بنیادی ایکشنز (درجہ بندی میں لیف نوڈز)
            self.primitive_actions = {
                "move_to_location", "pick_up_object", "place_object",
                "release_object", "detect_objects", "fill_container"
            }

        def decompose_task(self, task_name: str, parameters: Dict[str, Any] = None) -> List[RobotAction]:
            """ایک اعلیٰ سطحی ٹاسک کو بنیادی ایکشنز میں توڑیں"""
            if task_name in self.primitive_actions:
                # یہ پہلے ہی ایک بنیادی ایکشن ہے
                return [RobotAction(
                    action_type=self._get_action_type(task_name),
                    action_name=task_name,
                    parameters=parameters or {},
                    description=f"بنیادی ایکشن: {task_name}"
                )]

            if task_name not in self.methods:
                raise ValueError(f"ٹاسک کے لیے کوئی طریقہ بیان نہیں کیا گیا: {task_name}")

            # پہلا دستیاب طریقہ استعمال کریں (عملی طور پر، آپ کے پاس انتخاب کی منطق ہوگی)
            method = self.methods[task_name][0]

            plan = []
            for subtask in method["decomposition"]:
                subtask_name = subtask["task"]
                subtask_params = {**subtask.get("parameters", {}), **(parameters or {})}

                subplan = self.decompose_task(subtask_name, subtask_params)
                plan.extend(subplan)

            return plan

        def _get_action_type(self, action_name: str) -> RobotActionType:
            """ایکشن کے نام کو ایکشن کی قسم سے نقشہ بنائیں"""
            navigation_actions = {"move_to_location", "navigate_to_object"}
            manipulation_actions = {"pick_up_object", "place_object", "grasp_object", "release_object", "fill_container"}
            perception_actions = {"detect_objects"}
            interaction_actions = {"speak"}

            if action_name in navigation_actions:
                return RobotActionType.NAVIGATION
            elif action_name in manipulation_actions:
                return RobotActionType.MANIPULATION
            elif action_name in perception_actions:
                return RobotActionType.PERCEPTION
            elif action_name in interaction_actions:
                return RobotActionType.INTERACTION
            else:
                return RobotActionType.CONTROL

    # علمی منصوبہ ساز کے ساتھ انٹیگریشن
    class AdvancedCognitivePlanner(ContextAwarePlanner):
        def __init__(self, api_key: str = None, model: str = "gpt-4"):
            super().__init__(api_key, model)
            self.hierarchical_planner = HTNPlanner()

        def plan_with_hierarchical_decomposition(self, command: str) -> List[RobotAction]:
            """درجہ بندی ٹاسک ڈیکمپوزیشن کا استعمال کرتے ہوئے منصوبہ بنائیں"""
            # پہلے، یہ شناخت کرنے کی کوشش کریں کہ کیا یہ اعلیٰ سطحی ٹاسک ہے
            high_level_task = self._identify_high_level_task(command)

            if high_level_task:
                # ڈیکمپوزیشن کے لیے درجہ بندی منصوبہ ساز کا استعمال کریں
                try:
                    plan = self.hierarchical_planner.decompose_task(
                        high_level_task["task_name"],
                        high_level_task.get("parameters", {})
                    )
                    return plan
                except ValueError:
                    # اگر درجہ بندی کا طریقہ دستیاب نہ ہو تو LLM منصوبہ بندی پر واپس جائیں
                    pass

            # LLM پر مبنی منصوبہ بندی پر واپس جائیں
            return self.plan_with_context(command)

        def _identify_high_level_task(self, command: str) -> Optional[Dict[str, Any]]:
            """شناخت کریں کہ کیا کمانڈ اعلیٰ سطحی ٹاسک سے مطابقت رکھتا ہے"""
            command_lower = command.lower()

            # سادہ کلیدی الفاظ پر مبنی شناخت (عملی طور پر، زیادہ نفیس NLU استعمال کریں)
            if "clean" in command_lower:
                return {"task_name": "clean_room", "parameters": {}}
            elif any(word in command_lower for word in ["serve", "bring", "get"]):
                if any(word in command_lower for word in ["water", "drink", "beverage"]):
                    return {"task_name": "serve_drink", "parameters": {}}

            return None
    ```

    ## حفاظت اور توثیق

    ### منصوبہ کی توثیق اور حفاظتی جانچ

    ```python
    class PlanValidator:
        """حفاظت اور فزیبلٹی کے لیے منصوبوں کی توثیق کریں"""

        def __init__(self):
            self.safety_rules = []
            self.feasibility_constraints = []
            self._setup_default_rules()

        def _setup_default_rules(self):
            """پہلے سے طے شدہ حفاظتی اور فزیبلٹی کے قواعد مرتب کریں"""
            # حفاظتی قواعد
            self.safety_rules = [
                self._no_navigation_to_forbidden_areas,
                self._no_manipulation_of_heavy_objects,
                self._no_high_speed_navigation_in_crowded_areas
            ]

            # فزیبلٹی کی رکاوٹیں
            self.feasibility_constraints = [
                self._check_robot_reachability,
                self._check_payload_limits,
                self._check_battery_level
            ]

        def validate_plan(self, plan: List[RobotAction], context: Dict[str, Any]) -> Dict[str, Any]:
            """
            حفاظت اور فزیبلٹی کے لیے منصوبے کی توثیق کریں

            Returns:
                توثیق کے نتائج کے ساتھ لغت
            """
            results = {
                "is_valid": True,
                "safety_issues": [],
                "feasibility_issues": [],
                "warnings": [],
                "modified_plan": plan.copy()
            }

            # حفاظت کی جانچ کریں
            for rule in self.safety_rules:
                rule_result = rule(plan, context)
                if not rule_result["is_safe"]:
                    results["is_valid"] = False
                    results["safety_issues"].extend(rule_result["issues"])

            # فزیبلٹی کی جانچ کریں
            for constraint in self.feasibility_constraints:
                constraint_result = constraint(plan, context)
                if not constraint_result["is_feasible"]:
                    results["is_valid"] = False
                    results["feasibility_issues"].extend(constraint_result["issues"])

            # اگر ضرورت ہو تو ترامیم لاگو کریں
            if results["safety_issues"] or results["feasibility_issues"]:
                results["modified_plan"] = self._modify_plan_for_safety(
                    plan, results["safety_issues"], results["feasibility_issues"]
                )

            return results

        def _no_navigation_to_forbidden_areas(self, plan: List[RobotAction], context: Dict[str, Any]) -> Dict[str, Any]:
            """چیک کریں کہ نیویگیشن ممنوعہ علاقوں میں نہیں جاتی ہے"""
            issues = []

            forbidden_areas = context.get("safety_constraints", {}).get("forbidden_areas", [])

            for i, action in enumerate(plan):
                if (action.action_name == "move_to_location" and
                    action.parameters.get("location") in forbidden_areas):
                    issues.append(f"ایکشن {i}: ممنوعہ علاقے '{action.parameters['location']}' میں نیویگیشن")

            return {
                "is_safe": len(issues) == 0,
                "issues": issues
            }

        def _no_manipulation_of_heavy_objects(self, plan: List[RobotAction], context: Dict[str, Any]) -> Dict[str, Any]:
            """چیک کریں کہ روبوٹ اپنی پے لوڈ کی صلاحیت سے زیادہ بھاری اشیاء کو ہیرا پھیری کرنے کی کوشش نہیں کرتا ہے"""
            issues = []
            max_payload = context.get("robot_capabilities", {}).get("manipulation", {}).get("max_payload", 2.0)

            for i, action in enumerate(plan):
                if action.action_name == "pick_up_object":
                    # حقیقی نظام میں، آپ آبجیکٹ کا وزن تلاش کریں گے
                    object_weight = self._get_object_weight(action.parameters.get("object_id", ""))
                    if object_weight > max_payload:
                        issues.append(f"ایکشن {i}: {object_weight}kg وزنی آبجیکٹ اٹھانے کی کوشش کر رہا ہے، "
                                    f"پے لوڈ کی صلاحیت {max_payload}kg سے زیادہ ہے")

            return {
                "is_safe": len(issues) == 0,
                "issues": issues
            }

        def _check_robot_reachability(self, plan: List[RobotAction], context: Dict[str, Any]) -> Dict[str, Any]:
            """چیک کریں کہ مینیپولیشن کے اہداف روبوٹ کی پہنچ میں ہیں"""
            issues = []
            max_reach = context.get("robot_capabilities", {}).get("manipulation", {}).get("reach_distance", 1.2)

            # اس کے لیے مقامی استدلال کی ضرورت ہوگی تاکہ یہ چیک کیا جا سکے کہ آیا اہداف قابل رسائی ہیں
            # فی الحال، ہم ایک سادہ چیک کے ساتھ سیمولیٹ کریں گے
            for i, action in enumerate(plan):
                if action.action_name == "navigate_to_object":
                    # حقیقی نظام میں، آپ آبجیکٹ کا فاصلہ حساب کریں گے
                    distance = self._estimate_distance_to_object(action.parameters.get("object_id"))
                    if distance > max_reach:
                        issues.append(f"ایکشن {i}: آبجیکٹ پہنچ سے باہر ہے (فاصلہ: {distance:.2f}m، زیادہ سے زیادہ: {max_reach}m)")

            return {
                "is_feasible": len(issues) == 0,
                "issues": issues
            }

        def _modify_plan_for_safety(self, plan: List[RobotAction], safety_issues: List[str],
                                  feasibility_issues: List[str]) -> List[RobotAction]:
            """حفاظتی اور فزیبلٹی کے مسائل کو حل کرنے کے لیے منصوبے میں ترمیم کریں"""
            modified_plan = plan.copy()

            # سادہ ترمیم کا طریقہ: غیر محفوظ ایکشنز کو محفوظ متبادلات سے تبدیل کریں
            # عملی طور پر، آپ کے پاس زیادہ نفیس ری پلاننگ ہوگی

            for issue in safety_issues + feasibility_issues:
                # مسائل کی شناخت کے لیے مسئلے کو پارس کریں اور انہیں تبدیل کریں
                pass  # نفاذ مسئلے کے فارمیٹ پر منحصر ہوگا

            return modified_plan

        def _get_object_weight(self, object_id: str) -> float:
            """کسی آبجیکٹ کا وزن حاصل کریں (سیمولیٹڈ)"""
            # حقیقی نظام میں، یہ ایک نالج بیس کو استفسار کرے گا
            # سیمولیشن کے لیے، بے ترتیب وزن واپس کریں
            import random
            return random.uniform(0.1, 5.0)  # 0.1 سے 5.0 کلوگرام

        def _estimate_distance_to_object(self, object_id: str) -> float:
            """کسی آبجیکٹ کا فاصلہ کا اندازہ لگائیں (سیمولیٹڈ)"""
            # حقیقی نظام میں، یہ مقامی استدلال کا استعمال کرے گا
            import random
            return random.uniform(0.5, 3.0)  # 0.5 سے 3.0 میٹر
    ```

    ## کارکردگی کی اصلاح

    ### کیشنگ اور آپٹیمائزیشن کی تکنیکیں

    ```python
    import functools
    import hashlib
    from typing import Tuple

    class OptimizedCognitivePlanner(AdvancedCognitivePlanner):
        def __init__(self, api_key: str = None, model: str = "gpt-4"):
            super().__init__(api_key, model)

            # منصوبہ کیشنگ
            self.plan_cache = {}
            self.cache_size_limit = 100

            # کارکردگی کی نگرانی
            self.plan_generation_times = []

        def plan_from_command(self, command: str, use_cache: bool = True) -> List[RobotAction]:
            """کیشنگ اور کارکردگی کی نگرانی کے ساتھ منصوبہ تیار کریں"""
            start_time = time.time()

            # کیش کی کلید بنائیں
            cache_key = self._create_cache_key(command, self.world_state)

            if use_cache and cache_key in self.plan_cache:
                # کیشڈ منصوبہ واپس کریں
                cached_plan, generation_time = self.plan_cache[cache_key]
                self.get_logger().info(f"کیش سے منصوبہ حاصل کیا گیا (وقت: {generation_time:.2f}s)")
                self._update_performance_stats(time.time() - start_time, cached=True)
                return cached_plan

            # نیا منصوبہ تیار کریں
            plan = super().plan_from_command(command)

            # نئے منصوبے کو کیش کریں
            generation_time = time.time() - start_time
            self._cache_plan(cache_key, plan, generation_time)

            self._update_performance_stats(generation_time, cached=False)

            return plan

        def _create_cache_key(self, command: str, world_state) -> str:
            """کمانڈ اور دنیا کی حالت کے لیے کیش کی کلید بنائیں"""
            state_str = str(sorted(world_state.__dict__.items()))
            combined = f"{command}|{state_str}"
            return hashlib.md5(combined.encode()).hexdigest()

        def _cache_plan(self, cache_key: str, plan: List[RobotAction], generation_time: float):
            """ایک تیار کردہ منصوبے کو کیش کریں"""
            # اگر کیش بھرا ہوا ہو تو سب سے پرانی اندراجات کو ہٹائیں
            if len(self.plan_cache) >= self.cache_size_limit:
                # سب سے پرانی اندراج کو ہٹائیں (حقیقی نظام میں، آپ LRU استعمال کر سکتے ہیں)
                oldest_key = next(iter(self.plan_cache))
                del self.plan_cache[oldest_key]

            self.plan_cache[cache_key] = (plan, generation_time)

        def _update_performance_stats(self, generation_time: float, cached: bool):
            """کارکردگی کے اعدادوشمار کو اپ ڈیٹ کریں"""
            if not cached:
                self.plan_generation_times.append(generation_time)

                # صرف حالیہ پیمائشیں رکھیں (آخری 100)
                if len(self.plan_generation_times) > 100:
                    self.plan_generation_times = self.plan_generation_times[-100:]

        def get_performance_stats(self) -> Dict[str, float]:
            """کارکردگی کے اعدادوشمار حاصل کریں"""
            if not self.plan_generation_times:
                return {"avg_generation_time": 0.0, "min_time": 0.0, "max_time": 0.0}

            times = self.plan_generation_times
            return {
                "avg_generation_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "total_plans_generated": len(times),
                "cache_hit_rate": len([t for t in times if t < 0.1]) / len(times) if times else 0  # تخمینہ
            }
    ```

    ## علمی منصوبہ بندی کے لیے بہترین طریقے

    ### 1. غلطی کو سنبھالنا اور بازیابی (Error Handling and Recovery)

    ```python
    class RobustCognitivePlanner(OptimizedCognitivePlanner):
        def __init__(self, api_key: str = None, model: str = "gpt-4"):
            super().__init__(api_key, model)
            self.max_replan_attempts = 3
            self.recovery_strategies = [
                self._simplified_plan_recovery,
                self._ask_for_clarification,
                self._fallback_to_safe_behavior
            ]

        def execute_plan_with_recovery(self, plan: List[RobotAction]) -> bool:
            """بلٹ ان ریکوری میکانزم کے ساتھ منصوبہ پر عمل کریں"""
            for attempt in range(self.max_replan_attempts):
                try:
                    success = self._execute_plan_internal(plan)
                    if success:
                        return True
                except Exception as e:
                    self.get_logger().error(f"منصوبے پر عمل درآمد کی کوشش {attempt + 1} میں ناکام رہا: {e}")

                    if attempt < self.max_replan_attempts - 1:
                        # ریکوری کی حکمت عملی آزمائیں
                        plan = self._apply_recovery_strategy(plan, str(e))

            return False  # تمام کوششیں ناکام رہیں

        def _apply_recovery_strategy(self, failed_plan: List[RobotAction], error: str) -> List[RobotAction]:
            """غلطی کی قسم کی بنیاد پر ریکوری کی حکمت عملی لاگو کریں"""
            for strategy in self.recovery_strategies:
                new_plan = strategy(failed_plan, error)
                if new_plan:
                    return new_plan

            # اگر کوئی ریکوری کی حکمت عملی کام نہ کرے تو خالی منصوبہ واپس کریں
            return []

        def _simplified_plan_recovery(self, failed_plan: List[RobotAction], error: str) -> Optional[List[RobotAction]]:
            """منصوبے کا ایک سادہ ورژن بنانے کی کوشش کریں"""
            # نفاذ غلطی کی قسم پر منحصر ہوگا
            # مثال کے طور پر، اگر نیویگیشن ناکام ہو جائے تو، ایک سادہ راستہ آزمائیں
            pass

        def _ask_for_clarification(self, failed_plan: List[RobotAction], error: str) -> Optional[List[RobotAction]]:
            """وضاحت طلب کرنے کے لیے منصوبہ تیار کریں"""
            return [RobotAction(
                action_type=RobotActionType.INTERACTION,
                action_name="speak",
                parameters={"text": "مجھے ایک مسئلہ پیش آیا۔ کیا آپ براہ کرم اپنے کمانڈ کو واضح یا دوبارہ الفاظ میں بیان کر سکتے ہیں؟"},
                description="وضاحت کی درخواست"
            )]

        def _fallback_to_safe_behavior(self, failed_plan: List[RobotAction], error: str) -> Optional[List[RobotAction]]:
            """محفوظ رویے کے لیے منصوبہ تیار کریں"""
            return [RobotAction(
                action_type=RobotActionType.CONTROL,
                action_name="stop",
                parameters={},
                description="رکیں اور نئے کمانڈ کا انتظار کریں"
            )]
    ```

    ### 2. سیاق و سباق کا انتظام (Context Management)

    ```python
    class ContextManager:
        """علمی منصوبہ بندی کے لیے سیاق و سباق کا انتظام کریں"""

        def __init__(self):
            self.long_term_memory = {}  # سیشنز کے پار مستقل
            self.short_term_memory = {}  # موجودہ تعامل کے لیے
            self.context_windows = {}   # وقت پر مبنی سیاق و سباق کی ونڈوز

        def update_context(self, key: str, value: Any, duration: Optional[float] = None):
            """اختیاری مدت کے ساتھ سیاق و سباق کو اپ ڈیٹ کریں"""
            if duration:
                # میعاد ختم ہونے کے ساتھ اسٹور کریں
                self.context_windows[key] = {
                    "value": value,
                    "expires": time.time() + duration
                }
            else:
                # مستقل طور پر اسٹور کریں
                self.long_term_memory[key] = value

            # ہمیشہ شارٹ ٹرم میموری کو اپ ڈیٹ کریں
            self.short_term_memory[key] = value

        def get_context(self, key: str) -> Optional[Any]:
            """میعاد ختم ہونے کو ہینڈل کرتے ہوئے سیاق و سباق کی قدر حاصل کریں"""
            # پہلے سیاق و سباق کی ونڈوز کو چیک کریں (وہ دوسری یادوں کو اوور رائیڈ کرتی ہیں)
            if key in self.context_windows:
                window_data = self.context_windows[key]
                if time.time() < window_data["expires"]:
                    return window_data["value"]
                else:
                    # میعاد ختم شدہ سیاق و سباق کو ہٹائیں
                    del self.context_windows[key]

            # شارٹ ٹرم میموری کو چیک کریں
            if key in self.short_term_memory:
                return self.short_term_memory[key]

            # لانگ ٹرم میموری کو چیک کریں
            return self.long_term_memory.get(key)

        def clear_expired_context(self):
            """میعاد ختم شدہ سیاق و سباق کی ونڈوز کو ہٹائیں"""
            current_time = time.time()
            expired_keys = [
                key for key, data in self.context_windows.items()
                if current_time >= data["expires"]
            ]

            for key in expired_keys:
                del self.context_windows[key]
    ```

    ## عام مسائل کا حل (Troubleshooting)

    ### منصوبہ بندی کی ناکامیاں
    *   **مبہم کمانڈز**: وضاحت کی درخواستیں نافذ کریں۔
    *   **وسائل کی رکاوٹیں**: فزیبلٹی کی جانچ شامل کریں۔
    *   **حفاظتی خلاف ورزیاں**: حفاظتی رکاوٹیں اور توثیق نافذ کریں۔
    *   **کارکردگی کے مسائل**: کیشنگ اور آپٹیمائزیشن کی تکنیکیں استعمال کریں۔

    ### LLM انٹیگریشن کے مسائل
    *   **API کی حدود**: شرح کی حدوں اور ٹائم آؤٹس کو ہینڈل کریں۔
    *   **لاگت کا انتظام**: استعمال کی ٹریکنگ اور آپٹیمائزیشن نافذ کریں۔
    *   **جواب کی تغیر پذیری**: مستقل مزاجی کی جانچ اور توثیق شامل کریں۔

    ## ہینڈس آن مشق

    ایک مکمل علمی منصوبہ بندی کا نظام بنائیں جس میں شامل ہوں:

    1.  LLM پر مبنی قدرتی زبان کی سمجھ اور ٹاسک ڈیکمپوزیشن۔
    2.  دنیا کی حالت کے انتظام کے ساتھ سیاق و سباق سے آگاہ منصوبہ بندی۔
    3.  پیچیدہ ٹاسک ڈیکمپوزیشن کے لیے ہائیرارکیکل ٹاسک نیٹ ورک۔
    4.  حفاظتی توثیق اور فزیبلٹی کی جانچ۔
    5.  منصوبے پر عمل درآمد کے لیے ROS 2 انٹیگریشن۔
    6.  کیشنگ کے ساتھ کارکردگی کی اصلاح۔
    7.  غلطی کو سنبھالنے اور بازیابی کے میکانزم۔

    یہ مشق آپ کو ہیومنائیڈ روبوٹس کے لیے علمی منصوبہ بندی کے نظام کو نافذ کرنے اور قابل عمل روبوٹ ایکشنز میں قدرتی زبان کو ترجمہ کرنے میں شامل چیلنجوں کو سمجھنے کے ساتھ عملی تجربہ فراہم کرے گی۔
  </div>
</BilingualChapter>