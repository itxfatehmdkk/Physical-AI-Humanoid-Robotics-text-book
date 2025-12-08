---
sidebar_position: 6
title: "Chapter 6: Vision-Language-Action (VLA)"
---

# Chapter 6: Vision-Language-Action (VLA)

## LLM-Based Planning

Vision-Language-Action (VLA) models represent a significant advancement in robotics, bridging the gap between high-level language understanding and low-level motor control. These models enable robots to interpret natural language commands and execute them as sequences of physical actions.

### Understanding VLA Models

VLA models are large neural networks that jointly process visual inputs, language instructions, and action outputs. Unlike traditional robotics approaches where perception, planning, and control modules are separate, VLA models learn end-to-end mappings from vision and language to actions.

### Key Characteristics of LLM-Based Planning

1. **Multimodal Integration**: VLA models process visual and linguistic inputs simultaneously
2. **Zero-Shot Generalization**: Ability to follow novel instructions without task-specific training
3. **Real-World Grounding**: Actions are grounded in real-world visual observations
4. **Hierarchical Reasoning**: Can decompose complex tasks into primitive actions

### Architecture of VLA Systems

A typical VLA system consists of several components:

1. **Vision Encoder**: Processes visual input (images, video) into high-dimensional embeddings
2. **Language Embedder**: Converts natural language instructions into vector representations
3. **Fusion Network**: Combines visual and language features
4. **Action Decoder**: Generates motor commands based on combined features
5. **Policy Network**: Maps combined features to action parameters

### Example VLA Architecture

```python
import torch
import torch.nn as nn
import torchvision.models as models

class VisionLanguageActionModel(nn.Module):
    def __init__(self, action_dim, hidden_dim=512):
        super().__init__()
        
        # Vision encoder (using ResNet as example)
        self.vision_encoder = models.resnet50(pretrained=True)
        self.vision_encoder.fc = nn.Identity()  # Remove final classification layer
        
        # Language encoder (simplified example)
        self.language_encoder = nn.LSTM(
            input_size=300,  # Word embedding dimension
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(2048 + hidden_dim, hidden_dim),  # ResNet out + LSTM out
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, image, language):
        # Process visual input
        visual_features = self.vision_encoder(image)
        
        # Process language input
        language_features, _ = self.language_encoder(language)
        # Take the last output of LSTM
        language_features = language_features[:, -1, :]
        
        # Fuse visual and language features
        combined_features = torch.cat([visual_features, language_features], dim=1)
        fused_features = self.fusion_network(combined_features)
        
        # Generate actions
        actions = self.action_decoder(fused_features)
        
        return actions

# Example usage
model = VisionLanguageActionModel(action_dim=7)  # 7-DOF robot arm example
```

### Training VLA Models

Training VLA models typically involves:

1. **Data Collection**: Large datasets of demonstrations with visual, language, and action components
2. **Pre-training**: Pre-train vision and language encoders on large datasets
3. **Fine-tuning**: Fine-tune the entire system on robotics-specific data
4. **Curriculum Learning**: Gradually increase task complexity

### Challenges in LLM-Based Planning

1. **Reality Gap**: Models trained in simulation struggle to transfer to the real world
2. **Scalability**: Large models require significant computational resources
3. **Safety**: Ensuring actions are safe in unstructured environments
4. **Interpretability**: Understanding why models make certain decisions

## OpenAI Whisper Speech-to-Action

Speech interfaces provide a natural way for humans to interact with robotic systems. OpenAI's Whisper model offers state-of-the-art speech recognition capabilities that can be integrated into robotics applications.

### Introduction to Whisper for Robotics

Whisper is a robust automatic speech recognition (ASR) system capable of recognizing speech in various languages and acoustic conditions. When combined with robotics systems, Whisper can transform spoken commands into executable actions.

### Key Features of Whisper for Robotics

1. **Multilingual Support**: Recognizes speech in 99 languages
2. **Robustness**: Handles various accents, background noise, and acoustic conditions
3. **Real-time Processing**: Can process speech with minimal latency
4. **Transcription Accuracy**: High accuracy even in challenging conditions

### Implementing Whisper in Robotics

```python
import whisper
import torch
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class SpeechToActionNode:
    def __init__(self):
        rospy.init_node('speech_to_action')
        
        # Load Whisper model (choose appropriate model size for your application)
        self.model = whisper.load_model("base.en")  # For English, adjust as needed
        
        # Publishers and subscribers
        self.cmd_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.speech_subscriber = rospy.Subscriber('audio_input', String, self.speech_callback)
        
        # Command vocabulary for navigation
        self.navigation_commands = {
            'forward': (1.0, 0.0, 0.0),  # Linear x, y, z
            'backward': (-1.0, 0.0, 0.0),
            'left': (0.0, 0.0, 0.5),    # Angular z
            'right': (0.0, 0.0, -0.5),
            'stop': (0.0, 0.0, 0.0)
        }
        
    def speech_callback(self, msg):
        # Process audio file path from message
        audio_path = msg.data
        
        # Transcribe the audio using Whisper
        result = self.model.transcribe(audio_path)
        text = result['text'].strip().lower()
        
        rospy.loginfo(f"Speech recognized: {text}")
        
        # Parse the command and execute action
        self.parse_and_execute_command(text)
        
    def parse_and_execute_command(self, text):
        # Simple keyword-based command parsing
        # In practice, this would be more sophisticated
        for keyword, action in self.navigation_commands.items():
            if keyword in text:
                cmd_msg = Twist()
                cmd_msg.linear.x = action[0]
                cmd_msg.linear.y = action[1]
                cmd_msg.linear.z = 0.0
                cmd_msg.angular.x = 0.0
                cmd_msg.angular.y = 0.0
                cmd_msg.angular.z = action[2]
                
                self.cmd_publisher.publish(cmd_msg)
                rospy.loginfo(f"Executed command: {keyword}")
                return
                
        rospy.logwarn(f"Unrecognized command: {text}")

def main():
    node = SpeechToActionNode()
    rospy.spin()

if __name__ == '__main__':
    main()
```

### Advanced Whisper Integration

For more sophisticated integration, Whisper can be combined with NLP models to understand complex instructions:

```python
import whisper
import openai
from transformers import pipeline

class AdvancedSpeechToAction:
    def __init__(self, openai_api_key):
        # Initialize Whisper
        self.speech_model = whisper.load_model("base")
        
        # Initialize NLP pipeline for instruction parsing
        self.instruction_parser = pipeline(
            "text-classification", 
            model="microsoft/DialoGPT-medium"
        )
        
        # Set up OpenAI API
        openai.api_key = openai_api_key
        
    def process_speech_command(self, audio_path):
        # Transcribe speech
        result = self.speech_model.transcribe(audio_path)
        text = result['text'].strip()
        
        # Use LLM to parse complex instructions
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a command parser for a robot. Parse the user's natural language command and return a structured action. Respond in JSON format with 'action_type' and 'parameters' fields."},
                {"role": "user", "content": text}
            ]
        )
        
        # Parse the response and convert to robot action
        parsed_action = self.parse_command(response.choices[0].message['content'])
        return parsed_action
        
    def parse_command(self, response_text):
        # Extract structured action from LLM response
        # Implementation would extract JSON from response_text
        import json
        try:
            action_data = json.loads(response_text)
            return action_data
        except:
            return {"action_type": "unknown", "parameters": {}}
```

### Challenges with Speech-to-Action

1. **Acoustic Conditions**: Performance varies with noise and environmental factors
2. **Real-Time Processing**: Need for low-latency processing in dynamic environments
3. **Ambient Noise**: Background sounds can affect recognition accuracy
4. **Dialects and Accents**: Recognition accuracy varies across different speakers

## Converting Natural Language → ROS 2 Action Pipeline

The conversion of natural language commands to executable ROS 2 actions involves multiple stages of processing and understanding.

### Natural Language Processing Pipeline

The conversion process typically involves several stages:

1. **Speech Recognition**: Convert speech to text (if using voice commands)
2. **Intent Classification**: Determine the overall intent of the command
3. **Entity Extraction**: Identify specific objects, locations, or parameters
4. **Action Mapping**: Map the processed command to ROS 2 actions
5. **Execution**: Execute the mapped actions in the ROS 2 system

### ROS 2 Action Servers and Clients

ROS 2 provides action servers and clients for handling long-running tasks:

```python
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import String
import json

class NaturalLanguageActionServer(Node):
    def __init__(self):
        super().__init__('natural_language_action_server')
        
        # Create action server for navigation tasks
        self._action_server = ActionServer(
            self,
            NavigateWithCommand,
            'navigate_with_command',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)
        
        # Initialize NLP components
        self.nlp_processor = NLPProcessor()  # Custom NLP class
        
        # Publishers for navigation
        self.nav_publisher = self.create_publisher(Pose, 'goal_pose', 10)

    def goal_callback(self, goal_request):
        """Accept or reject goal requests."""
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject cancel requests."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT
    
    def execute_callback(self, goal_handle):
        """Execute the goal."""
        self.get_logger().info('Executing goal...')
        
        # Process the natural language command
        command_text = goal_handle.request.command
        parsed_command = self.nlp_processor.parse_command(command_text)
        
        # Execute the parsed command
        result = self.execute_parsed_command(parsed_command)
        
        # Return result
        goal_handle.succeed()
        result = NavigateWithCommand.Result()
        result.success = True
        result.message = f"Executed command: {command_text}"
        return result

    def execute_parsed_command(self, parsed_command):
        """Execute a parsed command."""
        command_type = parsed_command.get('type')
        
        if command_type == 'navigation':
            # Extract goal position
            x = parsed_command.get('x', 0.0)
            y = parsed_command.get('y', 0.0)
            
            # Create and publish goal pose
            goal_pose = Pose()
            goal_pose.position.x = x
            goal_pose.position.y = y
            goal_pose.position.z = 0.0
            self.nav_publisher.publish(goal_pose)
            
        elif command_type == 'manipulation':
            # Handle manipulation commands
            object_name = parsed_command.get('object')
            action = parsed_command.get('action')
            # Implementation for manipulation would go here
            
        return True

class NLPProcessor:
    """Process natural language commands."""
    def __init__(self):
        # Initialize NLP models
        pass
    
    def parse_command(self, command_text):
        """Parse a natural language command."""
        # This is a simplified example
        # In practice, you would use more sophisticated NLP techniques
        
        command_text_lower = command_text.lower()
        
        if 'go to' in command_text_lower or 'navigate to' in command_text_lower:
            # Extract location (this is a simplified extraction)
            import re
            # Simple regex to extract coordinates
            match = re.search(r'(\d+\.?\d*)\s*,\s*(\d+\.?\d*)', command_text)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                return {
                    'type': 'navigation',
                    'x': x,
                    'y': y
                }
            else:
                # Handle named locations
                if 'kitchen' in command_text_lower:
                    return {
                        'type': 'navigation',
                        'x': 5.0,
                        'y': 3.0
                    }
                elif 'living room' in command_text_lower:
                    return {
                        'type': 'navigation',
                        'x': -2.0,
                        'y': 1.0
                    }
        
        elif 'pick up' in command_text_lower or 'grasp' in command_text_lower:
            # Extract object to manipulate
            import re
            # Extract object name
            match = re.search(r'(?:pick up|grasp|get)\s+(.+)', command_text)
            if match:
                object_name = match.group(1).strip()
                return {
                    'type': 'manipulation',
                    'action': 'pick_up',
                    'object': object_name
                }
        
        # Default case - unrecognized command
        return {
            'type': 'unknown'
        }

def main(args=None):
    rclpy.init(args=args)
    action_server = NaturalLanguageActionServer()
    rclpy.spin(action_server)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Integration with OpenAI Models

For more sophisticated natural language understanding, we can use OpenAI models:

```python
import openai
import json

class OpenAICommandProcessor:
    def __init__(self, api_key):
        openai.api_key = api_key
        
    def parse_command(self, command_text):
        """Use OpenAI to parse natural language commands."""
        prompt = f"""
        Parse the following natural language command for a robot and return the structured output in JSON format:
        
        Command: "{command_text}"
        
        Return a JSON object with the following structure:
        {{
          "action_type": "navigation | manipulation | interaction | etc.",
          "parameters": {{
            "x": float, // If navigation
            "y": float, // If navigation
            "object": "string", // If manipulation
            "action": "string"  // If manipulation
          }},
          "confidence": float // Between 0 and 1
        }}
        
        Be precise with the JSON format and only return the JSON object with no additional text.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1  # Low temperature for more deterministic output
        )
        
        try:
            # Extract JSON from response
            content = response.choices[0].message['content'].strip()
            
            # Sometimes the API might return the JSON wrapped in markdown code blocks
            if content.startswith('```'):
                content = content[content.find('{'):content.rfind('}')+1]
            
            parsed_command = json.loads(content)
            return parsed_command
        except Exception as e:
            print(f"Error parsing command: {e}")
            return {
                "action_type": "unknown",
                "parameters": {},
                "confidence": 0.0
            }
```

## Bipedal Locomotion & Manipulation Planning

Combining locomotion and manipulation planning in humanoid robots is one of the most challenging problems in robotics, requiring coordination between the robot's ability to move around and interact with objects in its environment.

### Bipedal Locomotion Challenges

Humanoid robots face several unique challenges for locomotion:

1. **Balance**: Maintaining stability on two legs
2. **Dynamic Walking**: Managing the physics of walking
3. **Terrain Adaptation**: Adapting to different surfaces and obstacles
4. **Energy Efficiency**: Minimizing power consumption

### Locomotion Planning Approaches

#### Zero-Moment Point (ZMP) Based Control

ZMP control ensures the robot maintains balance by keeping the center of pressure within the support polygon:

```python
import numpy as np

class ZMPBasedWalker:
    def __init__(self, robot_mass, gravity=9.81):
        self.mass = robot_mass
        self.gravity = gravity
        self.g = gravity
        self.omega = np.sqrt(self.g / self.mass)  # Simplified omega for ZMP
        
    def compute_zmp_reference(self, com_trajectory, com_vel_trajectory, com_acc_trajectory):
        """
        Compute ZMP reference from center of mass trajectory
        This is a simplified version - real implementations are more complex
        """
        zmp_ref_x = com_trajectory[:, 0] - (com_acc_trajectory[:, 0] / self.omega**2)
        zmp_ref_y = com_trajectory[:, 1] - (com_acc_trajectory[:, 1] / self.omega**2)
        
        return np.column_stack((zmp_ref_x, zmp_ref_y))

    def generate_foot_steps(self, zmp_reference, step_length=0.3, step_width=0.2):
        """
        Generate footstep plan based on ZMP reference
        """
        footsteps = []
        
        # Simplified footstep generation
        # In practice, this would implement more sophisticated planning
        for i, zmp_point in enumerate(zmp_reference):
            if i % 2 == 0:  # Left foot
                foot_pos = [zmp_point[0], step_width/2.0, 0.0]
            else:  # Right foot
                foot_pos = [zmp_point[0], -step_width/2.0, 0.0]
                
            footsteps.append(foot_pos)
            
        return footsteps
```

#### Model Predictive Control (MPC) for Walking

MPC approaches optimize walking patterns over a prediction horizon:

```python
class ModelPredictiveWalker:
    def __init__(self, prediction_horizon=20, dt=0.1):
        self.horizon = prediction_horizon
        self.dt = dt
        self.A = np.array([[1, self.dt, self.dt**2 / 2],
                           [0, 1, self.dt],
                           [0, 0, 1]])
        # Simplified dynamics matrix for CoM motion
        
    def compute_control_sequence(self, current_state, reference_trajectory):
        """
        Compute control sequence using simplified MPC approach
        """
        # This is a simplified implementation
        # Real MPC would solve an optimization problem
        control_sequence = np.zeros((self.horizon, 2))  # 2D control (x, y)
        
        for i in range(self.horizon):
            # Simplified tracking control
            error = reference_trajectory[i] - current_state[:2]
            control_sequence[i] = 2 * error  # Simplified control law
            
        return control_sequence
```

### Manipulation Planning

Manipulation planning involves planning robot arm motions to interact with objects:

```python
import numpy as np

class ManipulationPlanner:
    def __init__(self, robot):
        self.robot = robot
        self.workspace_bounds = {
            'x': (-0.5, 0.5),
            'y': (-0.3, 0.3),
            'z': (0.1, 0.8)
        }
        
    def plan_reach_motion(self, target_pose):
        """
        Plan arm motion to reach a target pose
        """
        # Simplified motion planning
        # In practice, this would use more sophisticated planners like RRT*
        current_pose = self.robot.get_end_effector_pose()
        
        # Linear interpolation in Cartesian space
        num_steps = 20
        trajectory = []
        
        for i in range(num_steps + 1):
            t = i / num_steps
            interpolated_pose = {
                'x': current_pose['x'] + t * (target_pose['x'] - current_pose['x']),
                'y': current_pose['y'] + t * (target_pose['y'] - current_pose['y']),
                'z': current_pose['z'] + t * (target_pose['z'] - current_pose['z'])
            }
            trajectory.append(interpolated_pose)
            
        return trajectory
    
    def plan_grasp(self, object_pose):
        """
        Plan approach, grasp, and lift motion for an object
        """
        # Approach position (slightly above the object)
        approach_pose = object_pose.copy()
        approach_pose['z'] += 0.15  # 15cm above object
        
        # Grasp position (at object height)
        grasp_pose = object_pose.copy()
        
        # Lift position (above object)
        lift_pose = object_pose.copy()
        lift_pose['z'] += 0.2  # Lift 20cm
        
        # Generate the sequence of motions
        approach_traj = self.plan_reach_motion(approach_pose)
        grasp_traj = self.plan_reach_motion(grasp_pose)
        lift_traj = self.plan_reach_motion(lift_pose)
        
        return approach_traj + grasp_traj + lift_traj
```

### Coordinated Locomotion-Manipulation Planning

For humanoid robots, coordinating walking and manipulation is essential:

```python
class CoordinatedPlanner:
    def __init__(self, locomotion_planner, manipulation_planner):
        self.loco_planner = locomotion_planner
        self.mani_planner = manipulation_planner
        
    def plan_task(self, goal_location, object_to_manipulate):
        """
        Plan coordinated task involving both locomotion and manipulation
        """
        # 1. Plan path to approach the object
        navigation_plan = self.loco_planner.plan_to_object(
            object_to_manipulate['location']
        )
        
        # 2. Plan manipulation at the location
        manipulation_plan = self.mani_planner.plan_grasp(object_to_manipulate['pose'])
        
        # 3. Coordinate the plans to ensure balance during manipulation
        coordinated_plan = self.coordinate_loco_mani(navigation_plan, manipulation_plan)
        
        return coordinated_plan
        
    def coordinate_loco_mani(self, navigation_plan, manipulation_plan):
        """
        Coordinate locomotion and manipulation to maintain stability
        """
        # Simplified coordination strategy:
        # - Stop locomotion during manipulation
        # - Use upper body for manipulation while maintaining base stability
        
        # In practice, this would implement more sophisticated coordination
        # such as dynamic balancing or anticipatory postural adjustments
        
        return {
            'navigation_plan': navigation_plan,
            'manipulation_plan': manipulation_plan,
            'timing': self.calculate_timing(navigation_plan, manipulation_plan)
        }
        
    def calculate_timing(self, nav_plan, mani_plan):
        """
        Calculate timing to coordinate locomotion and manipulation
        """
        # Simplified timing calculation
        return {
            'start_manipulation_at_step': len(nav_plan) // 2,
            'manipulation_duration': len(mani_plan) * 0.1  # 0.1s per step
        }
```

Vision-Language-Action (VLA) models represent the cutting edge of robotics research, enabling robots to understand and respond to natural language commands through coordinated physical actions. By combining the power of large language models with perception and control systems, VLA systems are bringing us closer to truly general-purpose robots that can interact naturally with humans in complex environments.

## Conclusion

This chapter has explored the critical VLA systems that bridge language understanding with physical action in robotics. From LLM-based planning to speech-to-action pipelines and coordinated locomotion-manipulation planning, these technologies are essential for creating robots that can effectively interact with the physical world through natural language commands. The next chapter will integrate all these concepts in a capstone project to build an autonomous humanoid system.

## Additional Resources

For readers interested in exploring these concepts at a deeper level:

### Books and Publications
- ["Language and Robots" by Antonio Lieto](https://www.tandf.net/books/9781032058528) - Intersections between linguistics and robotics
- ["Robot Learning from Human Teachers" by Sonia Chernova and Andrea Thomaz](https://www.morganclaypool.com/doi/abs/10.2200/S00629ED1V01Y201501AIM029) - Human-in-the-loop learning approaches
- ["Deep Learning for Natural Language Processing" by Palash Goyal](https://www.apress.com/gp/book/9781484243534) - Foundation for language understanding systems
- ["Handbook of Spatial Logics" by Marco Aiello](https://link.springer.com/book/10.1007/978-1-4020-5587-4) - Spatial reasoning for robotics

### Research Papers
- ["Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents" by Huang et al. (2022)](https://arxiv.org/abs/2201.07207) - Using LLMs for robotic planning
- ["Inner Monologue: Embodied Reasoning through Planning in Language Models" by Brohan et al. (2022)](https://arxiv.org/abs/2207.05608) - Language-based reasoning for embodied tasks
- ["PaLM-E: An Embodied Multimodal Language Model" by Driess et al. (2023)](https://arxiv.org/abs/2303.03378) - Large-scale vision-language-action model
- ["Grounded Decoding for Language-Guided Multi-Object Rearrangement" by Chen et al. (2022)](https://arxiv.org/abs/2205.14818) - Language-guided manipulation
- ["Language-Conditioned Imitation Learning for Robot Manipulation Tasks" by Shridhar et al. (2022)](https://arxiv.org/abs/2203.06844) - Learning manipulation from language demonstrations
- ["Embodied Visual Active Learning for Semantic Segmentation" by Zhang et al. (2021)](https://arxiv.org/abs/2105.08444) - Active learning for perception
- ["RT-1: Robotics Transformer for Real-World Control at Scale" by Brohan et al. (2022)](https://arxiv.org/abs/2212.06817) - Scalable robotics transformer model
- ["OpenVLA: An Open-Source Vision-Language-Action Model" by Bahl et al. (2024)](https://openvla.github.io/) - Open-source VLA model

### Online Resources
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference) - Interface for language models in robotics
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) - Pre-trained models for NLP tasks
- [Robot Learning with Language Models](https://sites.google.com/view/robotics-transformers) - Research hub for VLA systems
- [Embodied AI Challenge](https://aihabitat.org/challenge/) - Annual challenge focused on embodied intelligence
- [RoboTurk: Human Activities in Robot Environments](https://roboturk.stanford.edu/) - Dataset for language-guided manipulation
- [ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks](https://askforalfred.com/) - Task-oriented dataset with language instructions
- [SayCan: Do As I Can, Not As I Say](https://say-can.github.io/) - Language-guided robot execution

### Technical Tutorials and Tools
- [OpenAI Whisper Documentation](https://github.com/openai/whisper) - Speech recognition for robotics applications
- [Speech-to-Action Pipeline Implementation](https://github.com/facebookresearch/SEARLE) - Example implementation of speech-driven robot control
- [Vision-Language-Action Model Training Guide](https://huggingface.co/docs/transformers/model_doc/vision_encoder_decoder) - Training VLA models
- [ROS 2 Natural Language Processing Integration](https://github.com/ros-industrial/ros2_nlp) - NLP tools for ROS 2
- [Language-Guided Reinforcement Learning Tutorials](https://github.com/google-research/language_to_reward) - LfR (Language to Reward) implementations
- [Embodied-AI Simulation Environments](https://github.com/facebookresearch/habitat-lab) - Habitat for embodied AI research
- [Prompt Engineering for Robotics](https://www.promptingguide.ai/) - Techniques for designing effective prompts for robotic systems