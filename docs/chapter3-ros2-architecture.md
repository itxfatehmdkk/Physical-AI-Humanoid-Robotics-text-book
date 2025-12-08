---
sidebar_position: 3
title: "Chapter 3: ROS 2: The Robotic Nervous System"
---

# Chapter 3: ROS 2: The Robotic Nervous System

## ROS 2 Architecture

Robot Operating System 2 (ROS 2) represents a complete rewrite of the original ROS to address its limitations and meet the requirements for production robotics applications. The architecture of ROS 2 is fundamentally different from ROS 1, with a focus on reliability, scalability, and real-time performance.

### Client Library Architecture

ROS 2 uses a client library architecture that sits on top of the Data Distribution Service (DDS) middleware. This allows ROS 2 to leverage the robust communication features provided by DDS implementations such as:

- **Fast DDS (formerly Fast RTPS)**: Default DDS implementation
- **Cyclone DDS**: Open-source, high-performance DDS
- **OpenDDS**: Open-source DDS implementation
- **RTI Connext DDS**: Commercial high-performance DDS

This design provides several advantages:

1. **Communication Middleware Abstraction**: Allows switching between different DDS implementations without changing application code
2. **Cross-Language Support**: Multiple client libraries (C++, Python, etc.) can interoperate seamlessly
3. **Transport Flexibility**: Supports various communication patterns and protocols
4. **Quality of Service (QoS)**: Provides fine-grained control over communication behavior

### Key Architectural Components

#### Nodes
The fundamental unit of computation in ROS 2. Nodes encapsulate functionality and communicate with other nodes through topics, services, and actions.

#### DDS Layer
Handles communication between nodes, providing features like:
- Discovery and matching of publishers and subscribers
- Data transport and delivery
- Quality of Service policies
- Security features

#### Client Libraries
Provide language-specific interfaces to the underlying DDS system:
- **rclcpp**: C++ client library
- **rclpy**: Python client library
- **rcl**: Common C client library used by language-specific libraries

## Nodes, Topics, Services, Actions

### Nodes

Nodes are processes that perform computation. In ROS 2, nodes are organized in a peer-to-peer network where each node can communicate directly with any other node once the discovery process is complete.

#### Creating a Node
```cpp
#include "rclcpp/rclcpp.hpp"

class MinimalNode : public rclcpp::Node
{
public:
    MinimalNode() : Node("minimal_node")
    {
        RCLCPP_INFO(this->get_logger(), "Hello, World!");
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalNode>());
    rclcpp::shutdown();
    return 0;
}
```

### Topics and Publishers/Subscribers

Topics provide asynchronous, many-to-many communication using a publish-subscribe model.

#### Publisher Example
```cpp
auto publisher = this->create_publisher<std_msgs::msg::String>("topic", 10);
auto message = std_msgs::msg::String();
message.data = "Hello World";
RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
publisher->publish(message);
```

#### Subscriber Example
```cpp
auto subscription = this->create_subscription<std_msgs::msg::String>(
    "topic", 10,
    [this](const std_msgs::msg::String::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
    });
```

### Services

Services provide synchronous, request-reply communication for task-oriented interactions.

#### Service Implementation
```cpp
class AddThreeIntsService : public rclcpp::Node
{
public:
    AddThreeIntsService() : Node("add_three_ints_service")
    {
        service_ = this->create_service<example_interfaces::srv::AddThreeInts>(
            "add_three_ints",
            [this](const example_interfaces::srv::AddThreeInts::Request::SharedPtr request,
                   example_interfaces::srv::AddThreeInts::Response::SharedPtr response) {
                response->sum = request->a + request->b + request->c;
                RCLCPP_INFO(rclcpp::get_logger("rclcpp"),
                            "Incoming request\na: %ld, b: %ld, c: %ld",
                            request->a, request->b, request->c);
                RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "sending back response: [%ld]",
                            (long int)response->sum);
            });
    }

private:
    rclcpp::Service<example_interfaces::srv::AddThreeInts>::SharedPtr service_;
};
```

### Actions

Actions provide a framework for long-running tasks with feedback, goal preemption, and result reporting.

#### Action Client
```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):

    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(
            self, Fibonacci, 'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result: {0}'.format(result.sequence))
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info('Received feedback: {0}'.format(feedback.partial_sequence))
```

## Writing ROS 2 Packages in Python

ROS 2 packages organized in Python follow a specific structure and use modern build tools.

### Package Structure
```
my_robot_package/
├── CMakeLists.txt
├── package.xml
├── setup.py
├── setup.cfg
├── resource/
│   └── my_robot_package
├── my_robot_package/
│   ├── __init__.py
│   └── my_module.py
├── launch/
│   └── my_launch_file.py
├── config/
│   └── my_params.yaml
└── test/
    └── test_my_module.py
```

### Creating a Python Package

#### setup.py
```python
from setuptools import find_packages, setup

package_name = 'my_robot_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_node = my_robot_package.my_node:main',
        ],
    },
)
```

#### Simple ROS 2 Node in Python
```python
import rclpy
from rclpy.node import Node

from std_msgs.msg import String


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Launch Files & Parameter Configs

ROS 2 uses Python-based launch files for starting multiple nodes together and managing configurations.

### Launch Files

Launch files are written in Python and provide a flexible way to start multiple nodes with specific configurations:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true'),
        
        Node(
            package='turtlesim',
            executable='turtlesim_node',
            name='sim',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
        ),
        
        Node(
            package='turtlesim',
            executable='turtle_teleop_key',
            name='teleop',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            remappings=[('/turtle1/cmd_vel', '/cmd_vel')]
        ),
    ])
```

### Parameter Configuration

Parameters in ROS 2 can be configured through YAML files, launch files, or command line arguments.

#### Parameter File Example (config/my_params.yaml)
```yaml
my_robot_namespace:
  ros__parameters:
    use_sim_time: false
    update_rate: 10.0
    robot_radius: 0.3
    controller:
      max_linear_velocity: 1.0
      max_angular_velocity: 1.5
      linear_proportional_gain: 1.0
      angular_proportional_gain: 1.0
```

### Loading Parameters

Parameters can be loaded in code:

```cpp
// Declare parameters with default values
this->declare_parameter("update_rate", 10.0);
this->declare_parameter("robot_radius", 0.3);

// Get parameter values
double update_rate = this->get_parameter("update_rate").as_double();
double robot_radius = this->get_parameter("robot_radius").as_double();
```

## URDF Overview for Humanoids

Unified Robot Description Format (URDF) is the standard way to describe robots in ROS. For humanoid robots, URDF becomes more complex due to multiple degrees of freedom and articulated structures.

### URDF Structure

A URDF file describes a robot as a collection of links connected by joints, with additional information like inertial properties, visual representations, and collision properties.

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.2 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.2 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
  
  <!-- Head link -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  
  <!-- Joint connecting base and head -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>
  
</robot>
```

### Key URDF Concepts for Humanoids

1. **Kinematic Chains**: Humanoids typically have multiple kinematic chains (arms, legs) that need to be modeled
2. **Fixed and Moving Joints**: Different joint types (revolute, prismatic, continuous) for various movements
3. **Inertial Properties**: Accurate mass, center of mass, and inertia values are crucial for simulation
4. **Visual and Collision Models**: Separate representations for visualization and physics simulation
5. **Transmission Elements**: Define how actuators connect to joints

### Xacro for Complex URDF

For complex humanoid robots, Xacro (XML Macros) helps manage URDF complexity:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_xacro">

  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="robot_width" value="0.2" />
  
  <xacro:macro name="simple_arm" params="prefix parent *origin">
    <joint name="${prefix}_shoulder_joint" type="revolute">
      <xacro:insert_block name="origin"/>
      <parent link="${parent}"/>
      <child link="${prefix}_upper_arm"/>
      <axis xyz="0 1 0"/>
      <limit lower="${-2*M_PI}" upper="${2*M_PI}" effort="30" velocity="1.0"/>
    </joint>

    <link name="${prefix}_upper_arm">
      <visual>
        <geometry>
          <cylinder radius="0.05" length="0.3"/>
        </geometry>
        <origin xyz="0 0 ${0.3/2}" rpy="0 0 0"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.05" length="0.3"/>
        </geometry>
        <origin xyz="0 0 ${0.3/2}" rpy="0 0 0"/>
      </collision>
      <xacro:cylinder_inertial radius="0.05" length="0.3" mass="2.0">
        <origin xyz="0 0 ${0.3/2}" rpy="0 0 0"/>
      </xacro:cylinder_inertial>
    </link>
  </xacro:macro>

  <!-- Robot body -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.6"/>
      </geometry>
    </visual>
  </link>

  <!-- Use the macro to create both arms -->
  <xacro:simple_arm prefix="left" parent="base_link">
    <origin xyz="0.2 0 0" rpy="0 0 0"/>
  </xacro:simple_arm>

  <xacro:simple_arm prefix="right" parent="base_link">
    <origin xyz="-0.2 0 0" rpy="0 0 0"/>
  </xacro:simple_arm>

</robot>
```

ROS 2 serves as the nervous system for robotic applications, providing the essential infrastructure for communication, coordination, and control. For humanoid robotics, this foundation becomes even more critical as the complexity of the system increases with multiple degrees of freedom and sophisticated control requirements.

## Conclusion

This chapter has provided a comprehensive overview of ROS 2, from its architecture to practical implementation details. As the robotics nervous system, ROS 2 enables complex robotic systems to function as coherent entities. The next chapters will explore how these concepts are applied in simulation environments and specialized platforms.

## Additional Resources

For readers interested in exploring these concepts at a deeper level:

### Books and Publications
- ["Programming Robots with ROS" by Morgan Quigley, Brian Gerkey, and William Smart](https://www.oreilly.com/library/view/programming-robots-with/9781449323852/) - Comprehensive guide to ROS development
- ["Effective Robotics Programming with ROS" by Anil Mahtani, Luis Sánchez, and Enrique Fernandez](https://www.packtpub.com/product/effective-robotics-programming-with-ros-third-edition/9781788478957) - Practical ROS programming techniques
- ["Mastering ROS for Robotics Programming" by Joseph Lentin](https://www.packtpub.com/product/mastering-ros-for-robotics-programming-second-edition/9781788478216) - Advanced ROS concepts and techniques

### Research Papers
- ["ROS 2 Design Overview" (2018)](https://design.ros2.org/articles/overview.html) - Architecture and design principles of ROS 2
- ["Middleware for Real-Time Robot Control: Analysis of DDS and Its Applicability" (2021)](https://arxiv.org/abs/1811.04607) - Technical analysis of DDS middleware used in ROS 2
- ["ROS 2 vs. ROS 1: An Analysis of Differences and Performance" (2020)](https://ieeexplore.ieee.org/document/9347055) - Comparative analysis of ROS versions
- ["Quality of Service in ROS 2: A Survey" (2022)](https://arxiv.org/abs/2201.03684) - QoS policies and their applications
- ["Real-Time Performance Analysis of ROS 2" (2019)](https://ieeexplore.ieee.org/document/8794184) - Performance considerations for real-time systems

### Online Resources
- [ROS 2 Official Documentation](https://docs.ros.org/en/humble/) - Comprehensive official documentation
- [ROS 2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html) - Step-by-step learning materials
- [ROS 2 Navigation System](https://navigation.ros.org/) - Advanced navigation framework
- [ROS 2 Control Framework](https://control.ros.org/) - Real-time control framework
- [ROS Index](https://index.ros.org/) - Package repository and search
- [ROS Discourse](https://discourse.ros.org/) - Community discussion forum
- [ROS Answers](https://answers.ros.org/questions/) - Community support platform

### Technical Tutorials and Tools
- [Create Your First ROS 2 Package](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html) - Getting started with ROS 2 development
- [ROS 2 Launch Files Tutorial](https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Creating-Launch-Files.html) - Advanced launch system configuration
- [ROS 2 Parameter Management](https://docs.ros.org/en/humble/Tutorials/Parameters/Understanding-Parameters.html) - Parameter configuration and management
- [ROS 2 Quality of Service (QoS) Examples](https://docs.ros.org/en/humble/Tutorials/Quality-of-Service.html) - Practical QoS implementation
- [Real-Time Programming with ROS 2](https://docs.ros.org/en/humble/Tutorials/Real-Time-Programming.html) - Techniques for real-time applications
- [ROS 2 Security Features](https://docs.ros.org/en/humble/Releases/Release-Humble-Hawksbill.html#security) - Security implementation guidelines
- [DDS Implementation Comparison](https://docs.ros.org/en/humble/Concepts/About-Data-Distribution-Service.html) - Comparison of different DDS implementations