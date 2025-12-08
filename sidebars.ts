import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Manual sidebar configuration with ordered chapters and unordered sub-topics
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Physical AI & Humanoid Robotics',
      items: [
        'index',
      ],
    },
    {
      type: 'category',
      label: 'Chapter 1: Introduction to Physical AI',
      link: {type: 'doc', id: 'chapter1-introduction-to-physical-ai'},
      items: [
        {
          type: 'link',
          label: 'Foundations of Physical AI',
          href: '/docs/chapter1-introduction-to-physical-ai#foundations-of-physical-ai',
        },
        {
          type: 'link',
          label: 'Historical Context of Physical AI',
          href: '/docs/chapter1-introduction-to-physical-ai#historical-context-of-physical-ai',
        },
        {
          type: 'link',
          label: 'Theoretical Framework',
          href: '/docs/chapter1-introduction-to-physical-ai#theoretical-framework',
        },
        {
          type: 'link',
          label: 'Digital AI → Embodied Intelligence Transition',
          href: '/docs/chapter1-introduction-to-physical-ai#digital-ai--embodied-intelligence-transition',
        },
        {
          type: 'link',
          label: 'Technical Challenges of the Transition',
          href: '/docs/chapter1-introduction-to-physical-ai#technical-challenges-of-the-transition',
        },
        {
          type: 'link',
          label: 'Research Approaches to Physical AI',
          href: '/docs/chapter1-introduction-to-physical-ai#research-approaches-to-physical-ai',
        },
        {
          type: 'link',
          label: 'Importance of Humanoid Robots',
          href: '/docs/chapter1-introduction-to-physical-ai#importance-of-humanoid-robots',
        },
        {
          type: 'link',
          label: 'Why Embodiment Matters',
          href: '/docs/chapter1-introduction-to-physical-ai#why-embodiment-matters',
        },
        {
          type: 'link',
          label: 'Physical Principles in Embodied Intelligence',
          href: '/docs/chapter1-introduction-to-physical-ai#physical-principles-in-embodied-intelligence',
        },
        {
          type: 'link',
          label: 'Physical AI Applications and Use Cases',
          href: '/docs/chapter1-introduction-to-physical-ai#physical-ai-applications-and-use-cases',
        },
        {
          type: 'link',
          label: 'Current Research Frontiers',
          href: '/docs/chapter1-introduction-to-physical-ai#current-research-frontiers',
        },
        {
          type: 'link',
          label: 'Advanced Topics and Further Reading',
          href: '/docs/chapter1-introduction-to-physical-ai#advanced-topics-and-further-reading',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 2: Robotics Fundamentals',
      link: {type: 'doc', id: 'chapter2-robotics-fundamentals'},
      items: [
        {
          type: 'link',
          label: 'Sensors: LiDAR, IMUs, Cameras',
          href: '/docs/chapter2-robotics-fundamentals#sensors-lidar-imus-cameras',
        },
        {
          type: 'link',
          label: 'LiDAR (Light Detection and Ranging)',
          href: '/docs/chapter2-robotics-fundamentals#lidar-light-detection-and-ranging',
        },
        {
          type: 'link',
          label: 'Types of LiDAR Systems',
          href: '/docs/chapter2-robotics-fundamentals#types-of-lidar-systems',
        },
        {
          type: 'link',
          label: 'LiDAR Data Processing',
          href: '/docs/chapter2-robotics-fundamentals#lidar-data-processing',
        },
        {
          type: 'link',
          label: 'IMUs (Inertial Measurement Units)',
          href: '/docs/chapter2-robotics-fundamentals#imus-inertial-measurement-units',
        },
        {
          type: 'link',
          label: 'Types of IMU Technologies',
          href: '/docs/chapter2-robotics-fundamentals#types-of-imu-technologies',
        },
        {
          type: 'link',
          label: 'IMU Calibration and Error Sources',
          href: '/docs/chapter2-robotics-fundamentals#imu-calibration-and-error-sources',
        },
        {
          type: 'link',
          label: 'Cameras',
          href: '/docs/chapter2-robotics-fundamentals#cameras',
        },
        {
          type: 'link',
          label: 'Technical Specifications of Camera Systems',
          href: '/docs/chapter2-robotics-fundamentals#technical-specifications-of-camera-systems',
        },
        {
          type: 'link',
          label: 'Image Processing Pipeline',
          href: '/docs/chapter2-robotics-fundamentals#image-processing-pipeline',
        },
        {
          type: 'link',
          label: 'Camera Calibration',
          href: '/docs/chapter2-robotics-fundamentals#camera-calibration',
        },
        {
          type: 'link',
          label: 'Actuators and Motor Control',
          href: '/docs/chapter2-robotics-fundamentals#actuators-and-motor-control',
        },
        {
          type: 'link',
          label: 'Types of Actuators',
          href: '/docs/chapter2-robotics-fundamentals#types-of-actuators',
        },
        {
          type: 'link',
          label: 'Motor Control Fundamentals',
          href: '/docs/chapter2-robotics-fundamentals#motor-control-fundamentals',
        },
        {
          type: 'link',
          label: 'Advanced Control Techniques',
          href: '/docs/chapter2-robotics-fundamentals#advanced-control-techniques',
        },
        {
          type: 'link',
          label: 'Control System Design Considerations',
          href: '/docs/chapter2-robotics-fundamentals#control-system-design-considerations',
        },
        {
          type: 'link',
          label: 'Kinematics Basics',
          href: '/docs/chapter2-robotics-fundamentals#kinematics-basics',
        },
        {
          type: 'link',
          label: 'Forward Kinematics',
          href: '/docs/chapter2-robotics-fundamentals#forward-kinematics',
        },
        {
          type: 'link',
          label: 'Mathematical Representation',
          href: '/docs/chapter2-robotics-fundamentals#mathematical-representation',
        },
        {
          type: 'link',
          label: 'Denavit-Hartenberg (DH) Convention',
          href: '/docs/chapter2-robotics-fundamentals#denavit-hartenberg-dh-convention',
        },
        {
          type: 'link',
          label: 'Inverse Kinematics',
          href: '/docs/chapter2-robotics-fundamentals#inverse-kinematics',
        },
        {
          type: 'link',
          label: 'Degrees of Freedom (DOF)',
          href: '/docs/chapter2-robotics-fundamentals#degrees-of-freedom-dof',
        },
        {
          type: 'link',
          label: 'Jacobian Matrix',
          href: '/docs/chapter2-robotics-fundamentals#jacobian-matrix',
        },
        {
          type: 'link',
          label: 'Overview of Robotics Landscape',
          href: '/docs/chapter2-robotics-fundamentals#overview-of-robotics-landscape',
        },
        {
          type: 'link',
          label: 'Advanced Control Concepts for Robotics',
          href: '/docs/chapter2-robotics-fundamentals#advanced-control-concepts-for-robotics',
        },
        {
          type: 'link',
          label: 'Motion Planning and Trajectory Generation',
          href: '/docs/chapter2-robotics-fundamentals#motion-planning-and-trajectory-generation',
        },
        {
          type: 'link',
          label: 'Sensor Integration and Fusion',
          href: '/docs/chapter2-robotics-fundamentals#sensor-integration-and-fusion',
        },
        {
          type: 'link',
          label: 'Advanced Topics and Further Reading',
          href: '/docs/chapter2-robotics-fundamentals#advanced-topics-and-further-reading',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 3: ROS 2: The Robotic Nervous System',
      link: {type: 'doc', id: 'chapter3-ros2-architecture'},
      items: [
        {
          type: 'link',
          label: 'ROS 2 Architecture Overview',
          href: '/docs/chapter3-ros2-architecture#ros-2-architecture-overview',
        },
        {
          type: 'link',
          label: 'Nodes, Topics, Services, Actions',
          href: '/docs/chapter3-ros2-architecture#nodes-topics-services-actions',
        },
        {
          type: 'link',
          label: 'Writing ROS 2 Packages in Python',
          href: '/docs/chapter3-ros2-architecture#writing-ros-2-packages-in-python',
        },
        {
          type: 'link',
          label: 'Launch Files & Parameter Configs',
          href: '/docs/chapter3-ros2-architecture#launch-files--parameter-configs',
        },
        {
          type: 'link',
          label: 'URDF Overview for Humanoids',
          href: '/docs/chapter3-ros2-architecture#urdf-overview-for-humanoids',
        },
        {
          type: 'link',
          label: 'Advanced Topics and Further Reading',
          href: '/docs/chapter3-ros2-architecture#advanced-topics-and-further-reading',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 4: Simulation with Gazebo and Unity',
      link: {type: 'doc', id: 'chapter4-simulation'},
      items: [
        {
          type: 'link',
          label: 'Gazebo Physics Environment',
          href: '/docs/chapter4-simulation#gazebo-physics-environment',
        },
        {
          type: 'link',
          label: 'SDF/URDF Robot Modeling',
          href: '/docs/chapter4-simulation#sdfurdf-robot-modeling',
        },
        {
          type: 'link',
          label: 'Simulated Sensors',
          href: '/docs/chapter4-simulation#simulated-sensors',
        },
        {
          type: 'link',
          label: 'Unity for High-Fidelity Visualization',
          href: '/docs/chapter4-simulation#unity-for-high-fidelity-visualization',
        },
        {
          type: 'link',
          label: 'Creating a Digital Twin Workflow',
          href: '/docs/chapter4-simulation#creating-a-digital-twin-workflow',
        },
        {
          type: 'link',
          label: 'Advanced Topics and Further Reading',
          href: '/docs/chapter4-simulation#advanced-topics-and-further-reading',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 5: NVIDIA Isaac Platform',
      link: {type: 'doc', id: 'chapter5-nvidia-isaac'},
      items: [
        {
          type: 'link',
          label: 'Isaac Sim',
          href: '/docs/chapter5-nvidia-isaac#isaac-sim',
        },
        {
          type: 'link',
          label: 'Isaac SDK & Accelerated ROS',
          href: '/docs/chapter5-nvidia-isaac#isaac-sdk--accelerated-ros',
        },
        {
          type: 'link',
          label: 'VSLAM and Navigation',
          href: '/docs/chapter5-nvidia-isaac#vslam-and-navigation',
        },
        {
          type: 'link',
          label: 'RL-Based Training',
          href: '/docs/chapter5-nvidia-isaac#rl-based-training',
        },
        {
          type: 'link',
          label: 'Sim-to-Real Techniques',
          href: '/docs/chapter5-nvidia-isaac#sim-to-real-techniques',
        },
        {
          type: 'link',
          label: 'Advanced Topics and Further Reading',
          href: '/docs/chapter5-nvidia-isaac#advanced-topics-and-further-reading',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 6: Vision-Language-Action (VLA)',
      link: {type: 'doc', id: 'chapter6-vla'},
      items: [
        {
          type: 'link',
          label: 'LLM-Based Planning',
          href: '/docs/chapter6-vla#llm-based-planning',
        },
        {
          type: 'link',
          label: 'OpenAI Whisper Speech-to-Action',
          href: '/docs/chapter6-vla#openai-whisper-speech-to-action',
        },
        {
          type: 'link',
          label: 'Converting Natural Language → ROS 2 Action Pipeline',
          href: '/docs/chapter6-vla#converting-natural-language--ros-2-action-pipeline',
        },
        {
          type: 'link',
          label: 'Bipedal Locomotion & Manipulation Planning',
          href: '/docs/chapter6-vla#bipedal-locomotion--manipulation-planning',
        },
        {
          type: 'link',
          label: 'Advanced Topics and Further Reading',
          href: '/docs/chapter6-vla#advanced-topics-and-further-reading',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 7: Capstone: Build an Autonomous Humanoid',
      link: {type: 'doc', id: 'chapter7-capstone'},
      items: [
        {
          type: 'link',
          label: 'Voice → Plan → Movement Pipeline',
          href: '/docs/chapter7-capstone#voice--plan--movement-pipeline',
        },
        {
          type: 'link',
          label: 'Object Identification & Manipulation',
          href: '/docs/chapter7-capstone#object-identification--manipulation',
        },
        {
          type: 'link',
          label: 'Full Pipeline with ROS 2 + Isaac + LLM',
          href: '/docs/chapter7-capstone#full-pipeline-with-ros-2--isaac--llm',
        },
        {
          type: 'link',
          label: 'Deployment Blueprint',
          href: '/docs/chapter7-capstone#deployment-blueprint',
        },
        {
          type: 'link',
          label: 'Advanced Topics and Further Reading',
          href: '/docs/chapter7-capstone#advanced-topics-and-further-reading',
        },
      ],
    },
  ],
};

export default sidebars;
