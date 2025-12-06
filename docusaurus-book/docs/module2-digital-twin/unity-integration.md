---
id: unity-integration
title: "Unity Integration for High-Fidelity Rendering"
sidebar_position: 4
---

# Unity Integration for High-Fidelity Rendering

## Introduction

While Gazebo provides excellent physics simulation capabilities, Unity offers superior visual rendering, complex environment creation, and advanced human-robot interaction scenarios. Unity's real-time rendering engine, vast asset library, and powerful development tools make it ideal for creating high-fidelity digital twins of humanoid robots.

Unity integration with ROS 2 enables the creation of photorealistic simulation environments that can be used for training computer vision models, testing human-robot interaction scenarios, and creating immersive teleoperation interfaces.

## Unity Robotics Setup

### Installing Unity

For robotics applications, we recommend Unity 2021.3 LTS or later with the following components:

1. Unity Hub for version management
2. Unity Editor with Linux Build Support (if targeting Linux)
3. Visual Studio or Rider for scripting

### Unity Robotics Hub

The Unity Robotics Hub provides essential tools for robotics simulation:

1. **Unity Robot Toolkit**: Pre-built robotics components and examples
2. **ROS-TCP-Connector**: Communication bridge between Unity and ROS 2
3. **Synthetic Data Tools**: For generating training data for AI models

Install via Unity Package Manager:
- Open Window → Package Manager
- Add package from git URL: `https://github.com/Unity-Technologies/ROS-TCP-Connector.git`

### ROS 2 Integration Packages

Install the necessary ROS 2 packages for Unity communication:

```bash
# Install the ROS 2 Unity integration packages
sudo apt install ros-humble-rosbridge-suite
sudo apt install ros-humble-tf2-web-republisher
```

## ROS-TCP-Connector

### Architecture

The ROS-TCP-Connector creates a bridge between Unity and ROS 2 using TCP/IP communication:

```
Unity (Client) ←→ ROS TCP Endpoint ←→ ROS 2 Network
```

### Basic Setup

1. Create a new Unity project
2. Import the ROS-TCP-Connector package
3. Add the ROSConnection prefab to your scene
4. Configure the connection settings

### Unity Script Example

```csharp
using System.Collections;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;

public class RobotController : MonoBehaviour
{
    ROSConnection ros;
    string rosIP = "127.0.0.1";
    int rosPort = 10000;

    // Robot joint angles
    float[] jointPositions = new float[6];

    void Start()
    {
        // Get the ROS connection static instance
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIP, rosPort);

        // Start the coroutine to send commands
        StartCoroutine(SendRobotCommands());
    }

    IEnumerator SendRobotCommands()
    {
        // Send joint positions every 0.1 seconds
        while (true)
        {
            // Update joint positions (example values)
            for (int i = 0; i < jointPositions.Length; i++)
            {
                jointPositions[i] = Mathf.Sin(Time.time + i) * 0.5f;
            }

            // Create and send the message
            var jointState = new sensor_msgs.msg.JointState()
            {
                name = new string[] { "joint1", "joint2", "joint3", "joint4", "joint5", "joint6" },
                position = jointPositions,
                header = new std_msgs.msg.Header()
                {
                    stamp = new builtin_interfaces.msg.Time()
                    {
                        sec = (int)Time.time,
                        nanosec = (int)(Time.time % 1 * 1000000000)
                    }
                }
            };

            ros.Publish("/joint_states", jointState);

            yield return new WaitForSeconds(0.1f);
        }
    }

    void OnApplicationQuit()
    {
        ros.Dispose();
    }
}
```

## Creating Robot Models in Unity

### Importing Robot Models

Unity supports various 3D model formats (FBX, OBJ, DAE). For best results:

1. Ensure models have proper joint hierarchies
2. Use appropriate scale (1 Unity unit = 1 meter)
3. Include collision meshes for interaction

### Robot Prefab Structure

Create a well-structured robot prefab:

```
Robot (Root GameObject)
├── BaseLink
├── Joint1
│   ├── Link1
│   └── Joint2
│       ├── Link2
│       └── Joint3
│           └── Link3
├── Sensors
│   ├── Camera
│   ├── LiDAR
│   └── IMU
└── ROS Components
    ├── Joint Controllers
    └── Sensor Publishers
```

### Joint Control System

Implement a joint control system for humanoid robots:

```csharp
using UnityEngine;

public class JointController : MonoBehaviour
{
    [Header("Joint Configuration")]
    public string jointName;
    public JointType jointType = JointType.Revolute;
    public float minAngle = -90f;
    public float maxAngle = 90f;
    public float maxVelocity = 1.0f;

    [Header("Current State")]
    public float currentAngle = 0f;
    public float targetAngle = 0f;

    // Joint state publisher
    private ROSConnection ros;
    private ArticulationBody articulationBody;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        articulationBody = GetComponent<ArticulationBody>();

        if (articulationBody == null)
        {
            Debug.LogError("ArticulationBody component is required for joint control");
        }
    }

    void Update()
    {
        // Move toward target angle
        if (articulationBody != null)
        {
            var drive = articulationBody.xDrive;
            drive.target = targetAngle;
            articulationBody.xDrive = drive;
        }
    }

    public void SetTargetAngle(float angle)
    {
        targetAngle = Mathf.Clamp(angle, minAngle, maxAngle);
    }

    public float GetCurrentAngle()
    {
        if (articulationBody != null)
        {
            return articulationBody.jointPosition.x;
        }
        return currentAngle;
    }
}
```

## Sensor Simulation in Unity

### Camera Sensors

Unity's built-in cameras can simulate various types of visual sensors:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using UnityEngine.Rendering;

public class CameraSensor : MonoBehaviour
{
    public Camera cameraComponent;
    public string topicName = "/camera/image_raw";
    public int imageWidth = 640;
    public int imageHeight = 480;
    public float publishRate = 30.0f;

    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private float lastPublishTime = 0f;

    void Start()
    {
        // Create render texture for camera
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        cameraComponent.targetTexture = renderTexture;

        // Create texture for reading
        texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
    }

    void Update()
    {
        if (Time.time - lastPublishTime > 1.0f / publishRate)
        {
            PublishCameraImage();
            lastPublishTime = Time.time;
        }
    }

    void PublishCameraImage()
    {
        // Copy render texture to regular texture
        RenderTexture.active = renderTexture;
        texture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture2D.Apply();

        // Convert to ROS message format and publish
        // (Implementation depends on specific ROS message format)
    }
}
```

### LiDAR Simulation

Create a LiDAR sensor using Unity's raycasting:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class LidarSensor : MonoBehaviour
{
    [Header("Lidar Configuration")]
    public int horizontalRays = 360;
    public int verticalRays = 1;
    public float minAngle = -90f;
    public float maxAngle = 90f;
    public float maxRange = 10.0f;
    public string topicName = "/scan";

    private ROSConnection ros;
    private List<float> ranges;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ranges = new List<float>(new float[horizontalRays]);
    }

    void Update()
    {
        ScanEnvironment();
    }

    void ScanEnvironment()
    {
        for (int i = 0; i < horizontalRays; i++)
        {
            float angle = Mathf.Lerp(minAngle, maxAngle, (float)i / horizontalRays);
            Vector3 direction = Quaternion.Euler(0, angle, 0) * transform.forward;

            if (Physics.Raycast(transform.position, direction, out RaycastHit hit, maxRange))
            {
                ranges[i] = hit.distance;
            }
            else
            {
                ranges[i] = maxRange;
            }
        }

        PublishLidarData();
    }

    void PublishLidarData()
    {
        // Create and publish LaserScan message
        // Implementation depends on ROS message structure
    }
}
```

## High-Fidelity Rendering Features

### Physically-Based Rendering (PBR)

Unity's PBR materials provide realistic lighting and material properties:

```csharp
using UnityEngine;

public class PBRSurfaceMaterial : MonoBehaviour
{
    [Header("PBR Properties")]
    public float metallic = 0.0f;
    public float smoothness = 0.5f;
    public Color baseColor = Color.white;

    void Start()
    {
        Renderer renderer = GetComponent<Renderer>();
        if (renderer != null)
        {
            Material material = renderer.material;
            material.SetFloat("_Metallic", metallic);
            material.SetFloat("_Smoothness", smoothness);
            material.SetColor("_BaseColor", baseColor);
        }
    }
}
```

### Realistic Lighting

Set up realistic lighting for humanoid robot simulation:

```csharp
using UnityEngine;

public class RealisticLightingSetup : MonoBehaviour
{
    [Header("Lighting Configuration")]
    public Light mainLight;
    public float intensity = 1.0f;
    public Color color = Color.white;
    public float shadowStrength = 1.0f;

    void Start()
    {
        if (mainLight != null)
        {
            mainLight.intensity = intensity;
            mainLight.color = color;
            mainLight.shadowStrength = shadowStrength;
        }

        // Configure global lighting
        RenderSettings.ambientLight = new Color(0.2f, 0.2f, 0.2f);
        RenderSettings.fog = true;
        RenderSettings.fogColor = Color.gray;
        RenderSettings.fogDensity = 0.01f;
    }
}
```

## Human-Robot Interaction Scenarios

### Social Interaction Simulation

Create scenarios for testing human-robot social interaction:

```csharp
using UnityEngine;

public class SocialInteractionScenario : MonoBehaviour
{
    [Header("Interaction Setup")]
    public GameObject humanoidRobot;
    public GameObject humanAvatar;
    public float interactionDistance = 2.0f;
    public float personalSpaceRadius = 1.0f;

    [Header("Behavior Parameters")]
    public float attentionSpan = 5.0f;
    public float responseDelay = 0.5f;

    private float lastInteractionTime = 0f;
    private bool isInInteraction = false;

    void Update()
    {
        float distance = Vector3.Distance(
            humanoidRobot.transform.position,
            humanAvatar.transform.position
        );

        if (distance <= interactionDistance)
        {
            if (!isInInteraction)
            {
                StartInteraction();
            }
            HandleInteraction();
        }
        else if (isInInteraction)
        {
            EndInteraction();
        }
    }

    void StartInteraction()
    {
        isInInteraction = true;
        lastInteractionTime = Time.time;
        // Trigger greeting behavior
    }

    void HandleInteraction()
    {
        // Implement interaction logic
        // Track gaze, gestures, conversation flow
    }

    void EndInteraction()
    {
        isInInteraction = false;
        // Trigger farewell behavior
    }
}
```

### Manipulation Tasks

Set up manipulation scenarios with realistic physics:

```csharp
using UnityEngine;

public class ManipulationScenario : MonoBehaviour
{
    [Header("Manipulation Setup")]
    public GameObject robotArm;
    public GameObject targetObject;
    public Transform targetPosition;
    public float reachDistance = 1.0f;

    [Header("Task Parameters")]
    public bool isGrasping = false;
    public float graspForce = 100f;

    void Update()
    {
        float distance = Vector3.Distance(
            robotArm.transform.position,
            targetObject.transform.position
        );

        if (distance <= reachDistance && !isGrasping)
        {
            AttemptGrasp();
        }
    }

    void AttemptGrasp()
    {
        // Implement grasping logic
        // Use Unity's physics constraints for realistic grasping
        ConfigurableJoint joint = robotArm.GetComponent<ConfigurableJoint>();
        if (joint != null)
        {
            joint.connectedBody = targetObject.GetComponent<Rigidbody>();
            isGrasping = true;
        }
    }

    void ReleaseObject()
    {
        ConfigurableJoint joint = robotArm.GetComponent<ConfigurableJoint>();
        if (joint != null)
        {
            joint.connectedBody = null;
            isGrasping = false;
        }
    }
}
```

## Performance Optimization

### Level of Detail (LOD)

Implement LOD systems for complex humanoid robots:

```csharp
using UnityEngine;

[CreateAssetMenu(fileName = "LODSettings", menuName = "Robotics/LOD Settings")]
public class LODSettings : ScriptableObject
{
    [Header("LOD Configuration")]
    public float[] screenPercentages = { 1.0f, 0.5f, 0.25f };
    public GameObject[] lodMeshes;
    public int[] renderQueues;
}

public class RobotLODController : MonoBehaviour
{
    public LODSettings lodSettings;
    private LODGroup lodGroup;

    void Start()
    {
        SetupLOD();
    }

    void SetupLOD()
    {
        lodGroup = GetComponent<LODGroup>();
        if (lodGroup == null)
        {
            lodGroup = gameObject.AddComponent<LODGroup>();
        }

        LOD[] lods = new LOD[lodSettings.screenPercentages.Length];
        for (int i = 0; i < lodSettings.screenPercentages.Length; i++)
        {
            lods[i] = new LOD(
                lodSettings.screenPercentages[i],
                new Renderer[] { /* Assign appropriate renderers */ }
            );
        }

        lodGroup.SetLODs(lods);
    }
}
```

### Occlusion Culling

Enable occlusion culling for large environments:

```csharp
using UnityEngine;

public class OcclusionCullingSetup : MonoBehaviour
{
    void Start()
    {
        // Ensure occlusion culling is enabled in build settings
        // This is typically done in the Unity Editor:
        // Window -> Rendering -> Occlusion Culling
    }
}
```

## Integration with NVIDIA Isaac Sim

### Omniverse Connection

Unity can connect to NVIDIA Omniverse for enhanced simulation capabilities:

```csharp
using UnityEngine;

public class OmniverseConnector : MonoBehaviour
{
    [Header("Omniverse Configuration")]
    public string serverAddress = "localhost";
    public int serverPort = 8080;
    public string assetPath = "/Isaac/Robots/";

    void Start()
    {
        // Initialize Omniverse connection
        // This would typically use NVIDIA's Omniverse Unity connector
        ConnectToOmniverse();
    }

    void ConnectToOmniverse()
    {
        // Implementation depends on specific Omniverse connector
        Debug.Log($"Connecting to Omniverse at {serverAddress}:{serverPort}");
    }
}
```

## Best Practices for Unity Robotics

### 1. Performance Considerations

- Use Object Pooling for frequently created/destroyed objects
- Optimize mesh complexity for real-time performance
- Use efficient collision detection methods
- Implement proper culling for large environments

### 2. Scale and Units

- Maintain 1:1 scale (1 Unity unit = 1 meter)
- Verify all models are properly scaled before import
- Use consistent units across all components

### 3. Physics Stability

- Use appropriate physics settings for humanoid dynamics
- Test joint limits and constraints thoroughly
- Validate center of mass and inertial properties

### 4. Communication Efficiency

- Optimize message frequency to avoid network bottlenecks
- Use appropriate data compression for sensor data
- Implement proper error handling for network connections

## Troubleshooting Common Issues

### Connection Problems
- Verify ROS 2 bridge is running
- Check IP addresses and ports
- Ensure firewall allows connections

### Performance Issues
- Reduce polygon count of models
- Use occlusion culling for large scenes
- Optimize shader complexity

### Physics Instability
- Adjust physics time step
- Verify mass and inertia properties
- Check joint limits and constraints

## Hands-on Exercise

Create a Unity scene that includes:

1. A humanoid robot model with proper joint hierarchy
2. ROS 2 communication setup using ROS-TCP-Connector
3. Camera and LiDAR sensor simulation
4. A realistic environment with proper lighting
5. A simple interaction scenario (e.g., following a moving target)
6. Performance optimization techniques

This exercise will give you hands-on experience with Unity integration for high-fidelity humanoid robot simulation and human-robot interaction scenarios.