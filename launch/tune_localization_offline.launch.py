from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    bag_path = LaunchConfiguration('bag_path', default='/home/gabriel/TelloControl/offline_localization_bag')
    bag_rate = LaunchConfiguration('bag_rate', default='1.0')
    bag_loop = LaunchConfiguration('bag_loop', default='true')
    playback_delay_sec = LaunchConfiguration('playback_delay_sec', default='2.0')

    image_topic = LaunchConfiguration('image_topic', default='/tello/image_raw')
    camera_info_topic = LaunchConfiguration('camera_info_topic', default='/tello/camera_info')
    camera_frame = LaunchConfiguration('camera_frame', default='tello_camera')

    base_link_pose_topic = LaunchConfiguration('base_link_pose_topic', default='/base_link_pose')
    filtered_pose_topic = LaunchConfiguration('filtered_pose_topic', default='/filtered_pose')

    enable_kalman_filter = LaunchConfiguration('enable_kalman_filter', default='true')
    kalman_measurement_timeout_sec = LaunchConfiguration('kalman_measurement_timeout_sec', default='0.5')
    kalman_timer_period_sec = LaunchConfiguration('kalman_timer_period_sec', default='0.01')
    kalman_sigma_a = LaunchConfiguration('kalman_sigma_a', default='30.0')
    kalman_measurement_var = LaunchConfiguration('kalman_measurement_var', default='0.1')

    return LaunchDescription([
        DeclareLaunchArgument('bag_path', default_value='/home/gabriel/TelloControl/offline_localization_bag', description='Absolute path to the rosbag to replay.'),
        DeclareLaunchArgument('bag_rate', default_value='1.0', description='Playback rate for ros2 bag play.'),
        DeclareLaunchArgument('bag_loop', default_value='true', description='Loop rosbag playback.'),
        DeclareLaunchArgument('playback_delay_sec', default_value='2.0', description='Delay before starting rosbag playback so nodes are ready.'),
        DeclareLaunchArgument('image_topic', default_value='/tello/image_raw', description='Image topic recorded in the bag.'),
        DeclareLaunchArgument('camera_info_topic', default_value='/tello/camera_info', description='Camera info topic recorded in the bag.'),
        DeclareLaunchArgument('camera_frame', default_value='tello_camera', description='Camera frame used by the ArUco stack.'),
        DeclareLaunchArgument('base_link_pose_topic', default_value='/base_link_pose', description='Pose topic published by aruco_tf_node.'),
        DeclareLaunchArgument('filtered_pose_topic', default_value='/filtered_pose', description='Output topic from kalman_filter_node.'),
        DeclareLaunchArgument('enable_kalman_filter', default_value='true', description='Launch kalman_filter_node for offline tuning.'),
        DeclareLaunchArgument('kalman_measurement_timeout_sec', default_value='0.5', description='Measurement timeout used by kalman_filter_node.'),
        DeclareLaunchArgument('kalman_timer_period_sec', default_value='0.01', description='Prediction timer period used by kalman_filter_node.'),
        DeclareLaunchArgument('kalman_sigma_a', default_value='30.0', description='Process noise acceleration sigma used by kalman_filter_node.'),
        DeclareLaunchArgument('kalman_measurement_var', default_value='0.1', description='Measurement variance used by kalman_filter_node.'),
        Node(
            package='ros2_aruco',
            executable='aruco_node',
            name='aruco_node',
            output='screen',
            parameters=[
                {'marker_size': 0.10},
                {'image_topic': image_topic},
                {'camera_info_topic': camera_info_topic},
                {'camera_frame': camera_frame},
            ],
            remappings=[
                ('/camera/image_raw', image_topic),
                ('/camera/camera_info', camera_info_topic),
            ],
        ),
        Node(
            package='ros2_aruco',
            executable='aruco_tf_node',
            name='aruco_tf_node',
            output='screen',
            parameters=[
                {'reference_tag_id': 2},
                {'map_tag_position': [1.1, 0.0, 0.85]},
                {'map_tag_euler': [1.57, 0.0, 1.5708]},
                {'multi_tag_ids': [1, 3, 4, 5, 6, 7]},
                {'multi_tag_map_positions': [
                    1.1, -0.4, 1.15,
                    1.1,  0.4, 1.15,
                    1.1,  0.0, 0.45,
                    1.1,  0.0, 1.15,
                    1.1, -0.4, 0.85,
                    1.1,  0.4, 0.85,
                ]},
                {'multi_tag_map_eulers': [
                    1.57, 0.0, 1.5708,
                    1.57, 0.0, 1.5708,
                    1.57, 0.0, 1.5708,
                    1.57, 0.0, 1.5708,
                    1.57, 0.0, 1.5708,
                    1.57, 0.0, 1.5708,
                ]},
                {'camera_frame': camera_frame},
                {'camera_info_topic': camera_info_topic},
                {'marker_size': 0.10},
                {'map_frame': 'map'},
                {'markers_topic': 'aruco_markers'},
                {'publish_base_link_tf': True},
                {'base_link_frame': 'base_link'},
                {'base_to_camera_translation': [0.0, 0.0, 0.0]},
                {'base_to_camera_euler': [-1.5708, 0.0, 1.5708]},
                {'publish_base_link_pose': True},
                {'base_link_pose_topic': base_link_pose_topic},
            ],
        ),
        Node(
            package='localization_pkg',
            executable='kalman_filter_node',
            output='screen',
            condition=IfCondition(enable_kalman_filter),
            parameters=[{
                'measurement_timeout_sec': ParameterValue(kalman_measurement_timeout_sec, value_type=float),
                'timer_period_sec': ParameterValue(kalman_timer_period_sec, value_type=float),
                'sigma_a': ParameterValue(kalman_sigma_a, value_type=float),
                'measurement_var': ParameterValue(kalman_measurement_var, value_type=float),
            }],
            remappings=[
                ('/natnet_ros/Bebop1/pose', base_link_pose_topic),
                ('/filtered_pose', filtered_pose_topic),
            ],
        ),
        TimerAction(
            period=playback_delay_sec,
            actions=[
                ExecuteProcess(
                    cmd=['ros2', 'bag', 'play', bag_path, '--rate', bag_rate, '--loop'],
                    output='screen',
                    condition=IfCondition(bag_loop),
                ),
                ExecuteProcess(
                    cmd=['ros2', 'bag', 'play', bag_path, '--rate', bag_rate],
                    output='screen',
                    condition=UnlessCondition(bag_loop),
                ),
            ],
        ),
    ])
