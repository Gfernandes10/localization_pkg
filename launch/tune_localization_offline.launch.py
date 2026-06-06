from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _bag_play_action(context):
    bag_path = LaunchConfiguration('bag_path').perform(context)
    bag_rate = LaunchConfiguration('bag_rate').perform(context)
    bag_loop = LaunchConfiguration('bag_loop').perform(context).lower()
    bag_topics = LaunchConfiguration('bag_topics').perform(context).strip()

    cmd = ['ros2', 'bag', 'play', bag_path, '--rate', bag_rate]
    if bag_loop in ('true', '1', 'yes', 'on'):
        cmd.append('--loop')

    if bag_topics and bag_topics.lower() != 'all':
        cmd.append('--topics')
        cmd.extend(bag_topics.split())

    return [ExecuteProcess(cmd=cmd, output='screen')]


def generate_launch_description():
    bag_path = LaunchConfiguration('bag_path', default='/home/gabriel/TelloControl/offline_localization_bag')
    bag_rate = LaunchConfiguration('bag_rate', default='1.0')
    bag_loop = LaunchConfiguration('bag_loop', default='true')
    bag_topics = LaunchConfiguration('bag_topics', default='/tello/image_raw /tello/camera_info')
    playback_delay_sec = LaunchConfiguration('playback_delay_sec', default='2.0')

    image_topic = LaunchConfiguration('image_topic', default='/tello/image_raw')
    camera_info_topic = LaunchConfiguration('camera_info_topic', default='/tello/camera_info')
    camera_frame = LaunchConfiguration('camera_frame', default='tello_camera')

    base_link_pose_topic = LaunchConfiguration('base_link_pose_topic', default='/base_link_pose')
    filtered_pose_topic = LaunchConfiguration('filtered_pose_topic', default='/filtered_pose')
    command_topic = LaunchConfiguration('command_topic', default='/tello/cmd_vel')

    enable_kalman_filter = LaunchConfiguration('enable_kalman_filter', default='true')
    kalman_measurement_timeout_sec = LaunchConfiguration('kalman_measurement_timeout_sec', default='0.5')
    kalman_timer_period_sec = LaunchConfiguration('kalman_timer_period_sec', default='0.01')
    kalman_sigma_a = LaunchConfiguration('kalman_sigma_a', default='30.0')
    kalman_measurement_var = LaunchConfiguration('kalman_measurement_var', default='0.1')
    kalman_publish_prediction_without_measurement = LaunchConfiguration('kalman_publish_prediction_without_measurement', default='true')
    kalman_max_prediction_duration_sec = LaunchConfiguration('kalman_max_prediction_duration_sec', default='2.0')
    kalman_use_control_input_model = LaunchConfiguration('kalman_use_control_input_model', default='false')
    kalman_control_input_timeout_sec = LaunchConfiguration('kalman_control_input_timeout_sec', default='0.25')
    kalman_control_input_delay_sec = LaunchConfiguration('kalman_control_input_delay_sec', default='0.0')
    kalman_control_model_max_step_sec = LaunchConfiguration('kalman_control_model_max_step_sec', default='0.02')
    kalman_control_model_gamma1 = LaunchConfiguration('kalman_control_model_gamma1', default='-1.22379710012708')
    kalman_control_model_gamma2 = LaunchConfiguration('kalman_control_model_gamma2', default='-0.5496637058581929')
    kalman_control_model_gamma3 = LaunchConfiguration('kalman_control_model_gamma3', default='1.230339924965825')
    kalman_control_model_gamma4 = LaunchConfiguration('kalman_control_model_gamma4', default='-0.5775755765146152')
    kalman_control_model_gamma5 = LaunchConfiguration('kalman_control_model_gamma5', default='0.6837187318618236')
    kalman_control_model_gamma6 = LaunchConfiguration('kalman_control_model_gamma6', default='-1.0763629574178275')
    kalman_control_model_gamma7 = LaunchConfiguration('kalman_control_model_gamma7', default='-2.033762293431863')
    kalman_control_model_gamma8 = LaunchConfiguration('kalman_control_model_gamma8', default='-2.036870107158528')
    aruco_reject_low_tag_count = LaunchConfiguration('aruco_reject_low_tag_count', default='true')
    aruco_min_pose_tag_count = LaunchConfiguration('aruco_min_pose_tag_count', default='3')
    aruco_reject_degenerate_tag_geometry = LaunchConfiguration('aruco_reject_degenerate_tag_geometry', default='true')
    aruco_min_tag_geometry_area = LaunchConfiguration('aruco_min_tag_geometry_area', default='0.03')
    aruco_min_tag_span_y = LaunchConfiguration('aruco_min_tag_span_y', default='0.20')
    aruco_min_tag_span_z = LaunchConfiguration('aruco_min_tag_span_z', default='0.20')
    aruco_use_pnp_extrinsic_guess = LaunchConfiguration('aruco_use_pnp_extrinsic_guess', default='true')
    aruco_pnp_guess_timeout_sec = LaunchConfiguration('aruco_pnp_guess_timeout_sec', default='0.50')

    return LaunchDescription([
        DeclareLaunchArgument('bag_path', default_value='/home/gabriel/TelloControl/offline_localization_bag', description='Absolute path to the rosbag to replay.'),
        DeclareLaunchArgument('bag_rate', default_value='1.0', description='Playback rate for ros2 bag play.'),
        DeclareLaunchArgument('bag_loop', default_value='true', description='Loop rosbag playback.'),
        DeclareLaunchArgument('bag_topics', default_value='/tello/image_raw /tello/camera_info', description='Space-separated list of topics to replay from the bag. Use "all" to replay every topic.'),
        DeclareLaunchArgument('playback_delay_sec', default_value='2.0', description='Delay before starting rosbag playback so nodes are ready.'),
        DeclareLaunchArgument('image_topic', default_value='/tello/image_raw', description='Image topic recorded in the bag.'),
        DeclareLaunchArgument('camera_info_topic', default_value='/tello/camera_info', description='Camera info topic recorded in the bag.'),
        DeclareLaunchArgument('camera_frame', default_value='tello_camera', description='Camera frame used by the ArUco stack.'),
        DeclareLaunchArgument('base_link_pose_topic', default_value='/base_link_pose', description='Pose topic published by aruco_tf_node.'),
        DeclareLaunchArgument('filtered_pose_topic', default_value='/filtered_pose', description='Output topic from kalman_filter_node.'),
        DeclareLaunchArgument('command_topic', default_value='/tello/cmd_vel', description='Command topic used by the optional Kalman control-input model.'),
        DeclareLaunchArgument('enable_kalman_filter', default_value='true', description='Launch kalman_filter_node for offline tuning.'),
        DeclareLaunchArgument('kalman_measurement_timeout_sec', default_value='0.5', description='Measurement timeout used by kalman_filter_node.'),
        DeclareLaunchArgument('kalman_timer_period_sec', default_value='0.01', description='Prediction timer period used by kalman_filter_node.'),
        DeclareLaunchArgument('kalman_sigma_a', default_value='30.0', description='Process noise acceleration sigma used by kalman_filter_node.'),
        DeclareLaunchArgument('kalman_measurement_var', default_value='0.1', description='Measurement variance used by kalman_filter_node.'),
        DeclareLaunchArgument('kalman_publish_prediction_without_measurement', default_value='true', description='Publish predicted filtered_pose for a limited time when pose measurements stop.'),
        DeclareLaunchArgument('kalman_max_prediction_duration_sec', default_value='2.0', description='Maximum time to publish predicted filtered_pose after the last pose measurement.'),
        DeclareLaunchArgument('kalman_use_control_input_model', default_value='false', description='Use cmd_vel and the gamma plant model in kalman_filter_node prediction.'),
        DeclareLaunchArgument('kalman_control_input_timeout_sec', default_value='0.25', description='Maximum age of a cmd_vel sample used by the Kalman control-input model.'),
        DeclareLaunchArgument('kalman_control_input_delay_sec', default_value='0.0', description='Delay applied when selecting cmd_vel samples for the Kalman control-input model.'),
        DeclareLaunchArgument('kalman_control_model_max_step_sec', default_value='0.02', description='Maximum internal integration step for the Kalman control-input model.'),
        DeclareLaunchArgument('kalman_control_model_gamma1', default_value='-1.22379710012708', description='Effective Gamma1 relative to the published cmd_vel.x used by the Kalman control-input model.'),
        DeclareLaunchArgument('kalman_control_model_gamma2', default_value='-0.5496637058581929', description='Gamma2 used by the Kalman control-input model.'),
        DeclareLaunchArgument('kalman_control_model_gamma3', default_value='1.230339924965825', description='Effective Gamma3 relative to the published cmd_vel.y used by the Kalman control-input model.'),
        DeclareLaunchArgument('kalman_control_model_gamma4', default_value='-0.5775755765146152', description='Gamma4 used by the Kalman control-input model.'),
        DeclareLaunchArgument('kalman_control_model_gamma5', default_value='0.6837187318618236', description='Gamma5 used by the Kalman control-input model.'),
        DeclareLaunchArgument('kalman_control_model_gamma6', default_value='-1.0763629574178275', description='Gamma6 used by the Kalman control-input model.'),
        DeclareLaunchArgument('kalman_control_model_gamma7', default_value='-2.033762293431863', description='Gamma7 used by the Kalman control-input model.'),
        DeclareLaunchArgument('kalman_control_model_gamma8', default_value='-2.036870107158528', description='Gamma8 used by the Kalman control-input model.'),
        DeclareLaunchArgument('aruco_reject_low_tag_count', default_value='true', description='Reject ArUco pose measurements when too few known tags are visible.'),
        DeclareLaunchArgument('aruco_min_pose_tag_count', default_value='3', description='Minimum number of known visible tags required before publishing a pose.'),
        DeclareLaunchArgument('aruco_reject_degenerate_tag_geometry', default_value='true', description='Reject ArUco pose measurements when visible tags form weak/near-collinear geometry.'),
        DeclareLaunchArgument('aruco_min_tag_geometry_area', default_value='0.03', description='Minimum triangle area in the tag-map YZ plane required for visible tag geometry.'),
        DeclareLaunchArgument('aruco_min_tag_span_y', default_value='0.20', description='Minimum visible tag span along map Y required for pose publication.'),
        DeclareLaunchArgument('aruco_min_tag_span_z', default_value='0.20', description='Minimum visible tag span along map Z required for pose publication.'),
        DeclareLaunchArgument('aruco_use_pnp_extrinsic_guess', default_value='true', description='Use previous accepted PnP pose as initial guess for the next multi-tag solvePnP call.'),
        DeclareLaunchArgument('aruco_pnp_guess_timeout_sec', default_value='0.50', description='Maximum time gap before resetting the previous PnP extrinsic guess.'),
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
                {'reject_low_tag_count': ParameterValue(aruco_reject_low_tag_count, value_type=bool)},
                {'min_pose_tag_count': ParameterValue(aruco_min_pose_tag_count, value_type=int)},
                {'reject_degenerate_tag_geometry': ParameterValue(aruco_reject_degenerate_tag_geometry, value_type=bool)},
                {'min_tag_geometry_area': ParameterValue(aruco_min_tag_geometry_area, value_type=float)},
                {'min_tag_span_y': ParameterValue(aruco_min_tag_span_y, value_type=float)},
                {'min_tag_span_z': ParameterValue(aruco_min_tag_span_z, value_type=float)},
                {'use_pnp_extrinsic_guess': ParameterValue(aruco_use_pnp_extrinsic_guess, value_type=bool)},
                {'pnp_guess_timeout_sec': ParameterValue(aruco_pnp_guess_timeout_sec, value_type=float)},
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
                'publish_prediction_without_measurement': ParameterValue(kalman_publish_prediction_without_measurement, value_type=bool),
                'max_prediction_duration_sec': ParameterValue(kalman_max_prediction_duration_sec, value_type=float),
                'command_topic': command_topic,
                'use_control_input_model': ParameterValue(kalman_use_control_input_model, value_type=bool),
                'control_input_timeout_sec': ParameterValue(kalman_control_input_timeout_sec, value_type=float),
                'control_input_delay_sec': ParameterValue(kalman_control_input_delay_sec, value_type=float),
                'control_model_max_step_sec': ParameterValue(kalman_control_model_max_step_sec, value_type=float),
                'control_model_gamma1': ParameterValue(kalman_control_model_gamma1, value_type=float),
                'control_model_gamma2': ParameterValue(kalman_control_model_gamma2, value_type=float),
                'control_model_gamma3': ParameterValue(kalman_control_model_gamma3, value_type=float),
                'control_model_gamma4': ParameterValue(kalman_control_model_gamma4, value_type=float),
                'control_model_gamma5': ParameterValue(kalman_control_model_gamma5, value_type=float),
                'control_model_gamma6': ParameterValue(kalman_control_model_gamma6, value_type=float),
                'control_model_gamma7': ParameterValue(kalman_control_model_gamma7, value_type=float),
                'control_model_gamma8': ParameterValue(kalman_control_model_gamma8, value_type=float),
            }],
            remappings=[
                ('/natnet_ros/Bebop1/pose', base_link_pose_topic),
                ('/filtered_pose', filtered_pose_topic),
            ],
        ),
        TimerAction(
            period=playback_delay_sec,
            actions=[
                OpaqueFunction(function=_bag_play_action),
            ],
        ),
    ])
