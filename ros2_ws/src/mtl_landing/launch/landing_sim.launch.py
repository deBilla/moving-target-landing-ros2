"""Full closed-loop simulation: sim_world -> aruco_detector -> target_estimator ->
landing_controller -> sim_world, plus flight_logger. The launch ends when the logger exits
(after touchdown or t_max).

  ros2 launch mtl_landing landing_sim.launch.py scenario:=weave mode:=corrected
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    scenario = LaunchConfiguration('scenario')
    mode = LaunchConfiguration('mode')
    seed = LaunchConfiguration('seed')
    out = LaunchConfiguration('output_dir')
    t_max = LaunchConfiguration('t_max')
    sim_time = {'use_sim_time': True}

    logger = Node(package='mtl_landing', executable='flight_logger', name='flight_logger',
                  parameters=[sim_time, {'scenario': scenario, 'mode': mode,
                                         'output_dir': out, 't_max': t_max}],
                  output='screen')

    return LaunchDescription([
        DeclareLaunchArgument('scenario', default_value='straight',
                              description='static|straight|accelerating|circle|weave|stop_go|fast'),
        DeclareLaunchArgument('mode', default_value='corrected', description='paper|naive|compensated|corrected'),
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('wind_sigma', default_value='0.6'),
        DeclareLaunchArgument('output_dir', default_value='/results'),
        DeclareLaunchArgument('t_max', default_value='90.0'),

        Node(package='mtl_landing', executable='sim_world', name='sim_world',
             parameters=[{'scenario': scenario, 'seed': seed,
                          'wind_sigma': LaunchConfiguration('wind_sigma')}],
             output='screen'),
        Node(package='mtl_landing', executable='aruco_detector', name='aruco_detector',
             parameters=[sim_time], output='screen'),
        Node(package='mtl_landing', executable='target_estimator', name='target_estimator',
             parameters=[sim_time, {'mode': mode}], output='screen'),
        Node(package='mtl_landing', executable='landing_controller', name='landing_controller',
             parameters=[sim_time], output='screen'),
        logger,
        RegisterEventHandler(OnProcessExit(target_action=logger,
                                           on_exit=[EmitEvent(event=Shutdown(reason='run complete'))])),
    ])
