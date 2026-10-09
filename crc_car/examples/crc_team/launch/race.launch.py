"""Start the race driver with one command (the Rulebook asks for exactly that).

    ros2 launch crc_team race.launch.py                 # config/race_car.yaml
    ros2 launch crc_team race.launch.py config:=sim     # config/race_sim.yaml
    ros2 launch crc_team race.launch.py config:=/data/my_race.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def make_nodes(context):
    name = LaunchConfiguration('config').perform(context)
    if os.sep not in name and not name.endswith('.yaml'):
        name = os.path.join(get_package_share_directory('crc_team'), 'config', f'race_{name}.yaml')
    return [Node(package='crc_team', executable='race', output='screen',
                 parameters=[{'config': name}])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value='car',
                              description='car, sim, or the path of a YAML file'),
        OpaqueFunction(function=make_nodes),
    ])
