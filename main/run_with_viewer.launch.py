from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='crc_sim',
            executable='starter',
            name='starter_node',
            output='screen',
        ),
        Node(
            package='crc_sim',
            executable='web_viewer',
            name='web_viewer_node',
            output='screen',
        ),
    ])
