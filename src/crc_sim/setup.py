from setuptools import find_packages, setup

package_name = 'crc_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='UEH CRC Team',
    maintainer_email='3itech@ueh.edu.vn',
    description='Gazebo simulation for UEH Creative Robot Contest 2026',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'starter = crc_sim.starter_node:main',
            'web_viewer = crc_sim.web_viewer:main',
        ],
    },
)
