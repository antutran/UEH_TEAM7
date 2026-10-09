from glob import glob

from setuptools import find_packages, setup

package_name = 'crc_bringup'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='UEH CRC 2026 Organising Committee',
    maintainer_email='crc@ueh.edu.vn',
    description='UEH CRC 2026 car bring-up (base driver, camera bridge, scan adapter)',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'bringup = crc_bringup.bringup:main',
            'base_driver = crc_bringup.base_driver:main',
            'camera_node = crc_bringup.camera_node:main',
            'scan_adapter = crc_bringup.scan_adapter:main',
            'static_frames = crc_bringup.static_frames:main',
        ],
    },
)
