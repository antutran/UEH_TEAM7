from glob import glob

from setuptools import setup

package_name = 'crc_team'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='UEH CRC 2026 Organising Committee',
    maintainer_email='crc@ueh.edu.vn',
    description='UEH CRC 2026 team starter: lane following, traffic light, signs and obstacles',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'drive_square = crc_team.drive_square:main',
        'race = crc_team.race_node:main',
    ]},
)
