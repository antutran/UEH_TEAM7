from setuptools import setup

package_name = 'red_follower'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='UEH CRC 2026 Organising Committee',
    maintainer_email='crc@ueh.edu.vn',
    description='Example team package: follow a red circle',
    license='Apache-2.0',
    entry_points={'console_scripts': ['red_follower = red_follower.red_follower:main']},
)
