from glob import glob

from setuptools import find_packages, setup

package_name = 'mtl_landing'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dimuthu Wickramanayake',
    maintainer_email='16587863+deBilla@users.noreply.github.com',
    description='Vision-based landing of a quadcopter on a moving ArUco target',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sim_world = mtl_landing.nodes.sim_world_node:main',
            'aruco_detector = mtl_landing.nodes.aruco_detector_node:main',
            'target_estimator = mtl_landing.nodes.target_estimator_node:main',
            'landing_controller = mtl_landing.nodes.landing_controller_node:main',
            'flight_logger = mtl_landing.nodes.flight_logger_node:main',
        ],
    },
)
