from setuptools import find_packages, setup

package_name = 'move_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='so',
    maintainer_email='sydneycan9@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nav_pkg = move_node.nav_pkg:main',
            'dest_move = move_node.destination_move:main',
            'waypoint = move_node.waypoint_move:main',
            'sub = move_node.sub:main',
        ],
    },
)
