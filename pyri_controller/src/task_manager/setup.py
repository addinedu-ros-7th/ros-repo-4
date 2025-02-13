from setuptools import find_packages, setup
import glob, os

package_name = 'task_manager'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob(os.path.join('launch', '*.launch.xml'))),
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
            'task_manager = task_manager.task_manager:main',
            'pyri_state = task_manager.pyri_state:main',
            'fire_door = task_manager.fire_door:main',
            'service_listener = task_manager.service_listener:main',
        ],
    },
)
