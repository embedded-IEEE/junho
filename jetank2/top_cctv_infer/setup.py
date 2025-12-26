from setuptools import setup, find_packages

package_name = 'top_cctv_infer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (
            'share/' + package_name,
            ['package.xml', 'top_cctv_infer/best.pt'],
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jwson',
    maintainer_email='jwson@ssafy.local',
    description='ROS2 nodes for CCTV OBB inference and service API',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'infer_node = top_cctv_infer.infer_node:main',
            'infer_service_server = top_cctv_infer.infer_service_server:main',
            'dummy_client = top_cctv_infer.dummy_client:main',
        ],
    },
)
