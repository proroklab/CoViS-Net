from setuptools import find_packages, setup
from glob import glob
import os

package_name = "control"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Include all launch files.
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jan",
    maintainer_email="jb2270@cam.ac.uk",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "simple_pose_control = control.simple_pose_control:main",
            "remote_control = control.remote_control:main",
            "trajectory_control = control.trajectory_control:main",
            "multi_agent_trajectory_control = control.multi_agent_trajectory_control:main",
        ],
    },
)
