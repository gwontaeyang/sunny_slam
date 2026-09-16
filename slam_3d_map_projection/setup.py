# Copyright 2026 Gole Robotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from glob import glob

from setuptools import find_packages, setup


package_name = "slam_3d_map_projection"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "docs"), glob("docs/*.md")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="gwontaeyang",
    maintainer_email="gwontaeyang@users.noreply.github.com",
    description="Project optimized 3D SLAM PCD maps into conservative Nav2 grids.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pcd_to_nav_grid = slam_3d_map_projection.cli:main",
            "refine_nav_grid_resolution = "
            "slam_3d_map_projection.multires_cli:main",
            "raw_scans_to_nav_grid = slam_3d_map_projection.raw_cli:main",
        ],
    },
)
