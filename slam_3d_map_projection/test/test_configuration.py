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

"""Projection parameter validation tests."""

import math

import pytest

from slam_3d_map_projection.configuration import ProjectionConfig


@pytest.mark.parametrize(
    "field_name",
    [
        "initial_padding_m",
        "max_floor_reference_distance_m",
        "robot_footprint_width_m",
        "trajectory_max_segment_dt_s",
    ],
)
def test_non_finite_real_parameters_are_rejected(field_name):
    with pytest.raises(ValueError, match="finite"):
        ProjectionConfig(**{field_name: math.nan})


def test_image_values_must_be_in_collision_probability_order():
    with pytest.raises(ValueError, match="occupied_value < unknown_value"):
        ProjectionConfig(free_value=100, unknown_value=200, occupied_value=0)


def test_integer_parameters_reject_fractional_values():
    with pytest.raises(ValueError, match="free_value must be an integer"):
        ProjectionConfig(free_value=254.5)


def test_boolean_parameters_reject_numeric_values():
    with pytest.raises(ValueError, match="must be a boolean"):
        ProjectionConfig(fill_enclosed_unknown_as_free=1)
