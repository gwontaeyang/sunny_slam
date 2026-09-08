#!/usr/bin/env python3
"""Start offline playback only after the custom SLAM executor responds."""

import argparse
import math
import os
import time

import rclpy
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def wait_for_slam(node, timeout_s):
    client = node.create_client(GetParameters, "/slam_toolbox/get_parameters")
    deadline = time.monotonic() + timeout_s
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if not client.wait_for_service(timeout_sec=min(0.2, remaining)):
                continue
            request = GetParameters.Request(names=["roughness_constraint_mode"])
            future = client.call_async(request)
            # The custom node starts spinning only after configure/loadPoseGraph.
            rclpy.spin_until_future_complete(
                node, future, timeout_sec=max(0.0, deadline - time.monotonic()))
            if future.done():
                response = future.result()
                if (response is not None and len(response.values) == 1
                        and response.values[0].type == ParameterType.PARAMETER_STRING):
                    return True
                return False
            future.cancel()
        return False
    finally:
        node.destroy_client(client)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True)
    parser.add_argument("--rate", type=positive_float, default=1.0)
    parser.add_argument("--timeout", type=positive_float, default=60.0)
    args = parser.parse_args()
    rclpy.init(args=[])
    node = rclpy.create_node("slam_2d_playback_readiness")
    try:
        ready = wait_for_slam(node, args.timeout)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    if not ready:
        parser.exit(1, "Custom SLAM did not become ready; bag was not started.\n")
    print("Custom SLAM is ready. Starting bag playback.", flush=True)
    os.execvp("ros2", ["ros2", "bag", "play", args.bag, "--clock",
                       "--rate", str(args.rate)])


if __name__ == "__main__":
    main()
