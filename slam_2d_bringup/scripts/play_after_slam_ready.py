#!/usr/bin/env python3
"""Wait for the project SLAM executor and scan subscription, then replay a bag."""

import argparse
import math
import os
import time

import rclpy
from rcl_interfaces.srv import GetParameters


def wait_for_slam(scan_topic, timeout_s):
    node = rclpy.create_node("slam_bag_readiness")
    client = node.create_client(GetParameters, "/slam_toolbox/get_parameters")
    deadline = time.monotonic() + timeout_s
    future = None
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            if future is None and client.service_is_ready():
                request = GetParameters.Request()
                request.names = ["scan_topic"]
                future = client.call_async(request)
            rclpy.spin_once(node, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))
            if future is not None and future.done():
                response = future.result()
                if response is None:
                    raise RuntimeError("SLAM readiness parameter request failed")
                # A response proves the node has finished configure/loadPoseGraph
                # and entered its executor; this node has no lifecycle state.
                subscriptions = node.get_subscriptions_info_by_topic(scan_topic)
                if any(info.node_name == "slam_toolbox" and info.node_namespace == "/"
                       and info.topic_type == "sensor_msgs/msg/LaserScan"
                       for info in subscriptions):
                    return
        raise RuntimeError(f"SLAM was not ready on {scan_topic} within {timeout_s:g}s")
    finally:
        node.destroy_node()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument("--scan-topic", default="/scan_laserscan")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--topics", nargs="+")
    args = parser.parse_args()
    for name, number, allow_zero in (
        ("rate", args.rate, False), ("start-offset", args.start_offset, True),
        ("timeout", args.timeout, False),
    ):
        if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
            parser.error(f"--{name} must be finite and {'non-negative' if allow_zero else 'positive'}")
    rclpy.init(args=[])
    try:
        wait_for_slam(args.scan_topic, args.timeout)
    except KeyboardInterrupt:
        return 130
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return 1
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    command = ["ros2", "bag", "play", args.bag, "--clock", "--rate", str(args.rate),
               "--start-offset", str(args.start_offset), "--disable-keyboard-controls"]
    if args.topics:
        command += ["--topics", *args.topics]
    # Replace this process so launch owns the player PID and delivers Ctrl-C.
    os.execvp(command[0], command)


if __name__ == "__main__":
    raise SystemExit(main())
