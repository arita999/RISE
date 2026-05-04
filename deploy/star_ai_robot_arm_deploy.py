#!/usr/bin/env python3
"""Run a trained OpenPi policy on a Star AI Robot Arm through ROS topics.

This script is intentionally generic: it subscribes to three RGB image topics
and one JointState topic, sends observations to the websocket policy server,
then publishes the selected action dimensions as a JointState command.
"""

from __future__ import annotations

import argparse
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from openpi_client import image_tools, websocket_client_policy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header


shutdown_event = threading.Event()


def _on_sigint(_signum, _frame):
    shutdown_event.set()
    try:
        rospy.signal_shutdown("SIGINT")
    except Exception:
        pass


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _jpeg_roundtrip(img: np.ndarray) -> np.ndarray:
    encoded = cv2.imencode(".jpg", img)[1].tobytes()
    return cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _prepare_policy_images(side_bgr: np.ndarray, rear_bgr: np.ndarray, onhand_bgr: np.ndarray) -> dict[str, np.ndarray]:
    # The green_tag training config maps side/rear/onhand to top_head/hand_left/hand_right.
    images_rgb = [
        cv2.cvtColor(side_bgr, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(rear_bgr, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(onhand_bgr, cv2.COLOR_BGR2RGB),
    ]
    images_rgb = image_tools.resize_with_pad(np.asarray(images_rgb), 224, 224)
    return {
        "top_head": images_rgb[0].transpose(2, 0, 1),
        "hand_left": images_rgb[1].transpose(2, 0, 1),
        "hand_right": images_rgb[2].transpose(2, 0, 1),
    }


class StreamActionBuffer:
    """Keeps one executable action sequence while blending in newer chunks."""

    def __init__(self, max_chunks: int = 10):
        self.lock = threading.Lock()
        self.max_chunks = max_chunks
        self.cur_chunk: deque[np.ndarray] = deque()
        self.k = 0
        self.last_action: np.ndarray | None = None

    def integrate_new_chunk(self, actions_chunk: np.ndarray, max_k: int, min_m: int = 8):
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return

            max_k = max(0, int(max_k))
            min_m = max(1, int(min_m))
            drop_n = min(self.k, max_k)
            if drop_n >= len(actions_chunk):
                return

            new_list = [np.asarray(a, dtype=float).copy() for a in actions_chunk[drop_n:]]
            if len(self.cur_chunk) == 0 and self.last_action is not None:
                old_list = [self.last_action.copy() for _ in range(min_m)]
                self.last_action = None
            else:
                old_list = list(self.cur_chunk)
                if len(old_list) == 0:
                    self.cur_chunk = deque(new_list, maxlen=None)
                    self.k = 0
                    return
                if len(old_list) < min_m:
                    old_list.extend([np.asarray(old_list[-1], dtype=float).copy() for _ in range(min_m - len(old_list))])

            overlap_len = min(len(old_list), len(new_list))
            if overlap_len <= 0:
                self.cur_chunk = deque(new_list, maxlen=None)
                self.k = 0
                return

            old_list = old_list[:overlap_len]
            if overlap_len == 1:
                w_old = np.array([1.0], dtype=float)
            else:
                w_old = np.linspace(1.0, 0.0, overlap_len, dtype=float)
            w_new = 1.0 - w_old
            smoothed = [w_old[i] * old_list[i] + w_new[i] * new_list[i] for i in range(overlap_len)]
            combined = smoothed + new_list[overlap_len:]
            self.cur_chunk = deque(combined, maxlen=None)
            self.k = 0

    def pop_next_action(self) -> np.ndarray | None:
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None
            if len(self.cur_chunk) == 1:
                self.last_action = np.asarray(self.cur_chunk[0], dtype=float).copy()
            action = np.asarray(self.cur_chunk.popleft(), dtype=float)
            self.k += 1
            return action


@dataclass
class Observation:
    qpos: np.ndarray
    side: np.ndarray
    rear: np.ndarray
    onhand: np.ndarray


class StarAIRosOperator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.side_images: deque[Image] = deque(maxlen=200)
        self.rear_images: deque[Image] = deque(maxlen=200)
        self.onhand_images: deque[Image] = deque(maxlen=200)
        self.joint_states: deque[JointState] = deque(maxlen=200)
        self.joint_names = _csv(args.joint_names)

        rospy.init_node(args.ros_node_name, anonymous=True)
        rospy.Subscriber(args.side_image_topic, Image, self._side_callback, queue_size=10, tcp_nodelay=True)
        rospy.Subscriber(args.rear_image_topic, Image, self._rear_callback, queue_size=10, tcp_nodelay=True)
        rospy.Subscriber(args.onhand_image_topic, Image, self._onhand_callback, queue_size=10, tcp_nodelay=True)
        rospy.Subscriber(args.joint_state_topic, JointState, self._joint_state_callback, queue_size=50, tcp_nodelay=True)
        self.joint_cmd_publisher = rospy.Publisher(args.joint_cmd_topic, JointState, queue_size=10)

    def _side_callback(self, msg: Image):
        with self.lock:
            self.side_images.append(msg)

    def _rear_callback(self, msg: Image):
        with self.lock:
            self.rear_images.append(msg)

    def _onhand_callback(self, msg: Image):
        with self.lock:
            self.onhand_images.append(msg)

    def _joint_state_callback(self, msg: JointState):
        with self.lock:
            self.joint_states.append(msg)

    def _extract_qpos(self, msg: JointState) -> np.ndarray:
        positions = np.asarray(msg.position, dtype=float)
        if self.joint_names and msg.name:
            index_by_name = {name: i for i, name in enumerate(msg.name)}
            missing = [name for name in self.joint_names if name not in index_by_name]
            if missing:
                raise RuntimeError(f"JointState is missing configured joints: {missing}")
            return np.asarray([positions[index_by_name[name]] for name in self.joint_names], dtype=float)

        if len(positions) < self.args.arm_dof:
            raise RuntimeError(f"JointState has {len(positions)} positions, but arm_dof={self.args.arm_dof}")
        return positions[: self.args.arm_dof].astype(float)

    def _decode_image(self, msg: Image) -> np.ndarray:
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        return _ensure_bgr(np.asarray(img))

    def latest_observation(self) -> Observation | None:
        with self.lock:
            if not self.side_images or not self.rear_images or not self.onhand_images or not self.joint_states:
                return None
            side_msg = self.side_images[-1]
            rear_msg = self.rear_images[-1]
            onhand_msg = self.onhand_images[-1]
            joint_msg = self.joint_states[-1]

        side = self._decode_image(side_msg)
        rear = self._decode_image(rear_msg)
        onhand = self._decode_image(onhand_msg)
        if self.args.jpeg_roundtrip:
            side = _jpeg_roundtrip(side)
            rear = _jpeg_roundtrip(rear)
            onhand = _jpeg_roundtrip(onhand)

        return Observation(qpos=self._extract_qpos(joint_msg), side=side, rear=rear, onhand=onhand)

    def wait_for_observation(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not shutdown_event.is_set():
            obs = self.latest_observation()
            if obs is not None:
                return obs
            rospy.loginfo_throttle(5.0, "Waiting for Star AI Robot Arm images and joint state...")
            rate.sleep()
        return None

    def publish_joint_command(self, target: np.ndarray):
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.joint_names or [f"joint{i}" for i in range(len(target))]
        msg.position = np.asarray(target, dtype=float).tolist()
        self.joint_cmd_publisher.publish(msg)


def inference_loop(
    args: argparse.Namespace,
    policy: websocket_client_policy.WebsocketClientPolicy,
    ros_operator: StarAIRosOperator,
    action_buffer: StreamActionBuffer,
):
    rate = rospy.Rate(args.inference_rate)
    while not rospy.is_shutdown() and not shutdown_event.is_set():
        try:
            obs = ros_operator.latest_observation()
            if obs is None:
                rate.sleep()
                continue

            payload = {
                "state": obs.qpos,
                "images": _prepare_policy_images(obs.side, obs.rear, obs.onhand),
                "prompt": args.prompt,
            }
            start = time.time()
            actions = np.asarray(policy.infer(payload)["actions"], dtype=float)
            rospy.loginfo_throttle(5.0, "Policy inference %.1f ms, chunk shape=%s", (time.time() - start) * 1000, actions.shape)
            action_buffer.integrate_new_chunk(actions, max_k=args.latency_k, min_m=args.min_smooth_steps)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Star AI inference loop error: %s", exc)
        rate.sleep()


def select_robot_action(args: argparse.Namespace, policy_action: np.ndarray, current_qpos: np.ndarray) -> np.ndarray:
    action = np.asarray(policy_action, dtype=float).reshape(-1)
    start = int(args.action_start)
    end = start + int(args.arm_dof)
    if len(action) < end:
        raise RuntimeError(f"Policy action has {len(action)} dims, cannot select [{start}:{end}]")

    target = action[start:end].copy()
    if args.max_joint_delta > 0:
        delta = np.full(args.arm_dof, float(args.max_joint_delta), dtype=float)
        target = np.clip(target, current_qpos - delta, current_qpos + delta)
    return target


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy the green_tag OpenPi policy on a Star AI Robot Arm.")
    parser.add_argument("--host", type=str, default="localhost", help="OpenPi websocket policy server host")
    parser.add_argument("--port", type=int, default=8000, help="OpenPi websocket policy server port")
    parser.add_argument("--prompt", type=str, default="Untangle the parts", help="Language prompt sent to the policy")

    parser.add_argument("--side_image_topic", type=str, default="/camera_side/color/image_raw")
    parser.add_argument("--rear_image_topic", type=str, default="/camera_rear/color/image_raw")
    parser.add_argument("--onhand_image_topic", type=str, default="/camera_onhand/color/image_raw")
    parser.add_argument("--joint_state_topic", type=str, default="/joint_states")
    parser.add_argument("--joint_cmd_topic", type=str, default="/joint_command")
    parser.add_argument("--joint_names", type=str, default="joint0,joint1,joint2,joint3,joint4,joint5,joint6")

    parser.add_argument("--arm_dof", type=int, default=7, help="Number of Star arm command dimensions")
    parser.add_argument("--action_start", type=int, default=0, help="First policy action dimension to publish")
    parser.add_argument("--max_joint_delta", type=float, default=0.05, help="Per-step joint delta clamp; set <=0 to disable")

    parser.add_argument("--publish_rate", type=int, default=30)
    parser.add_argument("--inference_rate", type=float, default=3.0)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--max_publish_step", type=int, default=10000)
    parser.add_argument("--latency_k", type=int, default=8)
    parser.add_argument("--min_smooth_steps", type=int, default=8)
    parser.add_argument("--buffer_max_chunks", type=int, default=10)
    parser.add_argument("--ros_node_name", type=str, default="star_ai_robot_arm_openpi_deploy")

    parser.add_argument("--jpeg_roundtrip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true", help="Run inference but do not publish robot commands")
    parser.add_argument("--no_confirm", action="store_true", help="Start publishing without the Enter confirmation")
    args = parser.parse_args()
    joint_names = _csv(args.joint_names)
    if joint_names and len(joint_names) != args.arm_dof:
        parser.error("--joint_names must contain arm_dof names, or pass --joint_names '' to use the first positions")
    return args


def main():
    args = get_arguments()
    signal.signal(signal.SIGINT, _on_sigint)

    ros_operator = StarAIRosOperator(args)
    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print(f"Server metadata: {policy.get_server_metadata()}")
    print(f"Prompt: {args.prompt}")
    print(f"Action slice: [{args.action_start}:{args.action_start + args.arm_dof}] -> {args.joint_cmd_topic}")

    obs = ros_operator.wait_for_observation()
    if obs is None:
        return
    print(f"Initial qpos: {np.array2string(obs.qpos, precision=4)}")
    if not args.no_confirm:
        input("Press Enter to start Star AI Robot Arm policy execution...")

    action_buffer = StreamActionBuffer(max_chunks=args.buffer_max_chunks)
    worker = threading.Thread(target=inference_loop, args=(args, policy, ros_operator, action_buffer), daemon=True)
    worker.start()

    rate = rospy.Rate(args.publish_rate)
    step = 0
    while step < args.max_publish_step and not rospy.is_shutdown() and not shutdown_event.is_set():
        action = action_buffer.pop_next_action()
        if action is None:
            rate.sleep()
            continue

        obs = ros_operator.latest_observation()
        if obs is None:
            rate.sleep()
            continue

        try:
            target = select_robot_action(args, action, obs.qpos)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Cannot publish Star AI action: %s", exc)
            rate.sleep()
            continue

        if args.dry_run:
            rospy.loginfo_throttle(1.0, "Dry run target: %s", np.array2string(target, precision=4))
        else:
            ros_operator.publish_joint_command(target)

        step += 1
        rospy.loginfo_throttle(2.0, "Published Star AI action step %d", step)
        rate.sleep()

    shutdown_event.set()


if __name__ == "__main__":
    main()
