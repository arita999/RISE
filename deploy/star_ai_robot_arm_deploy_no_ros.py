#!/usr/bin/env python3
"""Run a trained OpenPi policy without ROS.

Inputs:
- three OpenCV-readable image sources for side/rear/onhand cameras
- a joint-state source from zeros, a JSON file, or JSON lines on stdin

Outputs:
- selected policy action dimensions as JSON to stdout, a file, JSONL, UDP, or TCP
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openpi_client import image_tools, websocket_client_policy


shutdown_event = threading.Event()


def _on_sigint(_signum, _frame):
    shutdown_event.set()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_source(source: str) -> str | int:
    if source.isdigit():
        return int(source)
    return source


def _jpeg_roundtrip(img: np.ndarray) -> np.ndarray:
    encoded = cv2.imencode(".jpg", img)[1].tobytes()
    return cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise RuntimeError("image is None")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise RuntimeError(f"Unsupported image shape: {img.shape}")


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


class FrameSource:
    def __init__(self, source: str, name: str):
        self.source = source
        self.name = name
        self.static_image: np.ndarray | None = None
        self.capture: cv2.VideoCapture | None = None
        self._open()

    def _open(self):
        path = Path(self.source)
        if path.is_file():
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                self.static_image = _ensure_bgr(image)
                return

        self.capture = cv2.VideoCapture(_parse_source(self.source))
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open {self.name} image source: {self.source}")

    def read(self) -> np.ndarray:
        if self.static_image is not None:
            return self.static_image.copy()

        assert self.capture is not None
        ok, frame = self.capture.read()
        if not ok:
            # Allow finite video files to loop for bench testing.
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame from {self.name}: {self.source}")
        return _ensure_bgr(frame)

    def close(self):
        if self.capture is not None:
            self.capture.release()


class JointStateReader:
    def read(self) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _parse_payload(payload: Any, arm_dof: int) -> np.ndarray:
        if isinstance(payload, dict):
            if "position" in payload:
                payload = payload["position"]
            elif "qpos" in payload:
                payload = payload["qpos"]
            elif "state" in payload:
                payload = payload["state"]
        arr = np.asarray(payload, dtype=float).reshape(-1)
        if len(arr) < arm_dof:
            raise RuntimeError(f"Joint state has {len(arr)} dims, but arm_dof={arm_dof}")
        return arr[:arm_dof].copy()


class ZeroJointStateReader(JointStateReader):
    def __init__(self, arm_dof: int):
        self.arm_dof = arm_dof

    def read(self) -> np.ndarray:
        return np.zeros(self.arm_dof, dtype=float)


class FileJointStateReader(JointStateReader):
    def __init__(self, path: str, arm_dof: int):
        self.path = Path(path)
        self.arm_dof = arm_dof
        self.last = np.zeros(arm_dof, dtype=float)

    def read(self) -> np.ndarray:
        if not self.path.exists():
            return self.last.copy()
        with self.path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.last = self._parse_payload(payload, self.arm_dof)
        return self.last.copy()


class StdinJointStateReader(JointStateReader):
    def __init__(self, arm_dof: int):
        self.arm_dof = arm_dof
        self.lock = threading.Lock()
        self.last = np.zeros(arm_dof, dtype=float)
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        for line in sys.stdin:
            if shutdown_event.is_set():
                break
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                qpos = self._parse_payload(payload, self.arm_dof)
                with self.lock:
                    self.last = qpos
            except Exception as exc:
                print(f"[WARN] ignored stdin joint state: {exc}", file=sys.stderr)

    def read(self) -> np.ndarray:
        with self.lock:
            return self.last.copy()


class DirectStarAIArm:
    """Small no-ROS StarAI adapter using Fashionstar's UART SDK directly."""

    motor_names = ("Motor_0", "Motor_1", "Motor_2", "Motor_3", "Motor_4", "Motor_5", "gripper")

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.port_handler = None
        self.servo_ids = [int(x) for x in _csv(args.starai_servo_ids)]
        if len(self.servo_ids) != args.arm_dof:
            raise RuntimeError("--starai_servo_ids must contain arm_dof ids")
        self.motor_names = tuple(_csv(args.starai_motor_names) or self.motor_names)
        if len(self.motor_names) != args.arm_dof:
            raise RuntimeError("--starai_motor_names must contain arm_dof names")

    def connect(self):
        try:
            from fashionstar_uart_sdk.uart_pocket_handler import PortHandler
        except ImportError as exc:
            raise RuntimeError(
                "fashionstar-uart-sdk is required for --starai_port. Install requirements.txt first."
            ) from exc

        self.port_handler = PortHandler(self.args.starai_port, self.args.starai_baudrate)
        self.port_handler.openPort()
        for servo_id in self.servo_ids:
            if not self.port_handler.ping(servo_id):
                raise RuntimeError(f"StarAI servo id {servo_id} not found on {self.args.starai_port}")
        if self.args.starai_reset_loop:
            self.port_handler.ResetLoop(0xFF)

    def close(self):
        if self.port_handler is not None and self.port_handler.is_open:
            self.port_handler.closePort()

    def _degrees_to_policy(self, degrees: np.ndarray) -> np.ndarray:
        if self.args.starai_policy_units == "radians":
            return np.deg2rad(degrees)
        if self.args.starai_policy_units == "degrees":
            return degrees
        raise RuntimeError(f"Unknown starai_policy_units: {self.args.starai_policy_units}")

    def _policy_to_degrees(self, target: np.ndarray) -> np.ndarray:
        if self.args.starai_policy_units == "radians":
            return np.rad2deg(target)
        if self.args.starai_policy_units == "degrees":
            return target
        raise RuntimeError(f"Unknown starai_policy_units: {self.args.starai_policy_units}")

    def read_qpos(self) -> np.ndarray:
        if self.port_handler is None or not self.port_handler.is_open:
            raise RuntimeError("StarAI arm is not connected")
        ids_by_name = dict(zip(self.motor_names, self.servo_ids, strict=False))
        monitor_data = self.port_handler.sync_read["Monitor"](ids_by_name)
        degrees = np.asarray([monitor_data[name].current_position for name in self.motor_names], dtype=float)
        return self._degrees_to_policy(degrees)

    def send_policy_action(self, target: np.ndarray) -> dict[str, Any]:
        if self.port_handler is None or not self.port_handler.is_open:
            raise RuntimeError("StarAI arm is not connected")
        from fashionstar_uart_sdk.uart_pocket_handler import SyncPositionControlOptions

        degrees = np.clip(
            self._policy_to_degrees(np.asarray(target, dtype=float)),
            self.args.starai_min_degrees,
            self.args.starai_max_degrees,
        )
        command = {}
        for name, servo_id, degree in zip(self.motor_names, self.servo_ids, degrees, strict=False):
            command[name] = SyncPositionControlOptions(
                servo_id,
                int(float(degree) * 10.0),
                int(self.args.starai_motion_time_ms),
                int(self.args.starai_power),
                int(self.args.starai_acc_time_ms),
                int(self.args.starai_dec_time_ms),
            )
        self.port_handler.sync_write["Goal_Position"](command)
        return {f"{name}.pos": float(value) for name, value in zip(self.motor_names, target, strict=False)}


class StarAIJointStateReader(JointStateReader):
    def __init__(self, arm: DirectStarAIArm):
        self.arm = arm

    def read(self) -> np.ndarray:
        return self.arm.read_qpos()


class CommandSink:
    def publish(self, message: dict[str, Any]):
        raise NotImplementedError

    def close(self):
        pass


class StdoutCommandSink(CommandSink):
    def publish(self, message: dict[str, Any]):
        print(json.dumps(message, ensure_ascii=False), flush=True)


class JsonlCommandSink(CommandSink):
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def publish(self, message: dict[str, Any]):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")


class FileCommandSink(CommandSink):
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def publish(self, message: dict[str, Any]):
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(message, f, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, self.path)


class UdpCommandSink(CommandSink):
    def __init__(self, host: str, port: int):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self, message: dict[str, Any]):
        data = json.dumps(message, ensure_ascii=False).encode("utf-8")
        self.sock.sendto(data, self.addr)

    def close(self):
        self.sock.close()


class TcpCommandSink(CommandSink):
    def __init__(self, host: str, port: int):
        self.addr = (host, port)
        self.sock: socket.socket | None = None

    def _connect(self):
        if self.sock is None:
            self.sock = socket.create_connection(self.addr, timeout=2.0)

    def publish(self, message: dict[str, Any]):
        data = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            self._connect()
            assert self.sock is not None
            self.sock.sendall(data)
        except OSError:
            self.close()
            self._connect()
            assert self.sock is not None
            self.sock.sendall(data)

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None


class StarAICommandSink(CommandSink):
    def __init__(self, arm: DirectStarAIArm):
        self.arm = arm

    def publish(self, message: dict[str, Any]):
        self.arm.send_policy_action(np.asarray(message["position"], dtype=float))


class StreamActionBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.cur_chunk: deque[np.ndarray] = deque()
        self.k = 0
        self.last_action: np.ndarray | None = None

    def integrate_new_chunk(self, actions_chunk: np.ndarray, max_k: int, min_m: int):
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
            old_list = old_list[:overlap_len]
            if overlap_len <= 0:
                self.cur_chunk = deque(new_list, maxlen=None)
                self.k = 0
                return

            w_old = np.array([1.0], dtype=float) if overlap_len == 1 else np.linspace(1.0, 0.0, overlap_len)
            w_new = 1.0 - w_old
            smoothed = [w_old[i] * old_list[i] + w_new[i] * new_list[i] for i in range(overlap_len)]
            self.cur_chunk = deque(smoothed + new_list[overlap_len:], maxlen=None)
            self.k = 0

    def pop_next_action(self) -> np.ndarray | None:
        with self.lock:
            if not self.cur_chunk:
                return None
            if len(self.cur_chunk) == 1:
                self.last_action = np.asarray(self.cur_chunk[0], dtype=float).copy()
            action = np.asarray(self.cur_chunk.popleft(), dtype=float)
            self.k += 1
            return action


@dataclass
class LatestObservation:
    qpos: np.ndarray
    side: np.ndarray
    rear: np.ndarray
    onhand: np.ndarray


class NoRosObservationSource:
    def __init__(self, args: argparse.Namespace, starai_arm: DirectStarAIArm | None = None):
        self.args = args
        self.side = FrameSource(args.side_source, "side")
        self.rear = FrameSource(args.rear_source, "rear")
        self.onhand = FrameSource(args.onhand_source, "onhand")
        if args.joint_state_source == "zeros":
            self.joint_reader = ZeroJointStateReader(args.arm_dof)
        elif args.joint_state_source == "file":
            self.joint_reader = FileJointStateReader(args.joint_state_file, args.arm_dof)
        elif args.joint_state_source == "stdin":
            self.joint_reader = StdinJointStateReader(args.arm_dof)
        elif args.joint_state_source == "starai":
            if starai_arm is None:
                raise RuntimeError("--joint_state_source starai requires --starai_port")
            self.joint_reader = StarAIJointStateReader(starai_arm)
        else:
            raise ValueError(f"Unknown joint state source: {args.joint_state_source}")

    def read(self) -> LatestObservation:
        side = self.side.read()
        rear = self.rear.read()
        onhand = self.onhand.read()
        if self.args.jpeg_roundtrip:
            side = _jpeg_roundtrip(side)
            rear = _jpeg_roundtrip(rear)
            onhand = _jpeg_roundtrip(onhand)
        return LatestObservation(qpos=self.joint_reader.read(), side=side, rear=rear, onhand=onhand)

    def close(self):
        self.side.close()
        self.rear.close()
        self.onhand.close()


def build_sink(args: argparse.Namespace, starai_arm: DirectStarAIArm | None = None) -> CommandSink:
    if args.command_sink == "stdout":
        return StdoutCommandSink()
    if args.command_sink == "jsonl":
        return JsonlCommandSink(args.command_file)
    if args.command_sink == "file":
        return FileCommandSink(args.command_file)
    if args.command_sink == "udp":
        return UdpCommandSink(args.command_host, args.command_port)
    if args.command_sink == "tcp":
        return TcpCommandSink(args.command_host, args.command_port)
    if args.command_sink == "starai":
        if starai_arm is None:
            raise RuntimeError("--command_sink starai requires --starai_port")
        return StarAICommandSink(starai_arm)
    raise ValueError(f"Unknown command sink: {args.command_sink}")


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


def inference_loop(
    args: argparse.Namespace,
    policy: websocket_client_policy.WebsocketClientPolicy,
    observations: NoRosObservationSource,
    action_buffer: StreamActionBuffer,
):
    period = 1.0 / max(args.inference_rate, 1e-6)
    while not shutdown_event.is_set():
        start = time.time()
        try:
            obs = observations.read()
            payload = {
                "state": obs.qpos,
                "images": _prepare_policy_images(obs.side, obs.rear, obs.onhand),
                "prompt": args.prompt,
            }
            actions = np.asarray(policy.infer(payload)["actions"], dtype=float)
            action_buffer.integrate_new_chunk(actions, max_k=args.latency_k, min_m=args.min_smooth_steps)
            if args.verbose:
                print(f"[INFO] inference chunk shape={actions.shape}", file=sys.stderr)
        except Exception as exc:
            print(f"[WARN] inference failed: {exc}", file=sys.stderr)
        sleep_s = period - (time.time() - start)
        if sleep_s > 0:
            time.sleep(sleep_s)


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy the green_tag policy without ROS.")
    parser.add_argument("--host", type=str, default="localhost", help="OpenPi websocket policy server host")
    parser.add_argument("--port", type=int, default=8000, help="OpenPi websocket policy server port")
    parser.add_argument("--prompt", type=str, default="Untangle the parts", help="Language prompt sent to the policy")

    parser.add_argument("--side_source", type=str, required=True, help="OpenCV source for side camera: index, path, or URL")
    parser.add_argument("--rear_source", type=str, required=True, help="OpenCV source for rear camera: index, path, or URL")
    parser.add_argument("--onhand_source", type=str, required=True, help="OpenCV source for onhand camera: index, path, or URL")

    parser.add_argument("--joint_state_source", choices=["auto", "zeros", "file", "stdin", "starai"], default="auto")
    parser.add_argument("--joint_state_file", type=str, default="/tmp/star_ai_joint_state.json")
    parser.add_argument("--joint_names", type=str, default="joint0,joint1,joint2,joint3,joint4,joint5,joint6")
    parser.add_argument("--arm_dof", type=int, default=7)
    parser.add_argument("--action_start", type=int, default=0)
    parser.add_argument("--max_joint_delta", type=float, default=0.05)

    parser.add_argument("--command_sink", choices=["auto", "stdout", "jsonl", "file", "udp", "tcp", "starai"], default="auto")
    parser.add_argument("--command_file", type=str, default="/tmp/star_ai_command.json")
    parser.add_argument("--command_host", type=str, default="127.0.0.1")
    parser.add_argument("--command_port", type=int, default=5005)

    parser.add_argument("--starai_port", type=str, default="", help="Fashionstar UC-01 serial port, e.g. /dev/ttyUSB1")
    parser.add_argument("--starai_baudrate", type=int, default=1_000_000)
    parser.add_argument("--starai_servo_ids", type=str, default="0,1,2,3,4,5,6")
    parser.add_argument("--starai_motor_names", type=str, default="Motor_0,Motor_1,Motor_2,Motor_3,Motor_4,Motor_5,gripper")
    parser.add_argument("--starai_policy_units", choices=["radians", "degrees"], default="radians")
    parser.add_argument("--starai_motion_time_ms", type=int, default=350)
    parser.add_argument("--starai_acc_time_ms", type=int, default=50)
    parser.add_argument("--starai_dec_time_ms", type=int, default=50)
    parser.add_argument("--starai_power", type=int, default=1000)
    parser.add_argument("--starai_min_degrees", type=float, default=-180.0)
    parser.add_argument("--starai_max_degrees", type=float, default=180.0)
    parser.add_argument("--starai_reset_loop", action="store_true", help="Reset multi-turn angle on connect, like StarAI LeRobot bus")

    parser.add_argument("--publish_rate", type=float, default=30.0)
    parser.add_argument("--inference_rate", type=float, default=3.0)
    parser.add_argument("--max_publish_step", type=int, default=10000)
    parser.add_argument("--latency_k", type=int, default=8)
    parser.add_argument("--min_smooth_steps", type=int, default=8)
    parser.add_argument("--jpeg_roundtrip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true", help="Force stdout output and do not use command_sink")
    parser.add_argument("--run_once", action="store_true", help="Run one inference and output the first action")
    parser.add_argument("--no_confirm", action="store_true", help="Start without Enter confirmation")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    joint_names = _csv(args.joint_names)
    if joint_names and len(joint_names) != args.arm_dof:
        parser.error("--joint_names must contain arm_dof names, or pass --joint_names ''")
    return args


def main():
    args = get_arguments()
    signal.signal(signal.SIGINT, _on_sigint)

    joint_names = _csv(args.joint_names) or [f"joint{i}" for i in range(args.arm_dof)]
    starai_arm = DirectStarAIArm(args) if args.starai_port else None
    if args.joint_state_source == "auto":
        args.joint_state_source = "starai" if starai_arm is not None else "file"
    if args.command_sink == "auto":
        args.command_sink = "starai" if starai_arm is not None else "stdout"

    if starai_arm is not None:
        starai_arm.connect()

    observations = NoRosObservationSource(args, starai_arm)
    sink = StdoutCommandSink() if args.dry_run else build_sink(args, starai_arm)
    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print(f"[INFO] Server metadata: {policy.get_server_metadata()}", file=sys.stderr)
    print(f"[INFO] Action slice: [{args.action_start}:{args.action_start + args.arm_dof}]", file=sys.stderr)
    print(f"[INFO] Command sink: {'stdout(dry_run)' if args.dry_run else args.command_sink}", file=sys.stderr)

    first_obs = observations.read()
    print(f"[INFO] Initial qpos: {np.array2string(first_obs.qpos, precision=4)}", file=sys.stderr)
    if not args.no_confirm:
        input("Press Enter to start no-ROS policy inference...")

    action_buffer = StreamActionBuffer()

    if args.run_once:
        payload = {
            "state": first_obs.qpos,
            "images": _prepare_policy_images(first_obs.side, first_obs.rear, first_obs.onhand),
            "prompt": args.prompt,
        }
        actions = np.asarray(policy.infer(payload)["actions"], dtype=float)
        target = select_robot_action(args, actions[0], first_obs.qpos)
        sink.publish(
            {
                "timestamp": time.time(),
                "step": 0,
                "joint_names": joint_names,
                "position": target.tolist(),
                "prompt": args.prompt,
            }
        )
        sink.close()
        observations.close()
        if starai_arm is not None:
            starai_arm.close()
        return

    worker = threading.Thread(target=inference_loop, args=(args, policy, observations, action_buffer), daemon=True)
    worker.start()

    period = 1.0 / max(args.publish_rate, 1e-6)
    step = 0
    try:
        while step < args.max_publish_step and not shutdown_event.is_set():
            start = time.time()
            action = action_buffer.pop_next_action()
            if action is not None:
                obs = observations.read()
                target = select_robot_action(args, action, obs.qpos)
                sink.publish(
                    {
                        "timestamp": time.time(),
                        "step": step,
                        "joint_names": joint_names,
                        "position": target.tolist(),
                        "prompt": args.prompt,
                    }
                )
                step += 1
            sleep_s = period - (time.time() - start)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        shutdown_event.set()
        sink.close()
        observations.close()
        if starai_arm is not None:
            starai_arm.close()


if __name__ == "__main__":
    main()
