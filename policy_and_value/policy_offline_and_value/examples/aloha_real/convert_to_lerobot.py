import os
os.environ["SVT_LOG"] = "1"

import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
from pathlib import Path
import shutil
import logging
import time
from functools import partial
import subprocess

import numpy as np
import tyro
import av
import h5py
import pyarrow as pa
import pyarrow.parquet as pq

from mini_lerobot.builder import LeRobotDatasetBuilder
from mini_lerobot.metadata import DEFAULT_FEATURES


# * Example usage:
# * python examples/aloha_real/convert_aloha_data_to_lerobot_kai.py --data-dir /cpfs01/user/yangjiazhi/workspace/VLA/Datasets/KAI/rl_yjz/rl_init_pick_place_2/ --repo-ids aloha_mobile_dummy  --prompt ""Pick and sort bricks on the conveyor." --save-dir /cpfs01/user/yangjiazhi/workspace/VLA/Datasets/KAI/huggingface/kai_convert --save_repoid rl_yjz_init_pick_place_2 


OPTIONAL_FEATURES = ("noise", "inferred_action")
LEROBOT_DEFAULT_FEATURE_KEYS = set(DEFAULT_FEATURES.keys())

FEATURES = {
    "observation.images.top_head": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.hand_left": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.hand_right": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (14,),
    },
    "action": {
        "dtype": "float32",
        "shape": (14,),
    },


    # * Optional
    # * for offline PPO
    "noise":{
        "dtype": "float32",
        "shape": (700, ),
    },

    "inferred_action": {
        "dtype": "float32",
        "shape": (700,),
    }
}


@dataclass(frozen=True)
class LeRobotV3Episode:
    episode_index: int
    length: int
    tasks: tuple[str, ...]
    data_chunk_index: int
    data_file_index: int
    video_spans: dict[str, tuple[int, int, float, float]]


def is_lerobot_v3_dataset(data_dir: Path) -> bool:
    return (
        (data_dir / "meta" / "info.json").is_file()
        and (data_dir / "data").is_dir()
        and (data_dir / "meta" / "episodes").is_dir()
    )


def format_lerobot_v3_path(
    path_template: str,
    *,
    chunk_index: int,
    file_index: int,
    video_key: str | None = None,
) -> str:
    return path_template.format(
        chunk_index=chunk_index,
        file_index=file_index,
        video_key=video_key,
    )


def arrow_column_to_numpy(column: pa.ChunkedArray | pa.Array) -> np.ndarray:
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()

    if isinstance(column, pa.FixedSizeListArray):
        flat_values = column.flatten().to_numpy(zero_copy_only=False)
        return flat_values.reshape(len(column), column.type.list_size)

    if isinstance(column, (pa.ListArray, pa.LargeListArray)):
        return np.array(column.to_pylist())

    return column.to_numpy(zero_copy_only=False)


def load_lerobot_v3_info(data_dir: Path) -> dict[str, Any]:
    with open(data_dir / "meta" / "info.json", "r") as f:
        return json.load(f)


def resolve_lerobot_v3_features(info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    features = {}
    for key, feature in info["features"].items():
        if key in LEROBOT_DEFAULT_FEATURE_KEYS:
            continue

        resolved = {
            "dtype": feature["dtype"],
            "shape": tuple(feature["shape"]),
        }
        if feature.get("names") is not None:
            resolved["names"] = feature["names"]
        features[key] = resolved
    return features


def load_lerobot_v3_episodes(data_dir: Path, video_keys: list[str]) -> list[LeRobotV3Episode]:
    episode_files = sorted((data_dir / "meta" / "episodes").rglob("*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No episode metadata parquet files found under {data_dir / 'meta' / 'episodes'}")

    episodes = []
    for episode_file in episode_files:
        table = pq.read_table(episode_file)
        columns = table.to_pydict()

        for row in range(table.num_rows):
            video_spans = {}
            for video_key in video_keys:
                video_prefix = f"videos/{video_key}"
                video_spans[video_key] = (
                    int(columns[f"{video_prefix}/chunk_index"][row]),
                    int(columns[f"{video_prefix}/file_index"][row]),
                    float(columns[f"{video_prefix}/from_timestamp"][row]),
                    float(columns[f"{video_prefix}/to_timestamp"][row]),
                )

            episodes.append(
                LeRobotV3Episode(
                    episode_index=int(columns["episode_index"][row]),
                    length=int(columns["length"][row]),
                    tasks=tuple(columns["tasks"][row]),
                    data_chunk_index=int(columns["data/chunk_index"][row]),
                    data_file_index=int(columns["data/file_index"][row]),
                    video_spans=video_spans,
                )
            )

    episodes.sort(key=lambda episode: episode.episode_index)
    expected_indices = list(range(len(episodes)))
    actual_indices = [episode.episode_index for episode in episodes]
    if actual_indices != expected_indices:
        raise ValueError(f"Episode indices are not contiguous: first indices are {actual_indices[:10]}")
    return episodes


def resolve_output_features(valid_files: List[Path]) -> tuple[Dict, set[str]]:
    enabled_optional_features = set(OPTIONAL_FEATURES)
    for file in valid_files:
        with h5py.File(file, "r") as f:
            enabled_optional_features &= {key for key in OPTIONAL_FEATURES if key in f}
        if not enabled_optional_features:
            break

    resolved_features = {
        key: value
        for key, value in FEATURES.items()
        if key not in OPTIONAL_FEATURES or key in enabled_optional_features
    }
    return resolved_features, enabled_optional_features

def lazy_load_hdf5_dataset(
    episode_path: str | Path,
) -> Tuple[Dict, h5py.File]:
    """Load hdf5 dataset and return a dict with observations and actions"""
    f = h5py.File(episode_path, 'r')

    state_qpos = np.array(f["observations/qpos"])
    
    epi_len = state_qpos.shape[0]

    if "inferred_action" in f.keys():
        inferred_action = np.array(f["inferred_action"]).astype(np.float32)
        inferred_action = inferred_action.reshape((epi_len, -1))  # [ep_len, 14]
        


    if "noise" in f.keys():
        noise = np.array(f["noise"]).astype(np.float32)
        noise = noise.reshape((epi_len, -1))  # [epi_len, 1, 50, 14] -> [epi_len, 50 * 14]
        
        
        

    # epi_len = state_qpos.shape[0]
    episode = {
        "observation.state": state_qpos.reshape((epi_len, -1)),
        "observation.images.top_head": None,
        "observation.images.hand_left": None,
        "observation.images.hand_right": None,
        "action": state_qpos.reshape((epi_len, -1)),
        "epi_len": epi_len
    }

    if "inferred_action" in f.keys():
        episode["inferred_action"] = inferred_action
    if "noise" in f.keys():
        episode["noise"] = noise

    return episode, f


def encode_video_frames(
        images: np.ndarray, 
        dst: Path,
        fps: int,
        vcodec: str = "libsvtav1",
        pix_fmt: str = "yuv420p",
        g: int | None = 2,
        crf: int | None = 30,
        fast_decode: int = 0,
        log_level: int | None = av.logging.ERROR,
        overwrite: bool = False,
) -> bytes:
    """More info on ffmpeg arguments tuning on `benchmark/video/README.md`"""
    # Check encoder availability
    if vcodec not in ["h264", "hevc", "libsvtav1"]:
        raise ValueError(f"Unsupported video codec: {vcodec}. Supported codecs are: h264, hevc, libsvtav1.")

    video_path = Path(dst)

    video_path.parent.mkdir(parents=True, exist_ok=overwrite)

    # Encoders/pixel formats incompatibility check
    if (vcodec == "libsvtav1" or vcodec == "hevc") and pix_fmt == "yuv444p":
        print(
            f"Incompatible pixel format 'yuv444p' for codec {vcodec}, auto-selecting format 'yuv420p'"
        )
        pix_fmt = "yuv420p"

    # Define video output frame size (assuming all input frames are the same size)

    dummy_image = images[0]
    height, width, _ = dummy_image.shape

    # Define video codec options
    video_options = {}

    if g is not None:
        video_options["g"] = str(g)

    if crf is not None:
        video_options["crf"] = str(crf)

    if fast_decode:
        key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        video_options[key] = value

    # Set logging level
    if log_level is not None:
        # "While less efficient, it is generally preferable to modify logging with Python’s logging"
        logging.getLogger("libav").setLevel(log_level)

    # Create and open output file (overwrite by default)
    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        # Loop through input frames and encode them
        for input_image in images:
            # input_image = Image.open(input_data).convert("RGB")
            # input_frame = av.VideoFrame.from_image(input_image)
            input_frame = av.VideoFrame.from_ndarray(input_image, format="rgb24", channel_last=True)
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)

        # Flush the encoder
        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    # Reset logging level
    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")


def encode_video_segment(
    src: Path,
    dst: Path,
    fps: int,
    start_frame: int,
    num_frames: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int = 2,
    crf: int = 30,
):
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if not src.exists():
        raise FileNotFoundError(f"Source video not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    logging.getLogger("libav").setLevel(av.logging.ERROR)

    video_options = {"g": str(g), "crf": str(crf)}
    if os.getenv("FFMPEG_SINGLE_THREAD", "0") == "1":
        video_options["svtav1-params"] = "lp=1"

    with av.open(str(src), "r") as input_container, av.open(str(dst), "w") as output_container:
        input_stream = input_container.streams.video[0]
        input_stream.thread_type = "AUTO"

        frame_duration_pts = int(round((1 / fps) / input_stream.time_base))
        seek_pts = max(0, start_frame - 2) * frame_duration_pts
        input_container.seek(seek_pts, stream=input_stream, any_frame=False, backward=True)

        output_stream = output_container.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = input_stream.width
        output_stream.height = input_stream.height

        stop_frame = start_frame + num_frames
        written = 0
        done = False

        for packet in input_container.demux(input_stream):
            for frame in packet.decode():
                if frame.pts is None:
                    continue
                frame_index = int(round((frame.pts * input_stream.time_base) * fps))
                if frame_index < start_frame:
                    continue
                if frame_index >= stop_frame:
                    done = True
                    break

                output_frame = frame.to_rgb()
                output_frame.pts = None
                encoded_packet = output_stream.encode(output_frame)
                if encoded_packet:
                    output_container.mux(encoded_packet)
                written += 1

                if written == num_frames:
                    done = True
                    break

            if done:
                break

        encoded_packet = output_stream.encode()
        if encoded_packet:
            output_container.mux(encoded_packet)

    if written != num_frames:
        raise RuntimeError(f"Encoded {written} frames from {src}, expected {num_frames}")
    if not dst.exists():
        raise OSError(f"Video encoding did not work. File not found: {dst}.")


def read_lerobot_v3_episode_table(
    data_path: Path,
    episode_index: int,
    feature_specs: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    table = pq.read_table(data_path, columns=[*feature_specs.keys(), "episode_index"])
    episode_indices = arrow_column_to_numpy(table["episode_index"])
    mask = episode_indices == episode_index
    if not np.any(mask):
        raise ValueError(f"Episode {episode_index} not found in {data_path}")

    feature_data = {}
    for key, feature in feature_specs.items():
        array = arrow_column_to_numpy(table[key])[mask]
        feature_data[key] = array.astype(np.dtype(feature["dtype"]), copy=False)
    return feature_data


def produce_lerobot_v3_episode(
    video_map: dict[str, Path],
    episode: LeRobotV3Episode,
    *,
    source_root: Path,
    data_path_template: str,
    video_path_template: str | None,
    feature_specs: dict[str, dict[str, Any]],
    fps: int,
    prompt: str | None,
):
    data_path = source_root / format_lerobot_v3_path(
        data_path_template,
        chunk_index=episode.data_chunk_index,
        file_index=episode.data_file_index,
    )
    feature_data = read_lerobot_v3_episode_table(data_path, episode.episode_index, feature_specs)

    if video_map and video_path_template is None:
        raise ValueError("Video features are present but source metadata has no video_path")

    for video_key, video_dst in video_map.items():
        video_chunk_index, video_file_index, from_timestamp, to_timestamp = episode.video_spans[video_key]
        video_src = source_root / format_lerobot_v3_path(
            video_path_template,
            chunk_index=video_chunk_index,
            file_index=video_file_index,
            video_key=video_key,
        )
        start_frame = int(round(from_timestamp * fps))
        stop_frame = int(round(to_timestamp * fps))
        if stop_frame - start_frame != episode.length:
            print(
                f"  Warning: episode {episode.episode_index} metadata has "
                f"{stop_frame - start_frame} video frames but {episode.length} table rows; using table length."
            )
        encode_video_segment(video_src, video_dst, fps, start_frame, episode.length)

    task = prompt if prompt is not None else (episode.tasks[0] if episode.tasks else "")
    return feature_data, [task] * episode.length


def convert_lerobot_v3_dataset(
    data_dir: Path,
    save_dir: Path | str,
    save_repoid: str,
    prompt: str | None,
    max_workers: int,
    overwrite: bool,
    only_sync: bool,
) -> Path:
    info = load_lerobot_v3_info(data_dir)
    features = resolve_lerobot_v3_features(info)
    video_keys = [key for key, feature in features.items() if feature["dtype"] == "video"]
    table_feature_specs = {key: feature for key, feature in features.items() if feature["dtype"] != "video"}
    episodes = load_lerobot_v3_episodes(data_dir, video_keys)
    output_path = Path(save_dir) / save_repoid

    print(
        f"Detected LeRobot dataset at {data_dir}. "
        f"Converting {len(episodes)} episodes with features: {', '.join(features.keys())}"
    )

    if only_sync:
        return output_path

    if output_path.exists():
        if overwrite:
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(f"Output path {output_path} already exists. Use --overwrite to overwrite.")

    builder = LeRobotDatasetBuilder(
        repo_id=save_repoid,
        fps=int(info["fps"]),
        features=features,
        robot_type=info.get("robot_type"),
        root=output_path,
    )
    builder.add_episodes(
        partial(
            produce_lerobot_v3_episode,
            source_root=data_dir,
            data_path_template=info["data_path"],
            video_path_template=info.get("video_path"),
            feature_specs=table_feature_specs,
            fps=int(info["fps"]),
            prompt=prompt,
        ),
        episodes,
        max_workers=max_workers,
    )
    builder.flush()
    return output_path


def produce_episode(
    video_map: dict[str, Path],
    log_dir: Path,
    prompt: str,
    enabled_optional_features: set[str] | None = None,
):
    episode_start_time = time.time()
    enabled_optional_features = enabled_optional_features or set()
    
    camera_mapping = {
        "top_head": "cam_high",
        "hand_left": "cam_left_wrist", 
        "hand_right": "cam_right_wrist"
    }
    
    try:
        episode, f = lazy_load_hdf5_dataset(log_dir)

        epi_len = episode.pop("epi_len")
        tasks = [prompt] * epi_len

        feature_data = {
            "observation.state": episode["observation.state"],
            "action": episode["action"]
        }

        if "inferred_action" in episode.keys() and "inferred_action" in enabled_optional_features:
            feature_data["inferred_action"] = episode["inferred_action"]
        if "noise" in episode.keys() and "noise" in enabled_optional_features:
            feature_data["noise"] = episode["noise"]
        
        camera_list = [key for key in episode.keys() if key.startswith("observation.images.")]
        
        for camera_key in camera_list:
            camera_start_time = time.time()
            video_dst = video_map[camera_key]
            hdf5_camera_name = camera_key.replace("observation.images.", "")
            
            video_dir_name = camera_mapping.get(hdf5_camera_name, hdf5_camera_name)
            
            data_dir = log_dir.parent
            episode_name = log_dir.stem
            
            existing_video_path = data_dir / "video" / video_dir_name / f"{episode_name}.mp4"
            
            if existing_video_path.exists():
                # copy_start_time = time.time()
                video_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(existing_video_path, video_dst)
                # copy_time = time.time() - copy_start_time
                # copy_count += 1
            else:
                # encode_start_time = time.time()
                images = np.array(episode[camera_key])
                encode_video_frames(images, dst=video_dst, fps=30, overwrite=True)
        
        f.close()
        return feature_data, tasks
        
    except Exception as e:
        error_time = time.time() - episode_start_time
        print(f"  ❌ processing episode {log_dir.name} failed: {e}, Ignoring this episode.")
        print(f"  ⏱️  Error timestamp: {error_time:.2f}s")
        # raise

def main(
    data_dir: Path | str,
    save_dir: Path | str,
    repo_ids: List[str] | str | None = None,
    prompt: str | None = None,
    save_repoid: str | None = None,
    max_workers: int = 8,
    *,
    overwrite: bool = False,
    upload: bool = False,
    only_sync: bool = False,
):
    
    data_dir = Path(data_dir)
    if repo_ids is None:
        repo_ids = []
    elif type(repo_ids) is str:
        repo_ids = [repo_ids]
    
    task = data_dir.name.split('_')[0]
    
    if save_repoid is None:
        repoid = data_dir.name.split('_')
        # task = repoid[0]
        save_repoid = '_'.join(repoid[1: -1]) + '_lerobot'
        print(f"save_repoid will be set according to repo_ids: {save_repoid}")

    if is_lerobot_v3_dataset(data_dir):
        if repo_ids:
            print("Ignoring --repo-ids because --data-dir already points to a LeRobot dataset root.")
        output_path = convert_lerobot_v3_dataset(
            data_dir=data_dir,
            save_dir=save_dir,
            save_repoid=save_repoid,
            prompt=prompt,
            max_workers=max_workers,
            overwrite=overwrite,
            only_sync=only_sync,
        )
        if upload:
            raise NotImplementedError("--upload is only implemented for the legacy HDF5 cloth dataset path.")
        return

    if not repo_ids:
        raise ValueError("--repo-ids is required when converting legacy HDF5 directories.")

    log_files: List[Path] = []
    for repo_id in repo_ids:
        repo_path = data_dir / repo_id
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository path {repo_path} does not exist.")
        found_files = sorted(d for d in repo_path.iterdir() if not d.is_dir() and d.suffix == '.hdf5')
        log_files.extend(found_files)
    # filter invalid hdf5 files
    valid_files = []
    for file in log_files:
        data_dir = file.parent
        episode_name = file.stem
        all_videos_exist = True
        for video_dir_name in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
            existing_video_path = data_dir / "video" / video_dir_name / f"{episode_name}.mp4"
            if not existing_video_path.exists():
                print(f"  ⚠️  {existing_video_path} not found, skipping this file.")
                all_videos_exist = False
                break
        if not all_videos_exist:
            continue
        try:
            with h5py.File(file, 'r') as f:
                pass
            valid_files.append(file)
        except Exception as e:
            print(f"  ❌ Invalid {file}, error: {e}, Ignoring this file.")

    # output_path = Path(save_dir) / task / save_repoid
    output_path = Path(save_dir) / save_repoid


    if not only_sync:
        if output_path.exists():
            if overwrite:
                shutil.rmtree(output_path)
            else:
                raise FileExistsError(f"Output path {output_path} already exists. Use --overwrite to overwrite.")
        
        resolved_features, enabled_optional_features = resolve_output_features(valid_files)
        disabled_optional_features = sorted(set(OPTIONAL_FEATURES) - enabled_optional_features)
        if disabled_optional_features:
            print(
                "Excluding optional features not present in every valid episode: "
                + ", ".join(disabled_optional_features)
            )

        builder = LeRobotDatasetBuilder(
            repo_id=save_repoid,
            fps=30,
            features=resolved_features,
            robot_type='agilex',
            root=output_path,
        )
        if prompt is None:
            if task == 'iros':
                prompt = "fold the cloth"
            else:
                prompt = f"{task} the cloth"
        builder.add_episodes(
            partial(produce_episode, prompt=prompt, enabled_optional_features=enabled_optional_features),
            valid_files,
            max_workers=max_workers,
        )
        
        builder.flush()

    if upload:
        # begin upload data
        assert task in ['hang', 'fold', 'flat', 'iros'], f"Unknown task: {task}, expected one of ['hang', 'fold', 'flat', 'iros']"
        print(f"  ⏫ Starting upload...")
        if task == 'iros':
            remote_path = f"oss://oss-pai-d8dbg42zb0rplbe70e-cn-wulanchabu/data/fold_cloth/{task}/{save_repoid}"
        else:
            remote_path = f"oss://oss-pai-d8dbg42zb0rplbe70e-cn-wulanchabu/data/{task}_cloth/{save_repoid}"
        cmd = [
            "ossutil", "cp",
            "-r",                                
            str(output_path) + "/",
            remote_path,                         
            "-j", "100",                         
            "-u",                                             
        ]
        try:
            result = subprocess.run(cmd, text=True, check=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"{e.stderr}")

def add_episode(

):
    pass

if __name__ == "__main__":
    st = time.time()
    tyro.cli(main)
    print(f"Time taken: {time.time() - st} seconds")
