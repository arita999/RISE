# Inference Deployment on Piper

Real-time inference deployment for OpenPi policies on Agilex Piper dual-arm robots.

---

## Prerequisites

### System Requirements

- **OS**: Ubuntu 20.04
- **ROS**: ROS Noetic

### Hardware Setup

**1. Piper Dual Arms**

Configure master + slave arms following the [Piper Arm Setup Guide](../deploy/Piper_ros_private-ros-noetic/README.md).

**2. RealSense Cameras**

Install ROS wrapper:
```bash
sudo apt-get install ros-noetic-realsense2-camera
```

Update camera serials in `deploy/multi_camera.launch`:
```xml
<arg name="serial_no_camera1" default="YOUR_CAMERA_1_SERIAL"/>
<arg name="serial_no_camera2" default="YOUR_CAMERA_2_SERIAL"/>
<arg name="serial_no_camera3" default="YOUR_CAMERA_3_SERIAL"/>
```

Deploy and test:
```bash
cp deploy/multi_camera.launch /opt/ros/noetic/share/realsense2_camera/launch/
roslaunch realsense2_camera multi_camera.launch
```

---

## Installation

**1. Create Environment**
```bash
conda create -n deploy python=3.10 -y
conda activate deploy
```

**2. Install Dependencies**
```bash
cd deploy
pip install -r requirements.txt
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```
> For other CUDA versions, see [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

**3. Install openpi-client**
```bash
cd deploy/packages/openpi-client
pip install -e .
```

---

## Running Inference

Start the [openpi policy server](https://github.com/Physical-Intelligence/openpi) on your GPU machine first.

### green_tag Policy Server

For the `Pi05_style_training` run trained on the `green_tag` data, start the
latest saved checkpoint with:

```bash
conda activate rise
./deploy/serve_green_tag_policy.sh
```

If you want a specific checkpoint, pass it explicitly:

```bash
./deploy/serve_green_tag_policy.sh \
  policy_and_value/policy_offline_and_value/checkpoints/Pi05_style_training/Pi05_style_training/10000
```

### Star AI Robot Arm

The `green_tag` policy was trained with these camera mappings:

- `observation.images.side` -> `top_head`
- `observation.images.rear` -> `hand_left`
- `observation.images.onhand` -> `hand_right`

Run the generic Star AI Robot Arm ROS bridge after the policy server is up:

```bash
conda activate deploy
./deploy/run_star_ai_green_tag.sh \
  --host 172.16.99.11 \
  --port 8000 \
  --side_image_topic /camera_side/color/image_raw \
  --rear_image_topic /camera_rear/color/image_raw \
  --onhand_image_topic /camera_onhand/color/image_raw \
  --joint_state_topic /joint_states \
  --joint_cmd_topic /joint_command \
  --joint_names joint0,joint1,joint2,joint3,joint4,joint5,joint6
```

Use `--dry_run` first to check camera, joint-state, and policy-server wiring
without publishing robot commands. The script sends the Star arm state to the
policy and publishes action dimensions `[0:7]` by default; use `--action_start`
and `--arm_dof` if your Star command layout is different.

For the Seeed StarAI Arm setup that does not use ROS, create the uv environment
from the repository root and connect directly to the UC-01 serial port:

```bash
uv venv .venv-star --python 3.11
source .venv-star/bin/activate
uv pip install -r requirements.txt

./deploy/run_star_ai_green_tag_no_ros.sh \
  --host 172.16.99.11 \
  --port 8000 \
  --side_source 0 \
  --rear_source 1 \
  --onhand_source 2 \
  --starai_port /dev/ttyUSB1 \
  --dry_run \
  --run_once \
  --no_confirm
```

After dry-run succeeds, remove `--dry_run` and `--run_once` to publish actions
directly to the StarAI follower arm.

### Piper Dual Arms

**Launch in 3 terminals:**

```bash
# Terminal 1: Cameras
roslaunch realsense2_camera multi_camera.launch

# Terminal 2: Robot Arms
bash deploy/Piper_ros_private-ros-noetic/can_config.sh

roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true

# Terminal 3: Inference
conda activate deploy
python deploy/piper_deploy.py \
  --host 172.16.99.11 \
  --port 8000 \
  --ctrl_type joint \
  --use_temporal_smoothing \
  --chunk_size 50 \
  --lang_embeddings "Pick and sort bricks on the conveyor."
```

---

## Data Collection

Both scripts save data in HDF5 format with synchronized images, joint states, and actions.

**Output Structure:**
```
dataset_dir/
├── task_name/
│   ├── episode_0.hdf5
│   ├── episode_1.hdf5
│   └── video/
│       ├── cam_high/
│       ├── cam_left_wrist/
│       └── cam_right_wrist/
└── aloha_mobile/           # for inference data
    ├── aloha_mobile_success/
    └── aloha_mobile_fail/
```

### Teleoperation Collection

Manual control via master-slave setup.

```bash
# Terminal 1: Cameras
roslaunch realsense2_camera multi_camera.launch

# Terminal 2: Arms (master-slave mode)
bash deploy/Piper_ros_private-ros-noetic/can_config.sh

roslaunch piper start_ms_piper.launch mode:=0 auto_enable:=false

# Terminal 3: Collection
conda activate deploy
cd deploy/data_collection
python collect_data.py \
  --max_timesteps 5000 \
  --export_video \
  --dataset_dir ~/data/my_task \
  --episode_idx 0
```

**Controls:** Press `Space` to stop and save early.

### Inference Collection

Record autonomous policy execution. Requires running policy server.

```bash
# Terminal 1: Cameras
roslaunch realsense2_camera multi_camera.launch

# Terminal 2: Arms (inference mode)
bash deploy/Piper_ros_private-ros-noetic/can_config.sh

roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true

# Terminal 3: Collection
conda activate deploy
cd deploy/data_collection
python collect_inference_data.py \
  --use_temporal_smoothing \
  --ctrl_type joint \
  --export_video \
  --chunk_size 50 \
  --host 172.16.99.11 \
  --port 8000 \
  --dataset_dir ~/data/rl_task
```

**Workflow:**
1. Press `Enter` to start recording
2. Press `s` to stop robot
3. Label episode: `1` (success) or `0` (failure)
