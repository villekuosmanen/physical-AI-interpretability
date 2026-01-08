import time

import numpy as np

from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.act.modeling_act import ACTPolicy

from examples.encode_video import encode_video_ffmpeg
from physical_ai_interpretability.attention_maps import ACTPolicyWithAttention

# ...
# Assuming use in existing LeRobot code where we already have a policy config and dataset
cfg = {}
dataset = {}
robot = {}
fps = 20 # set to your robot control FPS

policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
if isinstance(policy, ACTPolicy):
    # We simply wrap the existing ACT policy with our ACTPolicyWithAttention plugin
    policy = ACTPolicyWithAttention(policy)

# We can optionally initialise in-mem buffers for saving the visualised attention maps into videos
video_buffers = None
side_by_side_buffer = []
num_cameras_detected = 0
episode_start_timestamp = time.strftime("%Y%m%d-%H%M%S")

# ...
# During inference, we add a bit of code to handle the attention maps without breaking other policy types
observation = robot.capture_observation()
res = policy.select_action(observation)
if isinstance(policy, ACTPolicyWithAttention):
    # Decompose res to action and attention maps
    (action, attention_maps) = res

    # Visualise attention maps
    visualizations = policy.visualize_attention(attention_maps=attention_maps, observation=observation)

    # OPTIONAL - save visualisation into video (we can just show it on screen with opencv as well)
    if video_buffers is None and visualizations:
        num_cameras_detected = len(visualizations)
        video_buffers = [[] for _ in range(num_cameras_detected)]
        print(f"Detected {num_cameras_detected} camera views for video buffering.")

    if video_buffers is not None: # Ensure buffers are initialized
        valid_frames_this_step = []
        for i, vis in enumerate(visualizations):
            if vis is not None:
                if i < len(video_buffers): # Check bounds
                    video_buffers[i].append(vis.copy()) # Append a copy
                    valid_frames_this_step.append(vis.copy())
            else:
                valid_frames_this_step.append(None) # Keep placeholder for side-by-side alignment

        # Create side-by-side frame if all frames for this step are available and heights match
        if len(valid_frames_this_step) == num_cameras_detected and all(f is not None for f in valid_frames_this_step):
            # Ensure all frames have the same height for hstack
            first_height = valid_frames_this_step[0].shape[0]
            if all(f.shape[0] == first_height for f in valid_frames_this_step):
                side_by_side_frame = np.hstack(valid_frames_this_step)
                side_by_side_buffer.append(side_by_side_frame)
            else:
                print(f"Warning: Mismatched heights for side-by-side view, skipping combined frame.")
else:
    # Continue supporting existing 
    action = res

# Send action to robot as usual
robot.send_action(action)

# At the end, optionally encode videos
if video_buffers is not None:
    for i, cam_buffer in enumerate(video_buffers):
        if cam_buffer: # Check if buffer is not empty
            output_filename = f"attention_cam_{i}_{episode_start_timestamp}.mp4"
            encode_video_ffmpeg(cam_buffer, output_filename, fps)
        else:
            print(f"Buffer for camera {i} is empty. No video will be generated.")
    
    # OPTIONAL - Clear buffers after encoding if you might run another episode in the same script execution
    for buffer in video_buffers:
        buffer.clear()

if side_by_side_buffer:
    output_filename_sbs = f"attention_side_by_side_{episode_start_timestamp}.mp4"
    encode_video_ffmpeg(side_by_side_buffer, output_filename_sbs, fps)
    
    # OPTIONAL - Clear side by side buffer if reused for another episode 
    side_by_side_buffer.clear()
else:
    print("Side-by-side buffer is empty. No combined video will be generated.")
