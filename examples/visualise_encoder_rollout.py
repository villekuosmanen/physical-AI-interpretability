#!/usr/bin/env python
"""Encoder attention rollout example.

For an ACT checkpoint and a sequence of (image, state) pairs on disk, compute
the encoder rollout for each frame, write per-frame overlays + a GIF, and dump
per-frame proprio/latent/vision share to JSON.

Why this is interesting: standard decoder cross-attention can show the decoder
reading 100% from the proprio token, even on a vision-driven task. Encoder
rollout reveals that the proprio token has *itself* absorbed vision content
through encoder self-attention. See `act_encoder_rollout.py` for details.

Usage:
    python visualise_encoder_rollout.py \\
        --ckpt path/to/ckpt \\
        --frames-dir frames/ \\
        --states states.json \\
        --output rollout_out/ \\
        --label left_pass

`frames-dir` should contain `*.jpg` frames sorted lexicographically; `states`
should be a JSON list of state vectors (one per frame).
"""
import argparse
import json
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from physical_ai_interpretability.attention_maps import ACTPolicyWithEncoderRollout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--states", required=True, help="JSON list of state vectors, one per frame")
    ap.add_argument("--output", required=True)
    ap.add_argument("--label", default="rollout")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument(
        "--image-key",
        default="observation.images.front",
        help="ACT input key for the camera image",
    )
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    policy = ACTPolicy.from_pretrained(args.ckpt)
    policy.eval().to(device)
    policy.config.device = device
    pre, _post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": device}},
    )

    wrapper = ACTPolicyWithEncoderRollout(policy, pre)
    H_feat, W_feat = wrapper.get_feature_grid_shape(device=device)
    print(
        f"encoder layers: {wrapper.num_encoder_layers}, "
        f"feature grid: ({H_feat}, {W_feat}) = {H_feat * W_feat} vision tokens"
    )

    states = json.loads(Path(args.states).read_text())
    frame_paths = sorted(Path(args.frames_dir).glob("*.jpg"))
    print(f"frames: {len(frame_paths)}, states: {len(states)}")

    viz_frames = []
    shares = []
    for i, fp in enumerate(frame_paths):
        img = cv2.imread(str(fp))
        H, W = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img_rgb).float() / 255.0
        img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(device)
        state = np.array(states[i], dtype=np.float32)
        state_t = torch.from_numpy(state).unsqueeze(0).to(device)
        obs = {args.image_key: img_t, "observation.state": state_t}

        result = wrapper.rollout_for_observation(obs)

        viz = wrapper.overlay_on_image(result.spatial_heatmap, img, alpha=args.alpha)
        h, w = viz.shape[:2]
        cv2.rectangle(viz, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.putText(viz, args.label, (10, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(viz, f"frame {i:02d}/{len(frame_paths) - 1}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(
            viz,
            f"vision_share={result.vision_share:.2f}  prop={result.proprio_share:.2f}",
            (220, h - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        viz_frames.append(viz)
        cv2.imwrite(str(out_dir / f"f{i:02d}.png"), viz)
        shares.append({"frame": i, **result.as_dict()})

    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in viz_frames]
    gif_path = out_dir / f"{args.label}_rollout.gif"
    imageio.mimsave(str(gif_path), rgb, duration=0.15, loop=0)
    print(f"wrote {gif_path}")
    (out_dir / "rollout_shares.json").write_text(json.dumps(shares, indent=2))
    print("wrote rollout_shares.json")


if __name__ == "__main__":
    main()
