# Physical AI Interpretability

[![PyPI version](https://badge.fury.io/py/physical-ai-interpretability.svg)](https://badge.fury.io/py/physical-ai-interpretability)
[![Documentation Status](https://github.com/villekuosmanen/physical-AI-interpretability/actions/workflows/docs.yml/badge.svg)](https://villekuosmanen.github.io/physical-AI-interpretability/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/villekuosmanen/physical-AI-interpretability.svg?style=social&label=Star)](https://github.com/villekuosmanen/physical-AI-interpretability)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Interpretability tools for transformer-based Physical AI and robotics models. Built on [LeRobot](https://github.com/huggingface/lerobot).

## Installation

```
pip install physical-ai-interpretability
```

## Attention maps

![Visualised attention maps for a robot picking up coffee capsules](https://github.com/villekuosmanen/physical-ai-interpretability/blob/main/assets/attention_coffee_prop.gif)


### Analyse existing dataset

Easiest way to use the attention mapper is to run a post-hoc attention analysis of an existing dataset. In this case, we run our pre-trained policy on episodes in the dataset and capture the attention maps. This requires no connection to any robots and should work out of the box.

```
python examples/visualise_original_data_attention.py --dataset-repo-id lerobot/svla_so101_pickplace --episode-id 29 --policy-path <path to your pre-trained policy> --output-dir ./output/attention_analysis_results
```

Pre-trained policy part may look something like this: `../lerobot/outputs/train/act_johns_arm/checkpoints/last/pretrained_model`

If you get an error with `ModuleNotFoundError: No module named 'physical_ai_interpretability'`, set the `PYTHONPATH` environment variable to the location of `physical-ai-interpretability` in your local directory, e.g.  
`PYTHONPATH=/home/ville/physical-ai-interpretability:$PYTHONPATH`.

### Use at test-time

Use the `ACTPolicyWithAttention` plugin in your project either by importing it from here or just copying the `physical_ai_interpretability/attention_maps/act_attention_mapper.py` file over.

See `examples/usage_with_act.py` for use of the attention mapper with the default LeRobot ACT policy.

I would like to add support for Pi0, SmolVLA, and other foundation models at some point! 

## Feature Extraction

Method of applying [Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features) as proposed by Anthropic (and implemented in the famous [Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude) language model) into robotics and physical AI.

[Demo in Hugging Face Spaces](https://huggingface.co/spaces/villekuosmanen/act-feature-visualiser).

To reproduce this demo, you will need to repeat the following steps:

0. Train a small ACT model for your task by changing the `dim_model` hyperparam value to `124` in LeRobot. You will likely need to increase your batch size and number of training steps as smaller models take longer to converge during training. (required if using RTX 4090 or other consumer GPUs)
1. Train a sparse autoencoder using `scripts/train_sae.py`
    - The system will automatically infer `num_tokens` from your ACT model configuration and structure. The total number of tokens in ACT models is calculated as `total_tokens = 2 + Σ(wᵢ/32 × hᵢ/32)` where 2 tokens are reserved for the VAE token and joint states, plus tokens from each camera (by dividing width and height by 32 due to ResNet patch size). For example, 2 images with dims (480 × 640) generates `2 + 2*((480/32) * (640/32)) = 602` tokens. With token sampling (default ratio 8), we get `num_tokens = 600 / 8 = 75`. The automatic inference tries multiple methods: model config inspection, positional embedding shapes, and fallback data-based inference.
    - Note that the training script is currently tested on my LeRobot fork and won't work with the most recent API updates to LeRobot. It should be easy to hack to get it working and if you do please send a PR!
2. Record feature activations using `scripts/record_feature_activations.py`. This generates Parquet files showing what features were active at each frame of your dataset - we will use them during manual analysis.
3. Analyse feature activations using the `scripts/analyse_features.ipynb` notebook. It should construct `.json` files describing the rop activating frames for features with the most variance in them.
4. Move or link the `scripts/feature_analysis_results` into the `examples/features_huggingface_space` directory, then run `examples/features_huggingface_space/ui.py`. It will open in Gradio and allow you to visualise and name individual features. You can even deploy the results into Hugging Face spaces using `gradio deploy` to share what you found with the world! (change the save button to non-interactive if you don't want other people editing your features!)

### Out of distribution detection

Check out [the blog post for how to build an out-of-distribution detector using the Physical AI Interpretability toolkit](https://villekuosmanen.medium.com/building-a-simple-out-of-distribution-detector-for-physical-ai-models-using-lerobot-bfa02b4a3876).

The SAE trained for feature extraction also provides a neat implementation for out of distribution detection in robotics. The SAETrainer and OODDetector classes implement this.

`scripts/demo_ood_detector.py` shows how to test the OOD Detector with a pre-trained SAE model. The call looks something like this:

```
python scripts/demo_ood_detector.py \
    --validation-repo-id villekuosmanen/dAgger_drop_footbag_into_dice_tower_1.7.0 \
    --test-repo-id villekuosmanen/eval_dAgger_drop_footbag_into_dice_tower_1.7.0 \
    --policy-path ../lerobot/outputs/train/reliable_robot_1.7.0_small_main/checkpoints/last/pretrained_model \
    --sae-experiment-path output/sae_drop_footbag_into_di_838a8c8b
```

You can also upload the saved SAE model to the Hugging Face Hub by setting the `--upload-to-hub` and `--hub-repo-id <model_repo_id>` params.

## Other cool stuff

[Pikodata](https://github.com/villekuosmanen/pikodata) is a Data Studio designed for LeRobot Datasets, offering a UI for deleting episodes and frames, as well as editing language descriptions for LeRobot Datasets.

[RewACT](https://github.com/villekuosmanen/rewACT) is a simple reward model built of ACT policies, used to measure the current task progress.

If you find my open-source Robotics and Physical AI work valuable, consider [sponsoring me on GitHub](https://github.com/sponsors/villekuosmanen)!
