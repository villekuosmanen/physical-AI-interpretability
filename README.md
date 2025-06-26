# Physical AI Interpretability

Interpretability tools for transformer-based Physical AI and robotics models.

## Attention maps

![Visualised attention maps for a robot picking up coffee capsules](https://github.com/villekuosmanen/physical-AI-attention-mapper/blob/main/assets/attention_coffee_prop.gif)

This project is more of an experiment rather than complete a library with a stable API so do keep that in mind.

### Analyse existing dataset

Easiest way to use the attention mapper is to run a post-hoc attention analysis of an existing dataset. In this case, we run our pre-trained policy on episodes in the dataset and capture the attention maps. This requires no connection to any robots and should work out of the box.

```
python examples/visualise_original_data_attention.py --dataset-repo-id lerobot/svla_so101_pickplace --episode-id 29 --policy-path <path to your pre-trained policy> --output-dir ./output/attention_analysis_results
```

Pre-trained policy part may look something like this: `../lerobot/outputs/train/act_johns_arm/checkpoints/last/pretrained_model`

If you get an error with `ModuleNotFoundError: No module named 'src'`, set the `PYTHONPATH` environment variable to the location of `physical-AI-attention-mapper` in your local directory, e.g.  
`PYTHONPATH=/home/ville/physical-AI-attention-mapper:$PYTHONPATH`.

### Use at test-time

Use the `ACTPolicyWithAttention` plugin in your project either by importing it from here or just copying the `src/act_attention_mapper.py` file over.

#### ACT

See `examples/usage_with_act.py` for use of the attention mapper with the default LeRobot ACT policy.

Note that LeRobot's ACT policy utilises a queue mechanism for the action chunks, meaning the policy itself is only inferenced once per `n_action_steps`. This means we won't get a smooth video of visualised attentions at every time step.

We can work around this by modifying the LeRobot ACT policy slightly - we just force the model to run at every time step, throwing away the outputs. This may increase latency slightly at high FPS or non-CUDA devices. Because the outputs are unused it won't impact the performance of the ACT model itself and.

```python
# github.com/huggingface/lerobot
# lerobot/common/policies/act/modeling_act.py

class ACTPolicy(PreTrainedPolicy):
    # everything else unchanged...

    @torch.no_grad
    def select_action(self,batch: dict[str, Tensor], force_model_run: bool = False)
        # we have added a new param `force_model_run`

        # everything else the same
        # ...

        if len(self._action_queue) == 0:
            # everything as before
            # ...
        # NEW: add this elif block
        elif force_model_run:
            # predict and throw away the results
            # this simply allows our attention mapper to capture the attention values during the inference run
            _ = self.model(batch)
        # return actions as before
        return self._action_queue.popleft()
```

If you do not want to modify LeRobot's ACT policy source code, you should delete the `force_model_run` param inside `src/act_attention_mapper.py`'s `policy.select_action()` call.

#### Future policies

I would like to add support for Pi0 and other VLA models at some point! 

## Feature Extration

Method of applying [Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features) as proposed by Anthropic (and implemented in the famous [Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude) language model) into robotics and physical AI.

[Demo in Hugging Face Spaces](https://huggingface.co/spaces/villekuosmanen/act-feature-visualiser).

To reproduce this demo, you will need to repeat the following steps:

0. Train a small ACT model for your task by changing the `dim_model` hyperparam value to `124` in LeRobot. You will likely need to increase your batch size and number of training steps as smaller models take longer to converge during training. (required if using RTX 4090 or other consumer GPUs)
1. Train a sparse autoencoder using `scripts/train_sae.py`
    - Your `num_tokens` value should equal to `tokens_for_sampling / sample_ratio`, where `sample_ratio` defaults to 8. You can check total_tokens in ACT models by debugging the model during training time, but I believe it is `total_tokens = 2 + Σ(wᵢ/32 × hᵢ/32)`. 2 tokens are reeserved for the VAE token and joint states, to which you add tokens from each camera, which you get by first dividing both width and height by 32 then multiplying it (resnets construct embeddings from patches of 32). For example, 2 images with dims (480 * 640) generates `2 + 2*((480 /32) * (640 / 32)) = 602`. With 2 fixed tokens, we get `num_tokens = 600 / 8 = 75`
    - Note that the training script is currently tested on my LeRobot fork and won't work with the most recent API updates to LeRobot. It should be easy to hack to get it working and if you do please send a PR!
2. Record feature activations using `scripts/record_feature_activations.py`. This generates Parquet files showing what features were active at each frame of your dataset - we will use them during manual analysis.
3. Analyse feature activations using the `scripts/analyse_features.ipynb` notebook. It should construct `.json` files describing the rop activating frames for features with the most variance in them.
4. Move or link the `scripts/feature_analysis_results` into the `examples/features_huggingface_space` directory, then run `examples/features_huggingface_space/ui.py`. It will open in Gradio and allow you to visualise and name individual features. You can even deploy the results into Hugging Face spaces using `gradio deploy` to share what you found with the world! (change the save button to non-interactive if you don't want other people editing your features!)

## Other cool stuff

[Pikodata](https://github.com/villekuosmanen/pikodata) is a Data Studio designed for LeRobot Datasets, offering a UI for deleting episodes and frames, as well as editing language descriptions for LeRobot Datasets.

If you find my open-source Robotics and Physical AI work valuable, consider [sponsoring me on GitHub](https://github.com/sponsors/villekuosmanen)!
