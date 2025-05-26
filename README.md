# Physical AI Attention Mapper
Attention mappers and visualisation for transformer-based Physical AI policies.

![Visualised attention maps for a robot picking up coffee capsules](https://github.com/villekuosmanen/physical-AI-attention-mapper/blob/main/assets/attention_coffee_prop.gif)

This project is more of an experiment rather than complete a library with a stable API so do keep that in mind.

## Getting started

Use the `ACTPolicyWithAttention` plugin in your project either by importing it from here or just copying the `src/act_attention_mapper.py` file over.

## ACT

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
            _ self.model(batch)
        # return actions as before
        return self._action_queue.popleft()
```

If you do not want to modify LeRobot's ACT policy source code, you should delete the `force_model_run` param inside `src/act_attention_mapper.py`'s `policy.select_action()` call.

## Future policies

I would like to add support for Pi0 and other VLA models at some point! 
