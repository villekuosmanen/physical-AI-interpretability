from typing import List

import torch
import logging
from pprint import pformat

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import (
    LeRobotDatasetMetadata,
    LeRobotDataset,
    MultiLeRobotDataset,
)
from lerobot.datasets.factory import IMAGENET_STATS

def make_dataset_without_config(
    repo_id: str,
    action_delta_indices: List,
    observation_delta_indices: List = None,
    root: str = None,
    video_backend: str = "pyav",
    episodes: list[int] | None = None,
    revision: str | None = None,
    use_imagenet_stats: bool = True,
) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    if repo_id.startswith('['):
        datasets = repo_id.strip('[]').split(',')
        datasets = [x.strip() for x in datasets]
        delta_timestamps = {}
        for ds in datasets:
            ds_meta = LeRobotDatasetMetadata(
                ds,
                root=HF_LEROBOT_HOME / ds,
            )
            d_ts = resolve_delta_timestamps_without_config(ds_meta, action_delta_indices, observation_delta_indices)
            delta_timestamps[ds] = d_ts
        dataset = MultiLeRobotDataset(
            datasets,
            delta_timestamps=delta_timestamps,
            video_backend=video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index , indent=2)}"
        )
    else:
        ds_meta = LeRobotDatasetMetadata(repo_id)
        delta_timestamps = resolve_delta_timestamps_without_config(ds_meta, action_delta_indices, observation_delta_indices)
        dataset = LeRobotDataset(
            repo_id,
            root=root,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            revision=revision,
            video_backend=video_backend,
        )

    if use_imagenet_stats:
        if isinstance(dataset, MultiLeRobotDataset):
            for ds in dataset._datasets:
                for key in ds.meta.camera_keys:
                    for stats_type, stats in IMAGENET_STATS.items():
                        ds.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
        else:
            for key in dataset.meta.camera_keys:
                for stats_type, stats in IMAGENET_STATS.items():
                    dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset

def resolve_delta_timestamps_without_config(
    ds_meta: LeRobotDatasetMetadata, action_delta_indices: List, observation_delta_indices: List = None
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == "action" and action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in action_delta_indices]
        if key.startswith("observation.state") and observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps

