from typing import Optional, Tuple, Dict, Hashable, Any, Callable, Union


import pathlib
import numpy as np
import pandas as pd


def num_episodes(dataset: pd.DataFrame) -> int:
    """Return number of episodes."""
    return len(np.unique(dataset["episode"].values))


def get_episode(
    dataset: pd.DataFrame,
    episode: Optional[int] = None,
    random: bool = False,
    use_index: bool = False,
) -> pd.DataFrame:
    """Return episode in dataset."""
    assert (episode is not None) ^ random, "Either episode or random."
    episode = (
        np.random.randint(low=0, high=num_episodes(dataset)) if random else episode
    )
    if random or use_index:
        episode = np.sort(np.unique(dataset["episode"].values))[episode]
    episode_data = dataset[dataset["episode"] == episode].copy()
    episode_data.reset_index(drop=True, inplace=True)
    assert np.all(
        np.diff(episode_data["timestep"].values) > 0
    ), "Episode not in increasing timesteps."

    return episode_data


def get_timestep_data(
    file_path: str,
) -> Tuple[int, str]:
    """Return timestep."""
    timestamp = None
    for s in pathlib.Path(file_path).stem.split("_"):
        if s[0] == "t" and s[1:].isnumeric():
            timestamp = s
            break
    assert timestamp is not None
    assert timestamp[0] == "t"
    return int(timestamp[1:]), timestamp


def get_episode_data(
    file_path: str,
) -> Tuple[int, str]:
    """Return episode."""
    epstamp = None
    for s in pathlib.Path(file_path).stem.split("_"):
        if s[:2] == "ep" and s[2:].isnumeric():
            epstamp = s
            break
    assert epstamp is not None
    return int(epstamp[2:]), epstamp


def get_sample_data(
    dataset: pd.DataFrame,
    episode: Optional[int] = None,
    epstamp: Optional[str] = None,
    timestep: Optional[int] = None,
    timestamp: Optional[str] = None,
) -> Dict[Hashable, Any]:
    """Return sample from dataset."""
    assert episode is not None or epstamp is not None, "Require episode for lookup."
    assert timestep is not None or timestamp is not None, "Require timestep for lookup."
    ep_key, ep_val = (
        ("episode", episode) if episode is not None else ("epstamp", epstamp)
    )
    t_key, t_val = (
        ("timestep", timestep) if timestep is not None else ("timestamp", timestamp)
    )
    row = dataset.loc[(dataset[ep_key] == ep_val) & (dataset[t_key] == t_val)]
    sample = row.to_dict(orient="records")[0]
    return sample


def aggr_episode_key_data(
    dataset: pd.DataFrame,
    key: str,
    aggr_fn: Callable[[np.ndarray], Union[float, np.ndarray]] = np.max,
    return_list: bool = False,
) -> np.ndarray:
    """Aggregate values over episodes."""
    data = []
    for i in range(num_episodes(dataset)):
        episode = get_episode(dataset, episode=i, use_index=True)
        data.append(aggr_fn(episode[key].values))

    if return_list:
        return data

    if isinstance(data[0], np.ndarray):
        data = np.concatenate(data)
    else:
        data = np.array(data)

    return data
