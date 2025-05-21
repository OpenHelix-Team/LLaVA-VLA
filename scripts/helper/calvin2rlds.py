import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import shortuuid
import multiprocessing
from functools import partial


def episode_process(episode: tuple, data_save: Path):
    ann, task, index_range = episode[0], episode[1], episode[2]
    start = index_range[0]
    end = index_range[1]
    episode = []

    for idx in range(start, end + 1):
        curr_data = f"episode_{idx:07d}.npz"
        curr_data = np.load(os.path.join(data_path, curr_data))

        image = curr_data["rgb_static"]
        wrist_image = curr_data["rgb_gripper"]
        state = curr_data["robot_obs"].astype(np.float32)
        action = curr_data["rel_actions"].astype(np.float32)
        language_instruction = ann

        curr_data_dict = {
            "image": image,
            "wrist_image": wrist_image,
            "state": state,
            "action": action,
            "language_instruction": language_instruction,
        }
        episode.append(curr_data_dict)

    episode = np.array(episode)
    uuid = shortuuid.ShortUUID().random(length=7)
    np.save(os.path.join(data_save, f"episode_{uuid}.npy"), episode)
    return


if __name__ == "__main__":
    target_path = (
        "/wangdonglin/calvin/task_ABCD_D/training/lang_annotations/auto_lang_ann.npy"
    )
    data_path = "/wangdonglin/calvin/task_ABCD_D/training/"
    data_save = "/storage/zhaowei/data/calvin_abcd2d_rlds_episode/"

    ann_data = np.load(target_path, allow_pickle=True).item()
    lang_ann = ann_data["language"]["ann"]
    lang_task = ann_data["language"]["task"]
    lang_index = ann_data["info"]["indx"]
    partial_episode_process = partial(episode_process, data_save=data_save)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(partial_episode_process, zip(lang_ann, lang_task, lang_index))

    print("Finished converting Calvin to RLDS")
