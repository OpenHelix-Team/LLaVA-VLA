import os
import json
import argparse
from pathlib import Path
import numpy as np
import multiprocessing
from tqdm import tqdm
from PIL import Image
from functools import partial
import shortuuid
import random


TARGET_IMG_SIZE = 334


def get_llm_data(
    instruction: str,
    task: str,
    split: str,
    sample: str,
    next_actions: list,
    robot_obs: np.array,
):
    flattened_actions = [action.flatten() for action in next_actions]
    flattened_actions = np.hstack(flattened_actions)
    actions_string = " ".join(map(str, flattened_actions))
    flattened_robot_obs = robot_obs.flatten()
    robot_obs_string = " ".join(map(str, flattened_robot_obs))

    llm_item = {
        "id": Path(sample).stem,
        "image": str(Path(split) / sample),
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n" + instruction + "\n" + robot_obs_string,
            },
            {"from": "gpt", "value": actions_string},
        ],
        "embody": True,
    }

    return llm_item


def get_llm_data_goal_img(
    split: str,
    sample: str,
    goal: str,
    next_actions: list,
    robot_obs: np.array,
):
    flattened_actions = [action.flatten() for action in next_actions]
    flattened_actions = np.hstack(flattened_actions)
    actions_string = " ".join(map(str, flattened_actions))
    flattened_robot_obs = robot_obs.flatten()
    robot_obs_string = " ".join(map(str, flattened_robot_obs))

    llm_item = {
        "id": Path(sample).stem,
        "image": str(Path(split) / sample),
        "goal": str(Path(split) / goal),
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n" + "<goal>\n" + robot_obs_string,
            },
            {"from": "gpt", "value": actions_string},
        ],
        "embody": True,
    }

    return llm_item


def process_episode(episode: tuple, data_path: Path, split: str, future_k: int = 5):
    llm_data_list = []
    ann, task, index_range = episode[0], episode[1], episode[2]

    for step in tqdm(range(index_range[0], index_range[1] + 1)):
        next_actions = []
        step_data = "episode_" + str(step).zfill(7) + ".npz"
        step_data = data_path / split / step_data
        assert step_data.exists(), "Invalid data path"

        for delta in range(future_k):
            future_step = step + delta
            future_data = "episode_" + str(future_step).zfill(7) + ".npz"
            future_data = data_path / split / future_data
            if future_step <= index_range[1]:
                assert future_data.exists(), "Invalid data path"
                actions = np.load(future_data)["rel_actions"]
            else:
                break
            next_actions.append(actions)
        
        if len(next_actions) < future_k:
            pad_num = future_k - len(next_actions)
            pad_action = next_actions[-1]
            next_actions.extend([pad_action] * pad_num)
        assert len(next_actions) == future_k, "Invalid future actions"

        total_data = np.load(step_data)
        rgb_static = total_data["rgb_static"]
        rgb_gripper = total_data["rgb_gripper"]
        robot_obs = total_data["robot_obs"]

        img_static = Image.fromarray(rgb_static)
        img_static = img_static.resize(
            (TARGET_IMG_SIZE, TARGET_IMG_SIZE // 2), Image.LANCZOS
        )
        img_gripper = Image.fromarray(rgb_gripper)
        img_gripper = img_gripper.resize(
            (TARGET_IMG_SIZE, TARGET_IMG_SIZE // 2), Image.LANCZOS
        )
        img_concat = Image.new("RGB", (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
        img_concat.paste(img_static, (0, 0))
        img_concat.paste(img_gripper, (0, TARGET_IMG_SIZE // 2))

        uuid = shortuuid.ShortUUID().random(length=7)
        sample = uuid + "_" + str(step).zfill(7) + ".jpg"
        os.makedirs(
            Path("/zhaowei/data/calvin") / data_path.stem / "vla_processed_goal_img" / split,
            exist_ok=True,
        )
        img_concat.save(
            Path("/zhaowei/data/calvin")
            / data_path.stem
            / "vla_processed_goal_img"
            / split
            / sample
        )
        llm_item = get_llm_data(ann, task, split, sample, next_actions, robot_obs)
        llm_data_list.append(llm_item)
    return llm_data_list


def process_episode_goal_img(index_range: tuple, data_path: Path, split: str, future_k: int = 5):
    llm_data_list = []

    for step in tqdm(range(index_range[0], index_range[1] + 1)):
        next_actions = []
        step_data = "episode_" + str(step).zfill(7) + ".npz"
        step_data = data_path / split / step_data
        assert step_data.exists(), "Invalid data path"

        for delta in range(future_k):
            future_step = step + delta
            future_data = "episode_" + str(future_step).zfill(7) + ".npz"
            future_data = data_path / split / future_data
            if future_step <= index_range[1]:
                assert future_data.exists(), "Invalid data path"
                actions = np.load(future_data)["rel_actions"]
            else:
                break
            next_actions.append(actions)
        
        if len(next_actions) < future_k:
            pad_num = future_k - len(next_actions)
            pad_action = next_actions[-1]
            next_actions.extend([pad_action] * pad_num)
        assert len(next_actions) == future_k, "Invalid future actions"

        total_data = np.load(step_data)
        rgb_static = total_data["rgb_static"]
        rgb_gripper = total_data["rgb_gripper"]
        robot_obs = total_data["robot_obs"]

        img_static = Image.fromarray(rgb_static)
        img_static = img_static.resize(
            (TARGET_IMG_SIZE, TARGET_IMG_SIZE // 2), Image.LANCZOS
        )
        img_gripper = Image.fromarray(rgb_gripper)
        img_gripper = img_gripper.resize(
            (TARGET_IMG_SIZE, TARGET_IMG_SIZE // 2), Image.LANCZOS
        )
        img_concat = Image.new("RGB", (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
        img_concat.paste(img_static, (0, 0))
        img_concat.paste(img_gripper, (0, TARGET_IMG_SIZE // 2))

        uuid = shortuuid.ShortUUID().random(length=7)
        sample = uuid + "_" + str(step).zfill(7) + ".jpg"
        os.makedirs(
            Path("/zhaowei/data/calvin") / data_path.stem / "vla_processed_goal_img" / split,
            exist_ok=True,
        )
        img_concat.save(
            Path("/zhaowei/data/calvin")
            / data_path.stem
            / "vla_processed_goal_img"
            / split
            / sample
        )

        # construct the goal image path
        goal = "goal_" + str(index_range[1]).zfill(7) + ".jpg"
        llm_item = get_llm_data_goal_img(split, sample, goal, next_actions, robot_obs)
        llm_data_list.append(llm_item)

    # Save the goal img per episode
    goal_data = "episode_" + str(index_range[1]).zfill(7) + ".npz"
    goal_data = data_path / split / goal_data
    assert goal_data.exists(), "Invalid data path"
    goal_data = np.load(goal_data)
    goal_rgb_static = goal_data["rgb_static"]
    goal_rgb_gripper = goal_data["rgb_gripper"]
    
    goal_static = Image.fromarray(goal_rgb_static)
    goal_static = goal_static.resize(
        (TARGET_IMG_SIZE, TARGET_IMG_SIZE // 2), Image.LANCZOS
    )
    goal_gripper = Image.fromarray(goal_rgb_gripper)
    goal_gripper = goal_gripper.resize(
        (TARGET_IMG_SIZE, TARGET_IMG_SIZE // 2), Image.LANCZOS
    )
    goal_concat = Image.new("RGB", (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    goal_concat.paste(goal_static, (0, 0))
    goal_concat.paste(goal_gripper, (0, TARGET_IMG_SIZE // 2))
    goal_concat.save(
        Path("/zhaowei/data/calvin")
        / data_path.stem
        / "vla_processed_goal_img"
        / split
        / goal
    )

    return llm_data_list


def build_json_lang(data_path, debug):
    data_path = Path(data_path)
    for split in ["training", "validation"]:
        lang_info = data_path / split / "lang_annotations" / "auto_lang_ann.npy"
        assert lang_info.exists(), "Invalid data path"
        ann_data = np.load(lang_info, allow_pickle=True).item()
        lang_ann = ann_data["language"]["ann"]
        lang_task = ann_data["language"]["task"]
        lang_index = ann_data["info"]["indx"]
        partial_episode_process = partial(
            process_episode, data_path=data_path, split=split
        )
        partial_episode_process_goal_img = partial(
            process_episode_goal_img, data_path=data_path, split=split
        )

        if not debug:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = pool.map(
                   partial_episode_process, zip(lang_ann, lang_task, lang_index)
                )
            llm_data_list = [item for sub_results in results for item in sub_results]
        else:
            for zip_item in zip(lang_ann, lang_task, lang_index):
                results = partial_episode_process(zip_item)
            llm_data_list = results
        
        # Add the self-supervised data with goal image
        goal_img_info = data_path / split / "ep_start_end_ids.npy"
        assert goal_img_info.exists(), "Invalid data path"
        goal_img_data = np.load(goal_img_info)
        goal_img_index = []
        for item in goal_img_data:
            item_start, item_end, item_step = item[0], item[1], 80
            item_ranges = np.arange(item_start, item_end, item_step)
            for idx in range(len(item_ranges) - 1):
                goal_img_index.append((item_ranges[idx], item_ranges[idx + 1]))
        
        if not debug:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = pool.map(
                   partial_episode_process_goal_img, goal_img_index
                )
            goal_img_data_list = [item for sub_results in results for item in sub_results]
        else:
            for episode in goal_img_index:
                results = partial_episode_process_goal_img(episode)
            
            # debug mode: only process the last episode
            goal_img_data_list = results

        llm_data_list += goal_img_data_list
        target_file = Path(__file__).parent / (data_path.stem + "_goal_img_" + split + ".json")
        with open(target_file, "w") as json_file:
            json.dump(llm_data_list, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load the original calvin data and convert it into a json file."
    )
    parser.add_argument(
        "--calvin_data_path",
        type=str,
        help="Path to the calvin dataset directory.",
        default="/wangdonglin/calvin/task_ABCD_D",
    )
    parser.add_argument("--debug", type=bool, help="Debug mode.", default=False)
    args = parser.parse_args()
    random.seed(1234)
    np.random.seed(1234)
    build_json_lang(args.calvin_data_path, args.debug)
