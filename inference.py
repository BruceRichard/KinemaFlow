from utils import parse_config_from_args
from lightning.pytorch import seed_everything
from models.kinematics_net.eval import Evaluater
from utils.mylogging import Log
from pathlib import Path

from rich import print
from tqdm import tqdm

import multiprocessing
import numpy as np
import time
import random
import shutil
import torch
import os

if __name__ == '__main__':
    config = parse_config_from_args()
    Log.info(f'Loading : {Evaluater}')
    evaluator = Evaluater(config)

    text_datasets = Path('data/datasets/3_text_condition')

    obj_paths = list(text_datasets.glob('*'))
    random.shuffle(obj_paths)
    tt = time.strftime("%m-%d-%I%p-%M-%S")

    multiprocessing.set_start_method("spawn")

    OPTION = config['OPTION']
    if OPTION == 1:
        target_category_keyword = "StorageFurniture"
        num_objects = 30  
        fixed_experiment_seed = 31

        candidate_dirs = []
        for d in text_datasets.iterdir():
            if d.is_dir() and target_category_keyword in d.name:
                candidate_dirs.append(d)
        candidate_dirs.sort(key=lambda x: x.name)

        rng = random.Random(fixed_experiment_seed)
        if len(candidate_dirs) > num_objects:
            selected_dirs = rng.sample(candidate_dirs, num_objects)
        else:
            Log.warning(f"Found only {len(candidate_dirs)} directories, less than target {num_objects}. Using all.")
            selected_dirs = candidate_dirs
        obj_name_list = []

        for d in selected_dirs:
            txt_files = sorted(d.glob('*.txt'), key=lambda f: int(f.stem))  # 0.txt, 1.txt...
            for f in txt_files:
                formatted_name = f"{f.parent.stem}_{f.stem}"
                obj_name_list.append(formatted_name)

        for obj_name in tqdm(obj_name_list, 'obj_list'):
            output_path = Path('elog') / f"final_output" / f"ours_StorageFurniture" / f"{obj_name}"
            obj_infos = obj_name.split('_')
            text_content = (text_datasets / '_'.join(obj_infos[:2]) / (str(obj_infos[2])+'.txt')).read_text()
            print("Processing", obj_name)

            for rep in range(3):
                evaluator.inference_to_output_path(text_content, output_path / str(rep), blender_generated_gif=True)
    else:
        print('NOT SUPPORT ANYMORE.')
