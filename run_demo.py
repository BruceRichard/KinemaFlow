"""
KinemaFlow Interactive Demo
Run: python run_demo.py --config configs/stage3_kinematics/text-eval.yaml
"""
from utils import parse_config_from_args
from lightning.pytorch import seed_everything
from models.kinematics_net.eval import Evaluater
from utils.mylogging import Log
from pathlib import Path

import multiprocessing
import time
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
torch.cuda.set_device(1)

if __name__ == '__main__':
    config = parse_config_from_args()
    Log.info(f'Loading : {Evaluater}')
    evaluator = Evaluater(config)
    multiprocessing.set_start_method("spawn")

    print("=" * 60)
    print("KinemaFlow Interactive Demo")
    print("Enter text prompts to generate articulated 3D objects.")
    print("Physical Plausibility Rectification: ENABLED")
    print(f"  Rectification scale: {config.get('rectification_scale', 1.5)}")
    print("Type 'quit' or Ctrl+C to exit.")
    print("=" * 60)

    while True:
        try:
            tt = time.strftime("%m-%d-%I%p-%M-%S")
            output_path = Path('elog') / "final_output" / tt
            text_content = input("\nInput text prompt: ")

            if text_content.lower() in ('quit', 'exit', 'q'):
                break

            for rep in range(2):
                evaluator.inference_to_output_path(
                    text_content, output_path / str(rep),
                    blender_generated_gif=True
                )
            print(f"Results saved to: {output_path}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            Log.error(f"Error: {e}")
            continue

    print("Demo ended.")
