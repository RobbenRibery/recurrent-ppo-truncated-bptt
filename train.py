import torch
from docopt import docopt
from trainer import PPOTrainer
from yaml_parser import YamlParser
import wandb
import json

import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        train.py [options]
        train.py --help
    
    Options:
        --config=<path>            Path to the yaml config file [default: ./configs/cartpole.yaml]
        --run-id=<path>            Specifies the tag for saving the tensorboard summary [default: run].
        --cpu                      Force training on CPU [default: False]
    """
    options = docopt(_USAGE)
    run_id = options["--run-id"]
    cpu = options["--cpu"]
    # Parse the yaml config file. The result is a dictionary, which is passed to the trainer.
    config = YamlParser(options["--config"]).get_config()
    recurrence_setting = json.dumps(config['recurrence'])
    if config['track']:

        if 'name' in config['environment']:
            project_name = config['environment']['name']
        else:
            project_name = config['environment']['type']
        entity = None 
        wandb.init(
            project=project_name,
            entity=entity,
            sync_tensorboard=True,
            config=config,
            name=run_id + f"_{recurrence_setting}",
            monitor_gym=True,
            save_code=True,
        )


    if not cpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")

    # Initialize the PPO trainer and commence training
    print(f"Training on device {device}")
    trainer = PPOTrainer(config, run_id=run_id, device=device)
    trainer.run_training()
    trainer.close()

if __name__ == "__main__":
    main()