import argparse
import sys
sys.path.append('.')

from lib.trainers.action_depth_canny_cond_trainer import ActionDepthCannyWMTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Arguments for the main train program."
    )

    parser.add_argument('-c', '--config_file', type=str, required=True, help='Path for the config file')

    args = parser.parse_args()

    trainer = ActionDepthCannyWMTrainer(args.config_file)

    trainer.prepare_dataset()
    trainer.prepare_models()
    trainer.prepare_trainable_parameters()
    trainer.prepare_optimizer()
    trainer.prepare_for_training()
    trainer.prepare_trackers()

    trainer.train()


if __name__ == "__main__":
  
    main()
