"""
Step 2 of 3: train the diffusion model.

    python train.py [--config config/coma.cfg] [--resume] [--epochs N]

Every setting comes from the config. Checkpoints and the training log go to
<work_dir>/checkpoints/predict_<predict>_<run_name>/. --resume continues an
interrupted run; any other flag is passed on to scripts/train_diffusion.py.
"""
import os
import sys
import argparse
import configparser

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--config', default=os.path.join(ROOT, 'config', 'coma.cfg'))
    args, extra = ap.parse_known_args()

    cp = configparser.RawConfigParser()
    cp.read(args.config)
    from config.config import read_config
    config = read_config(args.config)
    cache = os.path.join(config['dataset_dir'], 'processed_completion', 'training.pt')
    if not os.path.exists(cache):
        sys.exit(f'No processed data in {config["dataset_dir"]}. Run: python prepare_data.py')

    sys.argv = ['train_diffusion.py', '--config_file', args.config,
                '--tag', cp.get('training parameters', 'run_name', fallback='full'),
                '--seed', cp.get('training parameters', 'seed', fallback='1')] + extra
    from scripts.train_diffusion import main as train_main
    train_main()


if __name__ == '__main__':
    main()
