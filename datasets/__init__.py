import os

from torch.utils.data import ConcatDataset

from util.misc import parse_int_list
from .gopro import GOPRO_Train, GOPRO_Eval
from .bsergb import BSERGB_Train, BSERGB_Eval

GOPRO_TEST_SCENES = ['GOPR0384_11_00', 'GOPR0384_11_05', 'GOPR0385_11_01', 'GOPR0396_11_00', 'GOPR0410_11_00',
                     'GOPR0854_11_00', 'GOPR0862_11_00', 'GOPR0868_11_00', 'GOPR0869_11_00', 'GOPR0871_11_00',
                     'GOPR0881_11_01']
BSERGB_TEST_SCENES = ['acquarium_08', 'ball_05', 'ball_06', 'basket_08', 'basket_09', 'candies_03', 'eggs_04',
                      'elastic_bands_01', 'fire_02', 'football_04', 'horse_11', 'horse_12', 'horse_13', 'horse_18',
                      'horse_20', 'jacket_03', 'juggling_06', 'may28_axe_01', 'may29_redbull_01',
                      'may29_water_tank_pouring_02', 'orange_juice_02', 'paprika_1000_gain_control_02', 'pen_03',
                      'rope_jumping_01', 'street_crossing_08', 'watermelon_01']


def _scenes(folder):
    return sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))


def get_dataset(args):
    """Training set.  --skip_sampling concat: one dataset per skip (OF-AF);
    random: one dataset per scene that draws a skip for every sample."""
    skips = parse_int_list(args.train_skips)
    groups = [[s] for s in skips] if args.skip_sampling == 'concat' else [skips]
    datasets = []
    if args.dataset == 'gopro':
        root = os.path.join(args.gopro_root, 'train')
        for scene in _scenes(root):
            for g in groups:
                datasets.append(GOPRO_Train(args, os.path.join(root, scene), g, args.crop, args.targets_per_sample))
    elif args.dataset == 'bsergb':
        root = os.path.join(args.bsergb_root, '3_TRAINING')
        for scene in _scenes(root):
            for g in groups:
                datasets.append(BSERGB_Train(args, os.path.join(root, scene), g, args.crop, args.targets_per_sample))
    else:
        raise ValueError(args.dataset)
    datasets = [d for d in datasets if len(d) > 0]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def get_eval_sets(args, dataset, skip):
    """Yields one Dataset per test scene for the given skip."""
    frames = args.eval_frames
    if frames == 'auto':
        frames = 'ofaf' if dataset == 'gopro' else 'all'
    if dataset == 'gopro':
        root = os.path.join(args.gopro_root, 'test')
        scenes = GOPRO_TEST_SCENES if args.scenes == 'default' else _resolve(args.scenes, root)
        for s in scenes:
            yield GOPRO_Eval(args, os.path.join(root, s), skip, args.eval_stride, frames, args.max_windows)
    elif dataset == 'bsergb':
        root = os.path.join(args.bsergb_root, '1_TEST')
        scenes = BSERGB_TEST_SCENES if args.scenes == 'default' else _resolve(args.scenes, root)
        for s in scenes:
            yield BSERGB_Eval(args, os.path.join(root, s), skip, args.eval_stride, frames, args.max_windows)
    else:
        raise ValueError(dataset)


def _resolve(spec, root):
    return _scenes(root) if spec == 'all' else [s for s in spec.split(',') if s]
