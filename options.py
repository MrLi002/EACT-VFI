"""Command-line options shared by train_model.py and test_model.py."""


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ('1', 'true', 'yes', 'y', 'on')


MODEL_KEYS = ('voxel_bins', 'nb_of_flow', 'query', 'hermite_on', 'query_res', 'res_window', 'query_res_ctx',
              'splat', 'mask_mode', 'diff_tau', 'diff_temp', 'time_cond',
              'ebc_c_init', 'ebc_learn_c', 'ebc_pixel_c', 'ebc_fuse', 'ebc_alpha')
# data options that change what the model sees; restored from the checkpoint at test time
DATA_KEYS = ('time_axis', 'query_events')
PARTITION_KEYS = ('partition', 'eatp_lambda', 'eatp_dmax', 'eatp_mode', 'eatp_tile', 'eatp_top')


def add_model_args(p):
    g = p.add_argument_group('model')
    g.add_argument('--voxel_bins', default=128, type=int, help='total event bins = nb_of_flow * bins per segment')
    g.add_argument('--nb_of_flow', default=16, type=int, help='number of trajectory segments n')
    g.add_argument('--query', default='linear', choices=['ofaf', 'linear', 'hermite', 'hermite_res'],
                   help='[A2] base interpolation of the anchors; ofaf reproduces the released OF-AF weighting '
                        '(hermite_res: legacy name, mapped to hermite + all residuals)')
    g.add_argument('--hermite_on', default='fwd', choices=['fwd', 'all'],
                   help='[A2] with --query hermite: Hermite on F_{0->t}, F_{1->t} only (fwd) or on every stack')
    g.add_argument('--query_res', default='bwd', choices=['none', 'bwd', 'all'],
                   help='[A2] event residual inside the segment; bwd = F_{t->0}, F_{t->1} only')
    g.add_argument('--res_window', default='bump', choices=['bump', 'bump2'],
                   help='[A2] residual window, zero at both anchors: 4l(1-l), or 16l^2(1-l)^2 (keeps C1)')
    g.add_argument('--query_res_ctx', default=False, type=str2bool, help='[A2] residual also on the contexts')
    g.add_argument('--splat', default='ebc', choices=['avg', 'ebc', 'niklaus'],
                   help='[A3] forward splatting metric; niklaus = original softmax-splatting metric')
    g.add_argument('--mask_mode', default='geo', choices=['ofaf', 'geo'],
                   help='[A3] repair masks, the same for every splat: ofaf = binary hole + hard diff '
                        '(faithful, no gradient); geo = geometric coverage + soft diff mask')
    g.add_argument('--diff_tau', default=0.1, type=float, help='initial diff-mask threshold (geo)')
    g.add_argument('--diff_temp', default=0.02, type=float, help='diff-mask sigmoid temperature (geo)')
    g.add_argument('--time_cond', default=True, type=str2bool, help='[A1] feed segment length to BiFEB')
    g.add_argument('--ebc_c_init', default=0.2, type=float, help='[A3] initial contrast threshold C')
    g.add_argument('--ebc_learn_c', default=True, type=str2bool, help='[A3] learn C')
    g.add_argument('--ebc_pixel_c', default=True, type=str2bool, help='[A3] per-pixel C correction')
    g.add_argument('--ebc_fuse', default=True, type=str2bool, help='[A3] fuse L_hat from both key frames')
    g.add_argument('--ebc_alpha', default=5.0, type=float, help='[A3] initial metric scale alpha')
    return p


def add_partition_args(p):
    g = p.add_argument_group('temporal partition (A1)')
    g.add_argument('--partition', default='eatp', choices=['uniform', 'eatp'])
    g.add_argument('--eatp_lambda', default=0.25, type=float,
                   help='uniform-time weight; also the minimum segment length in units of 1/n')
    g.add_argument('--eatp_dmax', default=4.0, type=float, help='maximum segment length in units of 1/n')
    g.add_argument('--eatp_mode', default='global', choices=['global', 'dense'],
                   help='dense: count only events in the densest tiles')
    g.add_argument('--eatp_tile', default=32, type=int)
    g.add_argument('--eatp_top', default=0.1, type=float)
    return p


def add_data_args(p):
    g = p.add_argument_group('data')
    g.add_argument('--gopro_root', default='/DATASSD1/GOPRO', type=str)
    g.add_argument('--gopro_event_dir', default='events_clean_8x_v2', type=str)
    g.add_argument('--gopro_ts_file', default='timestamps_clean_8x_v2.txt', type=str)
    g.add_argument('--gopro_fps', default=240.0, type=float)
    g.add_argument('--bsergb_root', default='/DATASSD1/BSERGB', type=str)
    g.add_argument('--bsergb_ts_file', default='auto', type=str,
                   help="BS-ERGB frame timestamps relative to the scene dir, e.g. images_timestamps.txt; "
                        "'auto' tries images/timestamp.txt, images_timestamps.txt, ...; 'none' disables")
    g.add_argument('--time_axis', default='frame', choices=['frame', 'legacy'],
                   help='frame: key-frame times bound the window, events outside are dropped (A1-A3); '
                        'legacy: min/max of the uncropped window events (OF-AF-like baselines)')
    g.add_argument('--event_time_scale', default=1.0, type=float,
                   help='multiply event timestamps by this to reach the frame clock (e.g. 1e-6 for us -> s)')
    g.add_argument('--frame_time_scale', default=1.0, type=float, help='same for frame timestamps')
    g.add_argument('--query_events', default='exact', choices=['exact', 'approx'],
                   help='exact: cut raw timestamps at every query time (CTQ residual, EBC integral); '
                        'approx: split voxel bins with a triangular CDF')
    g.add_argument('--polarity', default='auto', choices=['auto', 'pm1', '01'],
                   help='GoPro npz polarity encoding')
    g.add_argument('--num_worker', default=8, type=int)
    return p


def add_eval_args(p):
    g = p.add_argument_group('evaluation')
    g.add_argument('--eval_stride', default='ofaf', choices=['ofaf', 'full'],
                   help='ofaf: windows start every skip frames; full: every skip+1 frames')
    g.add_argument('--eval_frames', default='auto', choices=['auto', 'ofaf', 'all', 'odd'],
                   help='auto: ofaf protocol on GoPro (j = 2, 4, ...), all frames on BS-ERGB; '
                        'odd: j = 1, 3, ... (with skip 31 these lie mid-way between 1/16 anchors)')
    g.add_argument('--scenes', default='default', type=str, help="'default', 'all' or comma separated names")
    g.add_argument('--max_windows', default=-1, type=int, help='limit windows per scene (quick runs)')
    g.add_argument('--dense', default=0, type=int,
                   help='also render this many uniformly spaced frames per window and report the '
                        'consecutive warping error (temporal consistency, no GT needed)')
    g.add_argument('--splat_diag', default=False, type=str2bool,
                   help='report the forward-splat projection error |I_{0->t} - I_t| in many-to-one '
                        '(collision) regions and elsewhere')
    g.add_argument('--collision_thr', default=1.5, type=float,
                   help='a target pixel is a collision pixel if its splatted coverage exceeds this')
    g.add_argument('--lpips', default=True, type=str2bool)
    g.add_argument('--save_pred', action='store_true')
    return p


def copy_keys(dst, src_dict, keys):
    for k in keys:
        if k in src_dict:
            setattr(dst, k, src_dict[k])
