from .model import EACTNet

# Defaults for every model option, so that checkpoints saved with an older
# argument set can still be rebuilt.  Values are those of the full model.
MODEL_DEFAULTS = dict(
    voxel_bins=128,
    nb_of_flow=16,
    query='linear',           # [A2] base interpolation: ofaf | linear | hermite
    hermite_on='fwd',         # [A2] stacks interpolated with Hermite when query='hermite': fwd | all
    query_res='bwd',          # [A2] event residual: none | bwd (F_{t->0}, F_{t->1}) | all
    res_window='bump',        # [A2] residual window: bump 4l(1-l) | bump2 16l^2(1-l)^2
    query_res_ctx=False,      # [A2] also correct the contexts
    splat='ebc',              # [A3] avg | ebc | niklaus
    mask_mode='geo',          # [A3] ofaf | geo (shared by every splat mode)
    diff_tau=0.1,
    diff_temp=0.02,
    time_cond=True,           # [A1]
    ebc_c_init=0.2,
    ebc_learn_c=True,
    ebc_pixel_c=True,
    ebc_fuse=True,
    ebc_alpha=5.0,
)

# Option values of checkpoints saved before the CTQ / mask fixes.
LEGACY_QUERY = {'hermite_res': dict(query='hermite', hermite_on='all', query_res='all', query_res_ctx=True)}


def build_model(args):
    for k, v in MODEL_DEFAULTS.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    if args.query in LEGACY_QUERY:
        for k, v in LEGACY_QUERY[args.query].items():
            setattr(args, k, v)
        print("[warn] query='hermite_res' (unwindowed residual) is replaced by query='hermite' + windowed "
              "residual; old weights load but the residual is now zero at the anchors.")
    if hasattr(args, 'soft_diff'):        # pre-fix option
        delattr(args, 'soft_diff')
    return EACTNet(args)
