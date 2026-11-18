#!/bin/bash
# Usage: bash train_model.sh <config> [extra args]
#
# Every row changes ONE thing relative to the row it is compared with:
#   a0 -> a0lin -> a0frame -> a0fix        baseline chain (query fix, time axis, masks)
#   a0fix -> a1 / a2h / a2r / a3n / a3     single-component ablations
#   full = a1 + a2r + a3c
# Minimum set: a0, a0fix, a1, a2r, a3, full.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
CFG=${1:-full}
shift || true

COMMON="--epoch 20 --lr 2e-4 --batch_size 1 --num_worker 8 --dataset gopro \
        --voxel_bins 128 --nb_of_flow 16 --seed 1234 \
        --gopro_root /DATASSD1/GOPRO --bsergb_root /DATASSD1/BSERGB"

# OF-AF training times: GoPro skip 31, all 31 targets (t = j/32 -> pct in {0, 0.5} with n = 16).
OFAF_T="--train_skips 31 --skip_sampling concat --targets_per_sample -1"
# Optional: more sub-segment positions for the CTQ residual (only for the *_mix rows).
MIXED="--train_skips 3,4,5,6,7,9,11,15 --skip_sampling random --targets_per_sample 3"

# corrected baseline: frame time axis, consistent linear query, geometric masks, no new component
FIX="--time_axis frame --partition uniform --query linear --query_res none --splat avg --mask_mode geo --time_cond false"

case $CFG in
  # ---- baseline chain -------------------------------------------------------------------
  a0)      ARGS="$OFAF_T --time_axis legacy --partition uniform --query ofaf --query_res none \
                 --splat avg --mask_mode ofaf --time_cond false" ;;               # OF-AF re-implementation
  a0lin)   ARGS="$OFAF_T --time_axis legacy --partition uniform --query linear --query_res none \
                 --splat avg --mask_mode ofaf --time_cond false" ;;               # + consistent query weights
  a0frame) ARGS="$OFAF_T --time_axis frame  --partition uniform --query linear --query_res none \
                 --splat avg --mask_mode ofaf --time_cond false" ;;               # + key-frame time axis
  a0fix)   ARGS="$OFAF_T $FIX" ;;                                                 # + geometric / soft masks
  # ---- A1 ----------------------------------------------------------------------------
  a1)      ARGS="$OFAF_T $FIX --partition eatp --time_cond true" ;;
  a1_n8)   ARGS="$OFAF_T $FIX --partition eatp --time_cond true --nb_of_flow 8" ;;
  # ---- A2 (evaluate on GoPro skip 31, --eval_frames odd) --------------------------------
  a2h)     ARGS="$OFAF_T $FIX --query hermite --hermite_on fwd" ;;                # Hermite alone
  a2r)     ARGS="$OFAF_T $FIX --query_res bwd --res_window bump" ;;               # linear + windowed bwd residual
  a2r2)    ARGS="$OFAF_T $FIX --query hermite --hermite_on fwd --query_res bwd --res_window bump2" ;;
  a2r_mix) ARGS="$MIXED  $FIX --query_res bwd --res_window bump" ;;
  # ---- A3 (same masks as a0fix; only the splat metric changes) --------------------------
  a3n)     ARGS="$OFAF_T $FIX --splat niklaus" ;;
  a3)      ARGS="$OFAF_T $FIX --splat ebc --ebc_learn_c false --ebc_pixel_c false" ;;
  a3c)     ARGS="$OFAF_T $FIX --splat ebc --ebc_learn_c true" ;;
  # ---- full ----------------------------------------------------------------------------
  full)    ARGS="$OFAF_T $FIX --partition eatp --time_cond true --query_res bwd --splat ebc --ebc_learn_c true" ;;
  full_mix) ARGS="$MIXED $FIX --partition eatp --time_cond true --query_res bwd --splat ebc --ebc_learn_c true" ;;
  *) echo "unknown config $CFG"; exit 1 ;;
esac

python train_model.py $COMMON $ARGS --grad_report 200 --save_dir train/$CFG "$@"
