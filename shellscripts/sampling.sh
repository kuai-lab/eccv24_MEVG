OUTDIR="results/" # PATH FOR THE RESULTS
PROMPTS="inputs/prompts.csv" # INPUT PROMPT
CHOICE=1 # INDEX OF THE PROMPTS CSV
CKPT_PATH="checkpoint/t2v.ckpt" # MODEL WEIGHTS
CONFIG_PATH="configs/text2video.yaml"

CUDA_VISIBLE_DEVICES=0 python scripts/sample_text2video_long.py \
    --save_dir $OUTDIR \
    --ckpt_path $CKPT_PATH \
    --config_path $CONFIG_PATH \
    --prompt $PROMPTS \
    --choice $CHOICE \
    --n_samples 1 \
    --batch_size 1 \
    --seed 1000 \
    --show_denoising_progress \
    --num_frames 16 \
    --lfai_scale 1000.0 \
    --sgs_scale 7.0 \