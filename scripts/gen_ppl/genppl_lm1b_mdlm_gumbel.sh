LENGTH=128
SEED=42
NUM_SAMPLES=100
GUMBEL_TEMP=0.85
DISABLE_EMA=False
CKPT_PATH=<path_to_checkpoint>
GPU_ID=0 
NUCLEUS_P=0.9

# ---------- First Hitting ---------- #
T=1000
FIRST_HITTING=True

CUDA_VISIBLE_DEVICES=$GPU_ID python -u -m main_gumbel \
    mode=sample_eval \
    loader.eval_batch_size=1 \
    data=lm1b-gpt2 \
    algo=mdlm_gumbel \
    algo.sampler=ddpm_caching \
    algo.T=$T \
    algo.backbone=dit_gumbel \
    sampling.noise_removal=True \
    sampling.gumbel_temperature=$GUMBEL_TEMP \
    block_size=128 \
    model.length=$LENGTH \
    model.attn_backend=sdpa \
    eval.checkpoint_path=$CKPT_PATH \
    wandb=null \
    seed=$SEED \
    sampling.num_sample_batches=$NUM_SAMPLES \
    sampling.nucleus_p=$NUCLEUS_P \
    sampling.logdir=$PWD/sample_logs/mdlm_gumbel_lm1b \
    eval.disable_ema=$DISABLE_EMA \
    sampling.first_hitting=$FIRST_HITTING &


# ---------- DDPM Caching ---------- #
T=32
FIRST_HITTING=False
NOISE_REMOVAL=True

CUDA_VISIBLE_DEVICES=$GPU_ID python -u -m main_gumbel \
    mode=sample_eval \
    loader.eval_batch_size=1 \
    data=lm1b-gpt2 \
    algo=mdlm_gumbel \
    algo.sampler=ddpm_caching \
    algo.T=$T \
    algo.backbone=dit_gumbel \
    block_size=128 \
    model.length=$LENGTH \
    model.attn_backend=sdpa \
    eval.checkpoint_path=$CKPT_PATH \
    wandb=null \
    seed=$SEED \
    sampling.num_sample_batches=$NUM_SAMPLES \
    sampling.nucleus_p=$NUCLEUS_P \
    sampling.first_hitting=$FIRST_HITTING \
    sampling.noise_removal=$NOISE_REMOVAL \
    sampling.gumbel_temperature=$GUMBEL_TEMP \
    sampling.logdir=$PWD/sample_logs/mdlm_gumbel_lm1b \
    eval.disable_ema=$DISABLE_EMA &
