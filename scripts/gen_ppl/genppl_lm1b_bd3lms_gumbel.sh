LENGTH=128
SEED=42
T=1000
NUM_SAMPLES=100
GUMBEL_TEMP=0.85
DISABLE_EMA=False
FIRST_HITTING=True
KV_CACHE=false

BLOCK_SIZE=4
CKPT_PATH=<path_to_checkpoint>
GPU_ID=0 
NUCLEUS_P=0.9

CUDA_VISIBLE_DEVICES=$GPU_ID python -u -m main_gumbel \
    mode=sample_eval \
    loader.eval_batch_size=1 \
    data=lm1b-gpt2 \
    algo=bd3lm_gumbel \
    algo.sampler=semi_ar \
    algo.T=$T \
    block_size=$BLOCK_SIZE \
    model.length=$LENGTH \
    model.attn_backend=sdpa \
    eval.checkpoint_path=$CKPT_PATH \
    wandb=null \
    seed=$SEED \
    sampling.num_sample_batches=$NUM_SAMPLES \
    sampling.nucleus_p=$NUCLEUS_P \
    sampling.gumbel_temperature=$GUMBEL_TEMP \
    sampling.kv_cache=$KV_CACHE \
    sampling.logdir=$PWD/sample_logs/bd3lms_gumbel_lm1b \
    eval.disable_ema=$DISABLE_EMA \
    sampling.first_hitting=$FIRST_HITTING & 


# ---------- DDPM Caching ---------- #

LENGTH=128
SEED=42
T=16
NUM_SAMPLES=100
GUMBEL_TEMP=0.85
DISABLE_EMA=False
FIRST_HITTING=False
KV_CACHE=false
NOISE_REMOVAL=True

BLOCK_SIZE=4

CUDA_VISIBLE_DEVICES=$GPU_ID python -u -m main_gumbel \
    mode=sample_eval \
    loader.eval_batch_size=1 \
    data=lm1b-gpt2 \
    algo=bd3lm_gumbel \
    algo.sampler=semi_ar \
    algo.T=$T \
    algo.backbone=dit_gumbel \
    block_size=$BLOCK_SIZE \
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
    sampling.kv_cache=$KV_CACHE \
    sampling.logdir=$PWD/sample_logs/bd3lms_gumbel_lm1b \
    eval.disable_ema=$DISABLE_EMA &
