SEED=42
NUM_SAMPLES=25
GUMBEL_TEMP=0.85
BLOCK_SIZE=4
CKPT_PATH=$PWD/checkpoints/bd3lms-gumbel-owt.ckpt
GPU_ID=0 
NUCLEUS_P=0.9

# ---------- First Hitting ---------- #

LENGTH=1024
T=5000
FIRST_HITTING=True
KV_CACHE=false

CUDA_VISIBLE_DEVICES=$GPU_ID python -u -m main_gumbel \
    mode=sample_eval \
    loader.eval_batch_size=1 \
    data=openwebtext-split \
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
    sampling.gumbel_temperature=$GUMBEL_TEMP \
    sampling.kv_cache=$KV_CACHE \
    sampling.first_hitting=$FIRST_HITTING \
    sampling.logdir=$PWD/sample_logs/bd3lms_gumbel_owt &


# ---------- DDPM Caching ---------- #

LENGTH=1024
T=512
FIRST_HITTING=False
NOISE_REMOVAL=True
KV_CACHE=false

CUDA_VISIBLE_DEVICES=$GPU_ID python -u -m main_gumbel \
    mode=sample_eval \
    loader.eval_batch_size=1 \
    data=openwebtext-split \
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
    sampling.gumbel_temperature=$GUMBEL_TEMP \
    sampling.kv_cache=$KV_CACHE \
    sampling.first_hitting=$FIRST_HITTING \
    sampling.noise_removal=$NOISE_REMOVAL \
    sampling.logdir=$PWD/sample_logs/bd3lms_gumbel_owt & 
