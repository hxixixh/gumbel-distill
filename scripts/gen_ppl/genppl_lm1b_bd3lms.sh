LENGTH=128
SEED=42
T=1000
NUM_SAMPLES=100
DISABLE_EMA=False
FIRST_HITTING=True
KV_CACHE=true

BLOCK_SIZE=4
CKPT_PATH=<path_to_checkpoint>
GPU_ID=0 
NUCLEUS_P=0.9

CUDA_VISIBLE_DEVICES=$GPU_ID python -u -m main \
    mode=sample_eval \
    loader.eval_batch_size=1 \
    data=lm1b-gpt2 \
    algo=bd3lm \
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
    sampling.kv_cache=$KV_CACHE \
    sampling.logdir=$PWD/sample_logs/bd3lms_lm1b \
    eval.disable_ema=$DISABLE_EMA \
    sampling.first_hitting=$FIRST_HITTING & 


# ---------- DDPM Caching ---------- #
LENGTH=128
SEED=42
T=1
NUM_SAMPLES=100
DISABLE_EMA=False
FIRST_HITTING=False
NOISE_REMOVAL=True
KV_CACHE=false

BLOCK_SIZE=4

CUDA_VISIBLE_DEVICES=$GPU_ID python -u -m main \
    mode=sample_eval \
    loader.eval_batch_size=1 \
    data=lm1b-gpt2 \
    algo=bd3lm \
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
    sampling.first_hitting=$FIRST_HITTING \
    sampling.noise_removal=$NOISE_REMOVAL \
    sampling.kv_cache=$KV_CACHE \
    sampling.logdir=$PWD/sample_logs/bd3lms_lm1b \
    eval.disable_ema=$DISABLE_EMA \
    sampling.first_hitting=$FIRST_HITTING & 

