# ---------- First Hitting ---------- #

LENGTH=128
SEED=42
T=1000
NUM_SAMPLES=100
DISABLE_EMA=False
FIRST_HITTING=True
CKPT_PATH=<path_to_checkpoint>
GPU_ID=0 
NUCLEUS_P=0.9

CUDA_VISIBLE_DEVICES=$GPU_ID python -u -m main \
    mode=sample_eval \
    loader.eval_batch_size=1 \
    data=lm1b-gpt2 \
    algo=mdlm \
    algo.sampler=ddpm_caching \
    algo.T=$T \
    sampling.noise_removal=True \
    block_size=128 \
    model.length=$LENGTH \
    model.attn_backend=sdpa \
    eval.checkpoint_path=$CKPT_PATH \
    wandb=null \
    seed=$SEED \
    sampling.num_sample_batches=$NUM_SAMPLES \
    sampling.nucleus_p=$NUCLEUS_P \
    sampling.logdir=$PWD/sample_logs/mdlm_lm1b \
    eval.disable_ema=$DISABLE_EMA \
    sampling.first_hitting=$FIRST_HITTING & 

# ---------- DDPM Caching ---------- #

LENGTH=128
SEED=42
T=16
NUM_SAMPLES=100
DISABLE_EMA=False
FIRST_HITTING=False
NOISE_REMOVAL=True

CUDA_VISIBLE_DEVICES=$GPU_ID python -u -m main \
    mode=sample_eval \
    loader.eval_batch_size=1 \
    data=lm1b-gpt2 \
    algo=mdlm \
    algo.sampler=ddpm_caching \
    algo.T=$T \
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
    sampling.logdir=$PWD/sample_logs/mdlm_lm1b \
    eval.disable_ema=$DISABLE_EMA &
