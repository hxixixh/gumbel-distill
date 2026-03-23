SEED=42
NUM_SAMPLES=25
CKPT_PATH=$PWD/checkpoints/mdlm-owt.ckpt
GPU_ID=0 
NUCLEUS_P=0.9

# ---------- First Hitting ---------- #

LENGTH=1024 
T=5000
FIRST_HITTING=True

CUDA_VISIBLE_DEVICES=$GPU_ID python -u -m main \
    mode=sample_eval \
    loader.eval_batch_size=1 \
    data=openwebtext-split \
    algo=mdlm \
    algo.sampler=semi_ar \
    algo.T=$T \
    block_size=1024 \
    model.length=$LENGTH \
    eval.checkpoint_path=$CKPT_PATH \
    wandb=null \
    seed=$SEED \
    sampling.num_sample_batches=$NUM_SAMPLES \
    sampling.nucleus_p=$NUCLEUS_P \
    sampling.first_hitting=$FIRST_HITTING \
    sampling.logdir=$PWD/sample_logs/mdlm_owt & 

# ---------- DDPM Caching ---------- #

LENGTH=1024 
T=128
FIRST_HITTING=False
NOISE_REMOVAL=True

CUDA_VISIBLE_DEVICES=$GPU_ID python -u -m main \
    mode=sample_eval \
    loader.eval_batch_size=1 \
    data=openwebtext-split \
    algo=mdlm \
    algo.sampler=semi_ar \
    algo.T=$T \
    block_size=1024 \
    model.length=$LENGTH \
    eval.checkpoint_path=$CKPT_PATH \
    wandb=null \
    seed=$SEED \
    sampling.num_sample_batches=$NUM_SAMPLES \
    sampling.nucleus_p=$NUCLEUS_P \
    sampling.first_hitting=$FIRST_HITTING \
    sampling.noise_removal=$NOISE_REMOVAL \
    sampling.logdir=$PWD/sample_logs/mdlm_owt & 
