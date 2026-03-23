# ---------- First Hitting ---------- #

LENGTH=1024
SEED=42
T=5000
FIRST_HITTING=True
NUM_SAMPLES=25
GUMBEL_TEMP=0.85
CKPT_PATH=$PWD/checkpoints/mdlm-gumbel-owt.ckpt
GPU_ID=0 
NUCLEUS_P=0.9

CUDA_VISIBLE_DEVICES=$GPU_ID python -u -m main_gumbel \
    mode=sample_eval \
    loader.eval_batch_size=1 \
    data=openwebtext-split \
    algo=mdlm_gumbel \
    algo.sampler=semi_ar \
    algo.T=$T \
    algo.backbone=dit_gumbel \
    block_size=1024 \
    model.length=$LENGTH \
    eval.checkpoint_path=$CKPT_PATH \
    wandb=null \
    seed=$SEED \
    sampling.num_sample_batches=$NUM_SAMPLES \
    sampling.nucleus_p=$NUCLEUS_P \
    sampling.gumbel_temperature=$GUMBEL_TEMP \
    sampling.first_hitting=$FIRST_HITTING \
    sampling.logdir=$PWD/sample_logs/mdlm_gumbel_owt &



# ---------- DDPM Caching ---------- #

LENGTH=1024
FIRST_HITTING=False
NOISE_REMOVAL=True
T=512

CUDA_VISIBLE_DEVICES=$GPU_ID python -u -m main_gumbel \
    mode=sample_eval \
    loader.eval_batch_size=1 \
    data=openwebtext-split \
    algo=mdlm_gumbel \
    algo.sampler=semi_ar \
    algo.T=$T \
    algo.backbone=dit_gumbel \
    block_size=1024 \
    model.length=$LENGTH \
    eval.checkpoint_path=$CKPT_PATH \
    wandb=null \
    seed=$SEED \
    sampling.num_sample_batches=$NUM_SAMPLES \
    sampling.nucleus_p=$NUCLEUS_P \
    sampling.gumbel_temperature=$GUMBEL_TEMP \
    sampling.first_hitting=$FIRST_HITTING \
    sampling.noise_removal=$NOISE_REMOVAL \
    sampling.logdir=$PWD/sample_logs/mdlm_gumbel_owt &
