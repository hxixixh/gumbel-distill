export WANDB_MODE=offline
LENGTH=1024
SEED=42
NUM_SAMPLES=25

CKPT_PATH=<path_to_checkpoint>

GPU_ID=0

NUCLEUS_P=0.9
CUDA_VISIBLE_DEVICES=$GPU_ID python -u -m main \
    mode=sample_eval \
    loader.eval_batch_size=1 \
    data=openwebtext-split \
    algo=ar \
    model.length=$LENGTH \
    eval.checkpoint_path=$CKPT_PATH \
    wandb=null \
    seed=$SEED \
    sampling.num_sample_batches=$NUM_SAMPLES \
    sampling.nucleus_p=$NUCLEUS_P \
    sampling.logdir=$PWD/sample_logs/ar_owt &
