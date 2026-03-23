BLOCK_SIZE=4
# Set PRETRAIN_CKPT to the path of a pretrained MDLM checkpoint
PRETRAIN_CKPT=<path_to_mdlm_checkpoint>

python -u main.py \
    loader.global_batch_size=512 \
    loader.eval_global_batch_size=512 \
    loader.batch_size=64 \
    loader.eval_batch_size=64 \
    model=small \
    algo=bd3lm \
    'algo.clip_search_widths=[0.5,0.6,0.7,0.8,0.9]' \
    data=lm1b-gpt2 \
    model.length=128 \
    block_size=${BLOCK_SIZE} \
    wandb.name=bd3lm-lm1b-gpt2-block_size${BLOCK_SIZE} \
    mode=train \
    model.attn_backend=sdpa \
    training.resample=True \
    training.from_pretrained=$PRETRAIN_CKPT

