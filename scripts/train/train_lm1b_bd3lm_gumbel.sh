BLOCK_SIZE=4

# Set PRETRAIN_CKPT to the path of a pretrained MDLM-Gumbel checkpoint
PRETRAIN_CKPT=<path_to_mdlm_gumbel_checkpoint>

python -u -m main_gumbel \
    loader.global_batch_size=512 \
    loader.eval_global_batch_size=512 \
    loader.batch_size=64 \
    loader.eval_batch_size=64 \
    model=small \
    algo=bd3lm_gumbel \
    'algo.clip_search_widths=[0.5,0.6,0.7,0.8,0.9]' \
    data=lm1b-gpt2 \
    model.length=128 \
    block_size=${BLOCK_SIZE} \
    wandb.name=bd3lm-gumbel-lm1b-gpt2-ablation-wokl-block_size${BLOCK_SIZE} \
    mode=train \
    model.attn_backend=sdpa \
    training.resample=True \
    algo.kl_loss=False \
    algo.kl_weight=0 \
    training.from_pretrained=$PRETRAIN_CKPT
