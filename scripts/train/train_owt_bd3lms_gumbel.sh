BLOCK_SIZE=4
PRETRAIN_CKPT=kuleshov-group/bd3lm-owt-block_size4

python -u -m main_gumbel \
    loader.global_batch_size=512 \
    loader.eval_global_batch_size=512 \
    loader.batch_size=16 \
    loader.eval_batch_size=16 \
    model=small \
    algo=bd3lm_gumbel \
    algo.kl_loss=False \
    algo.kl_weight=0 \
    'algo.clip_search_widths=[0.5,0.6,0.7,0.8,0.9]' \
    data=openwebtext-split \
    data.insert_train_special=False \
    data.insert_valid_special=False \
    data.insert_valid_eos=False \
    model.length=1024 \
    block_size=${BLOCK_SIZE} \
    wandb.name=bd3lm-gumbel-owt-ablation-wokl-block_size${BLOCK_SIZE} \
    mode=train \
    model.attn_backend=flex \
    training.resample=True \
    training.from_pretrained=$PRETRAIN_CKPT