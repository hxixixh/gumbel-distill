python -u -m main_gumbel \
    loader.global_batch_size=512 \
    loader.eval_global_batch_size=512 \
    loader.batch_size=16 \
    loader.eval_batch_size=16 \
    model=small \
    algo=mdlm_gumbel \
    data=openwebtext-split \
    data.insert_train_special=False \
    data.insert_valid_special=False \
    data.insert_valid_eos=False \
    model.length=1024 \
    wandb.name=mdlm-owt-gumbel-ablation-wokl \
    algo.kl_loss=False \
    algo.kl_weight=0
