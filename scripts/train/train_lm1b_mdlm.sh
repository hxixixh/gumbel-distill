python -u -m main \
    loader.global_batch_size=512 \
    loader.eval_global_batch_size=512 \
    loader.batch_size=128 \
    loader.eval_batch_size=128 \
    model=small \
    algo=mdlm \
    data=lm1b-gpt2 \
    model.length=128 \
    wandb.name=mdlm-lm1b-gpt2 &
