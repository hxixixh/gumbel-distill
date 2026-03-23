python -u main.py \
    loader.global_batch_size=512 \
    loader.eval_global_batch_size=512 \
    loader.batch_size=128 \
    loader.eval_batch_size=128 \
    model=small \
    algo=ar \
    data=lm1b-gpt2 \
    model.length=128 \    
    wandb.name=ar-lm1b-gpt2