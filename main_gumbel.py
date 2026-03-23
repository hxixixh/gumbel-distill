import os
import fsspec
import hydra
import lightning as L
import matplotlib.pyplot as plt
import omegaconf
import rich.syntax
import rich.tree
import torch
import transformers
import wandb

import dataloader
import diffusion
import utils

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.algo.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config,
    strict=False,
    weights_only=False).to('cuda')

@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)

def generate_samples(config, logger, tokenizer):
  logger.info('Generating samples.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  text_samples = model.restore_model_and_sample(
    num_steps=config.algo.T)
  print('Text samples:', text_samples)
  print('Generative perplexity:',
        model.metrics.gen_ppl.compute())
  print('Entropy:', model.metrics.gen_entropy.compute())
  print('Total entropy:', model.metrics.total_entropy)
  csv_path = config.sampling.logdir
  save_dict = {'gen_ppl': model.metrics.gen_ppls,
                'gen_nfes': model.metrics.gen_nfes,
                'gen_entropy': model.metrics.gen_entropies,
                'gen_lengths': model.metrics.gen_lengths,
                'samples': [[i] for i in text_samples],
                'seed': [config.seed for _ in range(len(text_samples))]}
  if config.sampling.var_length:
    print(text_samples)
    save_dict['samples'] = ['' for _ in range(len(text_samples))]
  utils.update_and_save_csv(save_dict, csv_path)
  
  # save a txt file with the averaged scores
  avg_scores = {
    'gen_ppl': model.metrics.gen_ppl.compute().item(),
    'gen_entropy': model.metrics.gen_entropy.compute().item(),
    'total_entropy': model.metrics.total_entropy,
    'nfes_avg': torch.tensor(model.metrics.gen_nfes).float().mean().item(),
  }
  file_path = f'{config.sampling.logdir}_avg_scores.txt'
  with open(file_path, 'w') as f:
    f.write('Average scores:\n')
    for k, v in avg_scores.items():
      f.write(f'{k}: {v}\n')
  return text_samples

def generate_samples_partial(config, logger, tokenizer):
  logger.info('Generating samples.')
  seed = config.seed
  
  if config.sampling.use_train_gumbel: 
    train_ds, _ = dataloader.get_dataloaders(
      config, tokenizer, skip_valid=True, valid_seed=seed, shuffle_train=False)
    ds = train_ds
  else:
    _, valid_ds = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, valid_seed=seed)
    ds = valid_ds
  
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  
  if not config.data.train == 'gpt2_distill' and config.sampling.use_train_gumbel:
    teacher_model = dataloader.get_teacher_model(config, tokenizer)
    model.set_teacher_model(teacher_model)
    model.teacher_model.to(model.device)
    
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  
  # get x0s
  x0s = []
  gumbel_seeds = []
  
  if config.data.train == 'gpt2_distill':
    for k, batch in enumerate(ds):
      if k >= config.sampling.num_sample_batches:
          break
      x0 = batch['input_ids']
      gumbel_seed = batch.get('gumbel_seeds', None)
      x0s.append(x0)
      gumbel_seeds.append(gumbel_seed)
    
    if x0[0].shape[0] != 1:
      x0s = torch.concat(x0s, dim=0)[:config.sampling.num_sample_batches]
      gumbel_seeds = torch.concat(gumbel_seeds, dim=0)[:config.sampling.num_sample_batches]
      
      x0s_new = []
      gumbel_seeds_new = []
      for i in range(config.sampling.num_sample_batches):
        x0s_new.append(x0s[i].unsqueeze(0))
        gumbel_seeds_new.append(gumbel_seeds[i].unsqueeze(0))
      x0s = x0s_new
      gumbel_seeds = gumbel_seeds_new
  else:
    for k, batch in enumerate(ds):
      if k >= config.sampling.num_sample_batches:
          break
      x0 = batch['input_ids']
      x0s.append(x0)
    if x0[0].shape[0] != 1:
      x0s = torch.concat(x0s, dim=0)[:config.sampling.num_sample_batches]
      x0s_new = []
      for i in range(config.sampling.num_sample_batches):
        x0s_new.append(x0s[i].unsqueeze(0))
      x0s = x0s_new
      gumbel_seeds = None
  
  gen_text_samples, original_text_samples, masked_text_samples, losses = model.restore_model_and_partial_sample(
    num_steps=config.algo.T, x0s=x0s, gumbel_seeds=gumbel_seeds)
  
  os.makedirs(config.sampling.logdir, exist_ok=True)
  avg_scores = {
    'gen_ppl': model.metrics.gen_ppl.compute().item(),
    'gen_entropy': model.metrics.gen_entropy.compute().item(),
    'total_entropy': model.metrics.total_entropy,
  }
  file_path = f'{config.sampling.logdir}_avg_scores.txt'
  with open(file_path, 'w') as f:
    f.write('Average scores:\n')
    for k, v in avg_scores.items():
      f.write(f'{k}: {v}\n')
      
  for i, (samples, original_samples, samples_masked) in enumerate(
      zip(gen_text_samples, original_text_samples, masked_text_samples)):
    print(f'Sample {i}:')
    print('Generated text:', samples)
    print('Original text:', original_samples)
    print('Masked text:', samples_masked)
    
    # Save comparison images
    utils.create_comparison_image(
      original_text=original_samples,
      predicted_text=samples,
      masked_text=samples_masked,
      output_filename=os.path.join(config.sampling.logdir, f"sample_{i}_comparison.png")
    )
    
  output_filename = f'{config.sampling.logdir}_last_k_losses.png'
  losses = torch.stack(losses, dim=0)
  last_k_losses = losses[:, 0, -config.algo.num_masked_tokens:].mean(dim=0)
  
  loss_values = last_k_losses.detach().cpu().numpy()

  fig, ax = plt.subplots(figsize=(5, 3))
  ax.plot(loss_values)

  # 4. Add labels and a title for clarity
  ax.set_title("Average Loss for Last K Tokens")
  ax.set_xlabel("Token Position Index")
  ax.set_ylabel("Average Loss")
  ax.grid(True)

  # 5. Save the plot to the specified file
  plt.savefig(output_filename, dpi=300, bbox_inches='tight')

  # 6. Close the plot to free up memory
  plt.close(fig)

  print(f"✅ Plot successfully saved to: {output_filename}")
  # for j in range(5):  # use different gumbels
  #   gen_text_samples, original_text_samples, masked_text_samples = model.restore_model_and_partial_sample(
  #     num_steps=config.algo.T, x0s=x0s)
    
  #   for i, (samples, original_samples, samples_masked) in enumerate(
  #       zip(gen_text_samples, original_text_samples, masked_text_samples)):
  #       print(f'Sample {i} / round {j}:')
  #       print('Generated text:', samples)
  #       print('Original text:', original_samples)
  #       print('Masked text:', samples_masked)
        
  #       # Save comparison images
  #       utils.create_comparison_image(
  #         original_text=original_samples,
  #         predicted_text=samples,
  #         masked_text=samples_masked,
  #         output_filename=os.path.join(config.sampling.logdir, f"sample_{i}_noise_{j}_comparison.png")
  #       )
  return samples

def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Eval.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  
  if not config.data.train == 'gpt2_distill' and config.sampling.use_train_gumbel:
    teacher_model = dataloader.get_teacher_model(config, tokenizer)
    model.set_teacher_model(teacher_model)

  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  seed = config.seed
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  L.seed_everything(seed)
  config.seed = seed
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=seed)
  trainer.validate(model, valid_ds)

def _train(config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
    
  # load teacher model
  teacher_model = dataloader.get_teacher_model(config, tokenizer)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
    logger.info(f'Resuming training at {ckpt_path}')
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  if config.training.from_pretrained is not None and ckpt_path is None:
    logger.info(f'Loading pretrained model from {config.training.from_pretrained}')
    # load pretraining checkpoint
    if 'kuleshov-group/' in config.training.from_pretrained:
      # load from hf
      model = diffusion.Diffusion(config, tokenizer=tokenizer)
      state_dict = transformers.AutoModelForMaskedLM.from_pretrained(
          config.training.from_pretrained,
          trust_remote_code=True
      ).state_dict()
      model.load_state_dict(state_dict, strict=False) # strict=False to allow for gumbel noise weights
    else:
      model = diffusion.Diffusion.load_from_checkpoint(
        config.training.from_pretrained,
        tokenizer=tokenizer,
        config=config,
        strict=False)
    # add buffers for grid search
    model.register_buffer('sampling_eps_min', torch.tensor(
      config.training.sampling_eps_min))
    model.register_buffer('sampling_eps_max', torch.tensor(
      config.training.sampling_eps_max))
  else:
    logger.info(f'Initializing new model')
    model = diffusion.Diffusion(
      config, tokenizer=valid_ds.tokenizer)
  
  if config.data.train != 'gpt2_distill':
    model.set_teacher_model(teacher_model)
    
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger, 
    num_sanity_val_steps=0)

  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)
  
@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)
  
  wandb.login(key=config.wandb_api_key)
  if config.mode == 'sample_eval':
    config.wandb = None
    samples = generate_samples(config, logger, tokenizer)
  elif config.mode == 'sample_partial_eval':
    config.wandb = None
    samples = generate_samples_partial(config, logger, tokenizer)
  elif config.mode == 'ppl_eval':
    config.wandb = None
    _ppl_eval(config, logger, tokenizer)
  else:
    _train(config, logger, tokenizer)


if __name__ == '__main__':
  main()