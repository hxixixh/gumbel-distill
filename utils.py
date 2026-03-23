"""Console logger utilities.

Copied from https://github.com/HazyResearch/transformers/blob/master/src/utils/utils.py
Copied from https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
"""

import csv
import functools
import logging
import math
import matplotlib.pyplot as plt

import fsspec
import lightning
import re
import string
import torch


from timm.scheduler import CosineLRScheduler


def count_parameters(model):
  return sum(p.numel()
             for p in model.parameters()
             if p.requires_grad)

def fsspec_exists(filename):
  """Check if a file exists using fsspec."""
  fs, _ = fsspec.core.url_to_fs(filename)
  return fs.exists(filename)


def fsspec_listdir(dirname):
  """Listdir in manner compatible with fsspec."""
  fs, _ = fsspec.core.url_to_fs(dirname)
  return fs.ls(dirname)


def fsspec_mkdirs(dirname, exist_ok=True):
  """Mkdirs in manner compatible with fsspec."""
  fs, _ = fsspec.core.url_to_fs(dirname)
  fs.makedirs(dirname, exist_ok=exist_ok)


def print_nans(tensor, name):
  if torch.isnan(tensor).any():
    print(name, tensor)

def update_and_save_csv(save_dict, csv_path):
  num_samples = len(save_dict['gen_ppl'])
  with fsspec.open(csv_path, 'a') as f:
    writer = csv.DictWriter(f, fieldnames=save_dict.keys())
    if fsspec_exists(csv_path) is False:
        writer.writeheader()
    for i in range(num_samples):
      row = {k: v[i] for k, v in save_dict.items()}
      writer.writerow(row)

class CosineDecayWarmupLRScheduler(
  CosineLRScheduler,
  torch.optim.lr_scheduler._LRScheduler):
  """Wrap timm.scheduler.CosineLRScheduler
  Enables calling scheduler.step() without passing in epoch.
  Supports resuming as well.
  Adapted from:
    https://github.com/HazyResearch/hyena-dna/blob/main/src/utils/optim/schedulers.py
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._last_epoch = -1
    self.step(epoch=0)

  def step(self, epoch=None):
    if epoch is None:
      self._last_epoch += 1
    else:
      self._last_epoch = epoch
    # We call either step or step_update, depending on
    # whether we're using the scheduler every epoch or every
    # step.
    # Otherwise, lightning will always call step (i.e.,
    # meant for each epoch), and if we set scheduler
    # interval to "step", then the learning rate update will
    # be wrong.
    if self.t_in_epochs:
      super().step(epoch=self._last_epoch)
    else:
      super().step_update(num_updates=self._last_epoch)


class LoggingContext:
  """Context manager for selective logging."""
  def __init__(self, logger, level=None, handler=None, close=True):
    self.logger = logger
    self.level = level
    self.handler = handler
    self.close = close

  def __enter__(self):
    if self.level is not None:
      self.old_level = self.logger.level
      self.logger.setLevel(self.level)
    if self.handler:
      self.logger.addHandler(self.handler)

  def __exit__(self, et, ev, tb):
    if self.level is not None:
      self.logger.setLevel(self.old_level)
    if self.handler:
      self.logger.removeHandler(self.handler)
    if self.handler and self.close:
      self.handler.close()


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
  """Initializes multi-GPU-friendly python logger."""

  logger = logging.getLogger(name)
  logger.setLevel(level)

  # this ensures all logging levels get marked with the rank zero decorator
  # otherwise logs would get multiplied for each GPU process in multi-GPU setup
  for level in ('debug', 'info', 'warning', 'error',
                'exception', 'fatal', 'critical'):
    setattr(logger,
            level,
            lightning.pytorch.utilities.rank_zero_only(
              getattr(logger, level)))

  return logger


class Sampler:
  def __init__(self, shape):
    self.shape = shape

  def _sampling_noise(self):
    pass
  
  def _hard_sample(self, logits):
    pass

  def _soft_sample(self, logits):
    return 0

  def _process_logits(self, logits):
    return logits

  def sample(self, logits):
    logits = self._process_logits(logits)
    noise = self._sampling_noise()
    noise = noise[: logits.shape[0], :]
    logits = logits + noise.to(
      dtype=logits.dtype, device=logits.device)
    hard_sample = self._hard_sample(logits)
    soft_sample = self._soft_sample(logits)
    return soft_sample + (hard_sample - soft_sample).detach()


@functools.lru_cache(maxsize=None)
def log_n_choose_k(n, k):
  prior_loss = 0.0
  for i in range(k):
    prior_loss += (math.log(n - i) - math.log(k - i))
  return prior_loss


@functools.lru_cache(maxsize=None)
def log_n_permute_k(n, k):
  ans = 0.0
  for i in range(k):
    ans += math.log(n - i)
  return ans


class TopKSampler(Sampler):
  def __init__(self, k, shape, gamma_tau=1.0,
               noise_type='sog'):
    super().__init__(shape)
    self.k = k
    self.gamma_tau = gamma_tau
    self.num_betas = 10
    self.sampler = torch.distributions.gamma.Gamma(
      1 / k * torch.ones(self.num_betas, * self.shape), 1.0)
    self.noise_type = noise_type

  def _sampling_noise(self):
    if self.noise_type == 'sog':
      noise = self.sampler.sample()
      beta = self.k / torch.arange(1, self.num_betas + 1, 1,
                                  dtype=torch.float32)
      beta = beta[:, None, None, None]
      assert beta.ndim == noise.ndim
      s = noise / beta
      s = torch.sum(s, axis=0)
      s = s - math.log(self.num_betas)
      return self.gamma_tau * (s / self.k)
    elif self.noise_type == 'gumbel':
      return - (1e-10 - (torch.rand(* self.shape)
                         + 1e-10).log()).log()
    elif self.noise_type == 'deterministic':
      return torch.zeros(* self.shape)

  def _process_logits(self, logits):
    assert logits.ndim == 3
    return logits

  def _hard_sample(self, logits):
    thresholds, _ = torch.sort(logits, dim=-1)
    thresholds = thresholds[:, :, - self.k][:, :, None]
    return (logits >= thresholds).type(logits.dtype)

  def _soft_sample(self, logits):
    soft_top_k = logits - torch.mean(logits, dim=-1,
                                     keepdim=True)
    return soft_top_k / torch.norm(soft_top_k, dim=-1,
                                   keepdim=True)

class GaussianSampler:
  def __init__(self, constrain_logits):
    self.constrain_logits = constrain_logits

  def gaussian_params_from_logits(self, logits):
    assert logits.ndim == 3
    n = logits.shape[-1] // 2
    mu = logits[:, :, :n]
    log_var = logits[:, :, n:]
    if self.constrain_logits:
      # mu \in [0, 1]
      mu = torch.tanh(mu)
      # var \in [0, 1]
      # log_var \in (- \inf, 0]
      log_var = - torch.nn.functional.softplus(log_var)
    return mu, log_var

  def sample(self, x):
    mu, log_var = self.gaussian_params_from_logits(x)
    sigma = log_var.exp().sqrt()
    return mu + sigma * torch.randn_like(mu)


def sample_gumbel(shape, eps=1e-20, temperature=1.0, device=None, return_uniform=False):
    U = torch.rand(size=shape, device=device)
    # Inverse transform sampling for a standard Gumbel(0, 1) distribution
    gumbel_standard = -torch.log(-torch.log(U + eps) + eps)
    if return_uniform:
      return U
    # Scale the standard Gumbel noise by the temperature
    return gumbel_standard * temperature


def get_posterior_gumbel(logp, samples):
    batch_size, dim = logp.shape
    xi0 = sample_gumbel((batch_size, 1), device=logp.device)
    xii = sample_gumbel((batch_size, dim), device=logp.device)
    post_gumbel_sample = -torch.log(torch.exp(-xii) + torch.exp(-xi0 + logp))

    batch_indices = torch.arange(batch_size, device=logp.device)
    post_gumbel_sample[batch_indices, samples] = (
        xi0.squeeze() - logp[batch_indices, samples]
    )
    return post_gumbel_sample


def create_comparison_image(
    original_text,
    predicted_text,
    masked_text,
    output_filename="sample_comparison.png",
    mask_token="[PAD]",
    endoftext_token="<|endoftext|>",
):
    """
    Generates an image comparing original, masked, and predicted text using matplotlib.
    This version uses an accurate text rendering engine to ensure proper spacing.
    """
    # --- 1. Process the text into a list of (word, color, style) tuples ---
    original_text = original_text.replace(endoftext_token, f' {endoftext_token} ')
    predicted_text = predicted_text.replace(endoftext_token, f' {endoftext_token} ')
    masked_text = masked_text.replace(mask_token, f' {mask_token} ')
    masked_text = masked_text.replace(endoftext_token, f' {endoftext_token} ')
    
    # def add_spacing_for_all_punctuation(text):
    #   punctuations = ['.', ',', '(', ')', '!', '?', ';', ':', '-'] # Add any other punctuation here
    #   for punc in punctuations:
    #       text = text.replace(punc, f' {punc} ')
    #   return text
    
    # original_text = add_spacing_for_all_punctuation(original_text)
    # predicted_text = add_spacing_for_all_punctuation(predicted_text)
    # masked_text = add_spacing_for_all_punctuation(masked_text)
    
    original_words = re.findall(r'\S+|\n', original_text)
    predicted_words = re.findall(r'\S+|\n', predicted_text)
    masked_words = re.findall(r'\S+|\n', masked_text)
    
    render_list = []
    
    for i in range(len(masked_words)):
        if i >= len(original_words) or i >= len(predicted_words): break

        original_word = original_words[i]
        predicted_word = predicted_words[i]
        masked_word = masked_words[i]

        if original_word == '\n':
            render_list.append(('\n', 'black', 'normal'))
            continue

        # if mask_token in masked_word:
        #     if predicted_word == original_word:
        #         render_list.append((predicted_word, 'blue', 'bold'))
        #     else:
        #         render_list.append((predicted_word, 'red', 'bold'))
        #         render_list.append((f"({original_word})", 'green', 'normal'))
        # else:
        #     render_list.append((original_word, 'black', 'normal'))
        if mask_token in masked_word:
            for original_word in original_words[i:]:
                render_list.append((original_word, 'green', 'bold'))
            render_list.append(("/", 'red', 'bold'))
            for predicted_word in predicted_words[i:]:
                render_list.append((predicted_word, 'blue', 'bold'))
            break
        else:
            render_list.append((original_word, 'black', 'normal'))

    # --- 2. Render the list of words onto a matplotlib figure ---
    fig, ax = plt.subplots(figsize=(16, 8), dpi=120)
    ax.set_axis_off()

    # Layout parameters
    x_start, y_start = 0.01, 0.95
    line_height = 0.05
    space_width_factor = 0.005 # Width of a single space
    x_pos, y_pos = x_start, y_start

    # Get the renderer to calculate text size accurately
    renderer = fig.canvas.get_renderer()

    for text, color, weight in render_list:
        if text == '\n':
            y_pos -= line_height
            x_pos = x_start
            continue
        
        # Check for word wrapping *before* drawing the text
        # Create a temporary text object to measure its width
        t_temp = ax.text(x_pos, y_pos, text, color=color, fontweight=weight,
                         ha='left', va='top', fontfamily='monospace', fontsize=20)
        
        bbox = t_temp.get_window_extent(renderer=renderer)
        text_width = bbox.transformed(fig.transFigure.inverted()).width * 1.5
        t_temp.remove() # Remove the temporary object

        if x_pos + text_width > 0.99: # 0.99 is the right margin
            y_pos -= line_height
            x_pos = x_start

        # Draw the text for real at the correct position
        t = ax.text(x_pos, y_pos, text, color=color, fontweight=weight,
                    ha='left', va='top', fontfamily='monospace', fontsize=20)

        # Update x_pos for the next word, including a space
        x_pos += text_width + space_width_factor

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1, facecolor='w')
    plt.close(fig)
    print(f"Image saved as {output_filename}")