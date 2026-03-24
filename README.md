# Gumbel Distillation for Parallel Text Generation

[![Conference](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://iclr.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper **[Gumbel Distillation for Parallel Text Generation](#)**, published at **ICLR 2025**.

> **Authors**: Chi Zhang*, Xixi Hu*, Bo Liu, Qiang Liu <br>
> *Department of Computer Science, The University of Texas at Austin*

## Abstract
The slow, sequential nature of autoregressive (AR) language models has driven the adoption of parallel decoding methods. However, these non-AR models often sacrifice generation quality as they struggle to model the complex joint distribution of token sequences. To narrow this performance gap, we introduce **Gumbel Distillation**, a novel distillation technique that enables parallel decoders to learn this distribution effectively. 

Our method leverages the Gumbel-Max trick to create a deterministic mapping from a latent Gumbel noise space to the output tokens of a high-performing AR teacher. As a model-agnostic technique, Gumbel Distillation seamlessly integrates with diverse parallel decoding architectures, including **MDLM** and **BD3-LM**. Experiments show that our method substantially improves generation quality, achieving a **30.0% improvement in MAUVE Score** and **10.5% in generative perplexity** over MDLM trained on the OpenWebText dataset.

## ⚙️ Getting Started

### Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/your-username/gumbel-distillation.git
cd gumbel-distillation

# Create a conda environment
conda create -n gumbel-distill python=3.10
conda activate gumbel-distill

# Install dependencies
pip install -r requirements.txt

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

pip install --no-cache-dir --no-deps --force-reinstall https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

If the flash attention install above doesn't work, you can install it from scratch:

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=8 pip install flash-attn==2.5.6 --no-build-isolation --no-cache-dir
```

*(Note: Please update the requirements.txt depending on your codebase).*

## 🚀 Training

We provide bash scripts to train the Autoregressive (AR) baselines, the original parallel models (MDLM, BD3-LM), and our proposed **Gumbel Distilled** variants on two datasets: `LM1B` and `OpenWebText`.

All training scripts are located in `scripts/train/`.

### 1. LM1B Dataset

**Autoregressive (AR) Baseline:**
```bash
bash scripts/train/train_lm1b_ar.sh
```
**Masked Diffusion Language Model (MDLM):**
```bash
# Standard MDLM
bash scripts/train/train_lm1b_mdlm.sh
# MDLM with Gumbel Distillation (Ours)
bash scripts/train/train_lm1b_mdlm_gumbel.sh
```
**Block Discrete Diffusion Language Model (BD3-LM):**
```bash
# Standard BD3-LM
bash scripts/train/train_lm1b_bd3lm.sh
# BD3-LM with Gumbel Distillation (Ours)
bash scripts/train/train_lm1b_bd3lm_gumbel.sh
```

### 2. OpenWebText Dataset

**Autoregressive (AR) Baseline:**
```bash
bash scripts/train/train_owt_ar.sh
```
**Masked Diffusion Language Model (MDLM):**
```bash
# Standard MDLM
bash scripts/train/train_owt_mdlm.sh
# MDLM with Gumbel Distillation (Ours)
bash scripts/train/train_owt_mdlm_gumbel.sh
```
**Block Discrete Diffusion Language Model (BD3-LM):**
```bash
# Standard BD3-LM
bash scripts/train/train_owt_bd3lms.sh
# BD3-LM with Gumbel Distillation (Ours)
bash scripts/train/train_owt_bd3lms_gumbel.sh
```

## 📦 Checkpoints

We release pretrained checkpoints on HuggingFace: [hxixixh/gumbel-distillation-models](https://huggingface.co/hxixixh/gumbel-distillation-models)

| Model | Dataset | Filename |
|-------|---------|----------|
| MDLM | OpenWebText | `mdlm-owt.ckpt` |
| MDLM + Gumbel Distillation | OpenWebText | `mdlm-gumbel-owt.ckpt` |
| BD3-LM | OpenWebText | `bd3lms-owt.ckpt` |
| BD3-LM + Gumbel Distillation | OpenWebText | `bd3lms-gumbel-owt.ckpt` |

To download the checkpoints:

```bash
# Install huggingface_hub if needed
pip install huggingface_hub

# Download all checkpoints
huggingface-cli download hxixixh/gumbel-distillation-models --local-dir ./checkpoints

# Or download a specific checkpoint
huggingface-cli download hxixixh/gumbel-distillation-models mdlm-gumbel-owt.ckpt --local-dir ./checkpoints
```

Then use the checkpoint for evaluation, e.g.:

```bash
python main_gumbel.py \
    eval.checkpoint_path=./checkpoints/mdlm-gumbel-owt.ckpt \
    ...
```

## 📊 Evaluation (Generative Perplexity & MAUVE)

After training, you can use the evaluation scripts to compute Generative PPL and generate samples. Scripts are located in `scripts/gen_ppl/`.

### LM1B Evaluation
```bash
bash scripts/gen_ppl/genppl_lm1b_ar.sh
bash scripts/gen_ppl/genppl_lm1b_mdlm.sh
bash scripts/gen_ppl/genppl_lm1b_mdlm_gumbel.sh
```

### OpenWebText Evaluation
```bash
bash scripts/gen_ppl/genppl_owt_ar.sh
bash scripts/gen_ppl/genppl_owt_mdlm.sh
bash scripts/gen_ppl/genppl_owt_mdlm_gumbel.sh

bash scripts/gen_ppl/genppl_owt_bd3lms.sh
bash scripts/gen_ppl/genppl_owt_bd3lms_gumbel.sh
```

## 📝 Citation
If you find our work helpful or use our code in your research, please consider citing our paper:

```bibtex
@inproceedings{zhang2025gumbel,
  title={Gumbel Distillation for Parallel Text Generation},
  author={Zhang, Chi and Hu, Xixi and Liu, Bo and Liu, Qiang},
  booktitle={The Fourteenth International Conference on Learning Representations (ICLR)},
  year={2025}
}
```