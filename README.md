# MinorityPrompt

Minority-Focused Text-to-Image Generation via Prompt Optimization (CVPR 2025 Oral)

This repository contains the reference implementation for the paper:
"Minority-Focused Text-to-Image Generation via Prompt Optimization".
It implements prompt-embedding optimization during diffusion sampling for
Stable Diffusion 1.5/2.0 and SDXL, including SDXL-Lightning.

## Highlights

- Prompt-embedding optimization during sampling (no UNet finetuning).
- Works with SD 1.5 / SD 2.0 and SDXL backends.
- Supports SDXL-Lightning for fast generation.
- Optional callbacks to visualize noisy and Tweedie reconstructions.
- Example scripts for single-image and MS-COCO batch generation.

## Setup

1) Create the environment:

```bash
git clone https://github.com/anonymous5293/MinorityPrompt
cd MinorityPrompt
conda env create -f environment.yaml
```

2) (Optional) SDXL-Lightning checkpoint:

Download `sdxl_lightning_4step_unet.safetensors` from
`https://huggingface.co/ByteDance/SDXL-Lightning/tree/main`
into the `ckpt/` directory. The scripts expect filenames like:

```
ckpt/sdxl_lightning_4step_unet.safetensors
ckpt/sdxl_lightning_8step_unet.safetensors
```

## Quick Start

### Text-to-Image

```bash
source scripts/text_to_img.sh
```

### MS-COCO Batch Generation

```bash
source scripts/text_to_mscoco.sh
```

The scripts wrap the Python entrypoints under `examples/`.
Edit them to change prompts, model, sampler, or prompt-opt settings.

## Usage

### Single Image

```bash
python examples/text_to_img.py \
  --prompt "a portrait of a chef" \
  --null_prompt "" \
  --model sdxl \
  --method ddim \
  --cfg_guidance 7.5 \
  --NFE 50 \
  --seed 42
```

### MS-COCO

```bash
python examples/text_to_mscoco.py \
  --prompt_dir examples/assets/coco_v2.txt \
  --model sdxl \
  --method ddim \
  --NFE 50 \
  --num_samples 1000 \
  --b_size 4
```

### Prompt Optimization Flags

Prompt optimization updates the embeddings of newly added placeholder tokens
(e.g. `*_0`, `*_1`) at selected diffusion steps. Only these embeddings are
optimized; the rest of the text encoder weights are restored each iteration.

Key options (available in both example scripts):

- `--prompt_opt`: enable prompt optimization
- `--p_ratio`: ratio for auxiliary timestep used in the loss
- `--p_opt_iter`: number of inner optimization steps
- `--p_opt_lr`: learning rate for embedding optimization
- `--t_lo`: threshold for starting optimization (fraction of timesteps)
- `--placeholder_string`: placeholder token prefix (default `*_0`)
- `--num_opt_tokens`: number of placeholder tokens per sample
- `--init_type`: `default`, `word`, `gaussian`, or `gaussian_white`
- `--init_word`: initializer word (for `word` mode)
- `--dynamic_pr`: dynamic `p_ratio` based on current timestep
- `--base_prompt_after_popt`: revert to base prompt between optimizations
- `--inter_rate`: optimization interval (every N steps)
- `--lr_decay_rate`: linear decay for prompt-opt LR
- `--init_rand_vocab`: random vocab initialization for placeholders
- `--sg_lambda`: weighting for the stop-gradient term
- `--placeholder_position`: `start` or `end`
- `--popt_diverse`: (MS-COCO only) encourage diversity in batch mode

## Models and Solvers

### SD 1.5 / SD 2.0 (`latent_diffusion.py`)

- `ddim`: baseline DDIM sampler
- `ddim_cfg++`: CFG++ variant
- `ddim_inversion`: DDIM inversion (editing)
- `ddim_edit`: editing via WardSwap

The default SD1.5 model key is `botp/stable-diffusion-v1-5` and SD2.0 is
`stabilityai/stable-diffusion-2-base`.

### SDXL (`latent_sdxl.py`)

- `ddim`: baseline SDXL sampler
- `ddim_cfg++`: CFG++ variant
- `ddim_edit`: editing via WardSwap
- `ddim_lightning`: SDXL-Lightning variant (CFG must be 1.0)
- `ddim_cfg++_lightning`: CFG++ Lightning variant (CFG must be 1.0)

SDXL uses two text encoders and the added conditioning (`text_embeds` and
`time_ids`). Prompt optimization updates the embedding tables for both encoders.

## Project Structure

- `latent_diffusion.py`: SD 1.5/2.0 sampling and prompt optimization
- `latent_sdxl.py`: SDXL sampling and prompt optimization
- `examples/`: runnable scripts
- `scripts/`: shell wrappers for examples
- `utils/`: callbacks, logging, metrics, and helpers
- `environment.yaml`: conda environment

## Callbacks (Optional)

Both example scripts can save intermediate states via callbacks:

- `draw_noisy`: decode and save `xt` at chosen steps
- `draw_tweedie`: decode and save `x0` estimates (Tweedie)

See `utils/callback_util.py` for details.

## Metrics

`utils/calculate_metrics.py` provides FID, LPIPS, and PSNR evaluation given
an input directory and a label directory.

Example:

```bash
python utils/calculate_metrics.py \
  --input_dir path/to/generated \
  --label_dir path/to/reference \
  --exp_name my_exp
```

## Citation

If you find this repository useful, please cite:

```
@article{um2024minorityprompt,
  title={MinorityPrompt: Text to Minority Image Generation via Prompt Optimization},
  author={Um, Soobin and Ye, Jong Chul},
  journal={arXiv preprint arXiv:2410.07838},
  year={2024}
}
```

## License

See `LICENSE`.
