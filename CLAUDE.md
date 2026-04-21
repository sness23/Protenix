# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Protenix is a trainable PyTorch reproduction of AlphaFold 3 — a biomolecular structure prediction model. It supports protein, RNA, DNA, and ligand complexes with optional MSA, RNA MSA, and template features. Multiple model variants (v2, v1.0.0, v0.5.0, mini, tiny) coexist in one codebase, selected at runtime via `--model_name`.

## Commonly Used Commands

### Install (local dev)
```bash
pip install -e .
# CPU-only variant strips nvidia/cuda extras:
python setup.py install --cpu
```

`python_requires>=3.11`. External tools `kalign` and `hmmer` are needed for template/RNA-MSA search (`apt-get install -y kalign hmmer`).

### Inference
The entry point `protenix` (installed via `console_scripts`) dispatches to `runner/batch_inference.py::protenix_cli`. Subcommands: `pred`/`predict`, `json`/`tojson`, `msa`, `mt`/`msatemplate`, `prep`/`inputprep`.

```bash
# End-to-end CLI (preprocessing handled for you)
protenix pred -i examples/input.json -o ./output -n protenix_base_default_v1.0.0 --use_template true --use_default_params true

# Raw runner (features must already be in the JSON — use `protenix prep` first)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 runner/inference.py --model_name <name> --input_json_path <path> --dump_dir <dir> ...
```

See `inference_demo.sh` for worked examples of every model variant and kernel combination.

### Training / Fine-tuning
```bash
bash train_demo.sh          # from scratch
bash finetune_demo.sh       # loads a pretrained + EMA checkpoint
# Multi-GPU:
torchrun --nproc_per_node=8 runner/train.py --model_name protenix_base_default_v1.0.0 [ARGS]
```

Training expects `PROTENIX_ROOT_DIR` set to a data root containing `mmcif_bioassembly/`, `mmcif_msa_template/`, `indices/`, etc. (see `docs/training_inference_instructions.md` for the full hierarchy; download via `scripts/database/download_protenix_data.sh --full`).

### Tests
```bash
python -m unittest discover -s tests        # whole suite
python -m unittest tests.test_installation  # single module
python -m unittest tests.test_frame.TestFrame.test_xxx  # single test
```

### Lint / format
Pre-commit hooks run flake8, ufmt (black + usort), pydoclint, and an Apache-2.0 license-header inserter on `.py`/`.sh` files.
```bash
pip install pre-commit && pre-commit install
pre-commit run --all-files
```

## Architecture

### Entry points and wiring
- `runner/batch_inference.py` — CLI dispatcher. Routes subcommands (`pred`, `prep`, `msa`, `mt`, `json`) and handles full preprocessing → inference orchestration.
- `runner/inference.py` — GPU-only inference loop; expects features already computed in the input JSON. Dynamically lowers precision based on `N_token` (`update_inference_configs` switches AMP on for `sample_diffusion` / `confidence_head` at >2560 / >3840 tokens to avoid OOM).
- `runner/train.py` — training loop, EMA, checkpointing.
- `runner/msa_search.py`, `rna_msa_search.py`, `template_search.py` — preprocessing pipelines (ColabFold-compatible MSA, HMMER-based template/RNA-MSA).

### Configuration system
Configs are plain-Python dicts with typed sentinels (`RequiredValue`, `ListValue`, `GlobalConfigValue`) from `protenix/config/extend_types.py`. All CLI flags map onto these dicts via dotted paths (e.g. `--model.N_cycle 4`, `--sample_diffusion.N_step 200`, `--data.weightedPDB_before2109_wopb_nometalc_0925.base_info.pdb_list …`).
- `configs/configs_base.py` — `basic_configs`, `data_configs`, `optim_configs` (training).
- `configs/configs_inference.py` — inference-only overrides.
- `configs/configs_model_type.py` — per-model hyperparameter presets keyed by `model_name`. Adding a new model means adding an entry here plus checkpoint logic.
- `configs/configs_data.py` — dataset registry (train/test set names used in `--data.train_sets`).

### Model stack (`protenix/model/`)
- `protenix.py::Protenix` — top-level module; assembles `InputFeatureEmbedder`, `RelativePositionEncoding`, `ConstraintEmbedder`, `MSAModule`, `TemplateEmbedder`, `PairformerStack`, `DiffusionModule`, `ConfidenceHead`, `DistogramHead`.
- `modules/` — the major building blocks (pairformer, diffusion, confidence, embedders, primitives, transformer, frames, fused_ops, head).
- `triangular/` + `tri_attention/` — pluggable kernels for triangle attention / triangle multiplicative updates. Backends selected via `--triangle_attention {triattention,cuequivariance,deepspeed,torch}` and `--triangle_multiplicative {cuequivariance,torch}`. Not every backend works with every model; see `inference_demo.sh` for the tested combos.
- `layer_norm/` — custom LayerNorm kernel (independent impl, not from OpenFold). Disable with `LAYERNORM_TYPE=torch`.
- `generator.py` — diffusion samplers (`sample_diffusion`, `sample_diffusion_training`, noise schedulers).
- `sample_confidence.py` — post-hoc confidence/ranking.

### Data (`protenix/data/`)
- `pipeline/` — `data_pipeline.py`, `dataset.py`, `dataloader.py` build the training batches from preprocessed mmCIF bioassemblies.
- `msa/`, `template/` — feature builders consumed by the pipeline.
- `tokenizer.py`, `constants.py` — residue/atom tokenization and chemical constants.
- `inference/` — inference-time feature assembly (the JSON input format is documented in `docs/infer_json_format.md`).
- `esm/` — ESM-2 protein language model integration for `protenix_mini_esm_*` variants.
- `constraint/` — atom-level contact and pocket constraint features.

### Utils (`protenix/utils/`)
`cropping.py`, `geometry.py`, `permutation/` (symmetry-aware chain permutation for loss/metrics), `metrics.py`, `lr_scheduler.py` (AF3-style schedule), `training.py`, `distributed.py`, `torch_utils.py` (incl. `autocasting_disable_decorator` used to pin FP32 regions under BF16 mixed precision).

### OpenFold borrowings
`protenix/openfold_local/` contains selected modules adapted from OpenFold (Apache-2.0). LayerNorm was re-implemented from scratch — don't confuse `openfold_local/` LayerNorm with `protenix/model/layer_norm/`.

## Environment variables

- `PROTENIX_ROOT_DIR` — data root (checkpoints at `$PROTENIX_ROOT_DIR/checkpoint/`, plus training data hierarchy). Defaults to `$HOME`.
- `CUTLASS_PATH` — needed for DeepSpeed kernel builds (e.g. `/opt/cutlass/`).
- `LAYERNORM_TYPE` — `fast_layernorm` (default) or `torch` to fall back to stock PyTorch LayerNorm.
- `PYTHONPATH` — must include the repo root when running `runner/*.py` directly (demo scripts prepend `$(pwd)`).

## Model variants — key facts

| Model | MSA | RNA MSA | Template | Params | Notes |
|---|---|---|---|---|---|
| `protenix-v2` | ✅ | ✅ | ✅ | 464M | Latest; larger representation dims |
| `protenix_base_default_v1.0.0` | ✅ | ✅ | ✅ | 368M | AF3-equivalent data cutoff (2021-09-30) |
| `protenix_base_20250630_v1.0.0` | ✅ | ✅ | ✅ | 368M | 2025-06-30 cutoff, practical use |
| `protenix_base_default_v0.5.0` | ✅ | ❌ | ❌ | 368M | Legacy, for v0.5.0-derived work |
| `protenix_mini_{esm,ism,default}_v0.5.0` | — | — | — | — | Lightweight; `esm` uses ESM-2, `ism` uses ISM, no MSA |
| `protenix_tiny_default_v0.5.0` | — | — | — | — | Ultra-light |
| `protenix_base_constraint_v0.5.0` | ✅ | — | — | — | Supports contact/pocket constraints |

Template and RNA MSA features are v1.0.0+ only. `--use_default_params true` auto-fills recommended cycles/steps per model.
