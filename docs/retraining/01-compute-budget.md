# Compute Budget

Before you do anything else, estimate what your run will actually cost. This page gives you the numbers for each of the three paths, plus rules of thumb for trimming them.

All throughput numbers below come from the Protenix team's benchmarking table in `../training_inference_instructions.md:241-255`. They are specifically for A100-80G with `--dtype bf16`. If you're on H100, divide wall-time by roughly 1.5–2×; on A100-40G you can only do the initial stage (crop 384) without shrinking the model.

## Path A — Training v2 from scratch

### Step count

AlphaFold 3 (and therefore Protenix) trains in four stages. The demo `train_demo.sh` only covers the **initial** stage. The full recipe that would reproduce a published checkpoint looks like this (numbers are approximations — ByteDance published the loss weights in their table but not the exact step counts per stage):

| Stage | Step count (approx.) | Crop | Diffusion batch | A100-80G s/step |
|---|---|---|---|---|
| Initial | 100,000 | 384 | 48 | 12 |
| Fine-tune 1 | 20,000 | 640 | 32 | 30 |
| Fine-tune 2 | 10,000 | 768 | 32 | 44 |
| Fine-tune 3 | 10,000 (confidence-only) | 768 | 32 | 13 |

AF3 reports using ~100k initial training steps plus tens of thousands of fine-tune steps. The exact numbers depend on loss convergence; use eval metrics (see [`06-monitoring-and-eval.md`](./06-monitoring-and-eval.md)) to decide when to stop each stage.

### GPU-hours

Multiply step count × seconds per step, then divide by 3600:

- Initial: `100,000 × 12 / 3600 ≈ 333` GPU-hours per-GPU-equivalent
- FT-1: `20,000 × 30 / 3600 ≈ 167`
- FT-2: `10,000 × 44 / 3600 ≈ 122`
- FT-3: `10,000 × 13 / 3600 ≈ 36`

These are **per-GPU sequential-equivalent hours**. In practice you'll run multi-GPU with data parallelism, so wall-time drops roughly linearly with GPU count (with imperfect scaling; expect 0.85× efficiency at 64 GPUs, 0.7× at 256 GPUs).

Total sequential GPU-hours: **~660 per-GPU-equivalent**. On an 8-GPU node that's ~82 hours of wall-time *if* scaling is perfect; realistically ~100 hours (~4 days). On a 64-GPU cluster: ~14 hours per stage-equivalent but remember there are four stages and checkpoint/restart overhead — budget 2–3 weeks of wall-clock end-to-end.

But wait — **these numbers are for the 368M v1 model**. `protenix-v2` is 464M with `hidden_scale_up: True` in four modules (see `configs/configs_model_type.py:60-80`). Expect ~1.3–1.5× the per-step cost of v1. So **realistic total for v2 ≈ 1000 GPU-hours sequential**, which on 8× A100-80G is about 5 days of pure step time and 1–2 weeks including restarts.

### Dollar costs (rented cloud)

Rough market prices in early 2026:

| Provider tier | $/GPU-hour (A100-80G) | $/GPU-hour (H100) |
|---|---|---|
| Spot / preemptible | $1–2 | $2–3 |
| On-demand | $2–4 | $4–6 |
| Reserved (1yr) | $1–2 | $2–4 |

On-demand on-demand pricing, 1000 GPU-hours × $3 ≈ **$3,000 per stage-pass**. Multi-stage with fine-tuning data-loader overhead: plan for **$8–15k** spot or **$25–50k** on-demand. Add 20–30% for failed runs you'll inevitably have.

### Hardware picks

- **A100-80G**: reference hardware for all published benchmarks. Safe bet.
- **H100**: ~1.5–2× faster per-step, worth it for long runs. Some kernels (cuEquivariance) have better H100 support than others — check `../kernels.md` and `inference_demo.sh` for tested combos.
- **A100-40G**: only OK for stage 1 (crop 384) with careful memory management. For later stages you must either shrink `diffusion_batch_size` or reduce `model.pairformer.nblocks`, both of which distort the recipe.
- **A800/H800**: same throughput as A100/H100 respectively for this workload.
- **AMD MI300X**: not tested by the Protenix team. CUDA-only kernels (cuEquivariance, DeepSpeed) will need `--triangle_attention torch --triangle_multiplicative torch`, which costs ~2× throughput.

## Path B — Fine-tuning v1.0.0

Dramatically cheaper. The dominant cost is the data loader spinning up against the full preprocessed bioassembly cache; actual training is quick because you're doing thousands of steps, not hundreds of thousands.

### Typical budget

| Fine-tune scope | Steps | Wall-time (1× A100-80G) | GPU-hours |
|---|---|---|---|
| Confidence-only (stage-3-style) | 5,000 | ~18 h | ~18 |
| Small family subset (100–500 PDBs) | 10,000 | ~33 h | ~33 |
| Broad domain (10k+ PDBs) | 30,000 | ~100 h | ~100 |
| Crop-768 ligand-focused FT | 10,000 | ~122 h | ~122 |

On 8× A100-80G overnight, all of these are feasible in 2–15 wall-clock hours. **A single 8-GPU node is the right fine-tuning hardware.**

### Dollar costs

At $2.5/GPU-hour on-demand 8×A100 = $20/hour, a 15-hour overnight run is ~$300. A full family fine-tune with eval is under $1000.

## Path C — Distillation

Cost is dominated by teacher inference over your unlabelled sequence corpus:

- Teacher = v1.0.0. From `../training_inference_instructions.md:263-269`, inference on `N_token=1000, N_atom=10000` takes ~59 s on A100-80G.
- A MGnify-scale corpus of 10M sequences at median length ~200 residues ≈ ~3M "small" predictions. At ~20 s each: ~16k GPU-hours just for teacher inference.
- Student training on the distilled outputs is similar cost to fine-tuning (~100–500 GPU-hours).

**Total: ~15–20k GPU-hours, ~$30–50k on-demand.** Cheaper than from-scratch, more expensive than fine-tuning. Only worth it if you specifically want v2's architecture and can't get there via fine-tuning.

See [`08-distillation.md`](./08-distillation.md) for why you can trim this substantially by subsampling the corpus intelligently.

## Storage and memory you'll actually consume

- **Preprocessed training data**: ~1.5 TB (see `scripts/database/download_protenix_data.sh` and [`02-data-preparation.md`](./02-data-preparation.md)). You need this for paths A and B.
- **Checkpoints**: each v2 checkpoint with EMA is ~2 × 1.9 GB = ~3.8 GB. At `checkpoint_interval=400` over a 100k-step stage, that's ~950 GB if you keep them all. In practice prune aggressively and keep only best-by-eval plus last-three.
- **WandB logs**: trivial (~MB).
- **Peak per-GPU memory**: 34–48 GB depending on stage; see `../training_inference_instructions.md:253`. Stage 2 (crop 768, batch 32) is the tightest.

## Rules of thumb for trimming budget

1. **Smaller crop** → big memory and time savings. `train_crop_size 384 → 256` roughly halves per-step cost at ~5–10% metric cost.
2. **Fewer cycles**. Default `model.N_cycle 4` in the demo; `2` cuts per-step time ~30% at noticeable accuracy cost. Only acceptable during debugging.
3. **Lower `diffusion_batch_size`** if you're memory-bound. Doesn't change total compute, but opens smaller GPUs.
4. **Skip stage 2 and stage 3**. If you only care about structure quality (not confidence calibration or bond geometry), stage 1 alone gives you ~80% of final performance. Stage 3 in particular is cheap (~13 s/step) and high-value — don't skip it.
5. **Use `--dtype bf16`** (the default). `fp32` is ~2× slower with negligible quality gain.
6. **EMA overhead is ~5%** — keep it. Disabling costs more than it saves in quality.

## Quick pick

| You have | Do this |
|---|---|
| A single workstation with a 3090/4090 | Nothing. Really. Use released checkpoints or rent. |
| A single A100-80G | Fine-tune (Path B). Skip from-scratch. |
| 8× A100 node for a weekend | Fine-tune or confidence-only Path B. |
| 8× A100 node for a month | Distillation (Path C) or aggressive Path B with multiple recipes. |
| 64+ A100/H100s for 2+ weeks | Path A, from scratch. |

Continue to [`02-data-preparation.md`](./02-data-preparation.md) for downloading and preparing the training data.
