# Retraining Protenix: A Practical Guide

ByteDance released the **code** for `protenix-v2` (a 464M-parameter AlphaFold 3 variant) but **not the trained weights**. This directory documents the three realistic paths to obtaining your own high-quality Protenix weights, along with the cost, time, and trade-offs of each.

If you just want to run inference with a released checkpoint, stop reading — go back to `../training_inference_instructions.md` and use `protenix_base_default_v1.0.0` or `protenix_base_20250630_v1.0.0`.

## Why this directory exists

The original Protenix documentation assumes you either:

1. Download a pretrained checkpoint and run inference, or
2. Fine-tune a pretrained checkpoint on your own subset.

It has much less to say about the realistic *economics* of (a) training v2 from scratch, (b) fine-tuning wisely, or (c) getting close to v2's quality without v2's compute bill. These docs fill that gap and bring together scattered details from `training_inference_instructions.md`, `prepare_training_data.md`, `PX2.pdf`, and the config source itself.

Every command and config value quoted here is cross-checked against the code as of the current `main` branch. If a number looks stale, grep for it in `configs/configs_base.py` or `configs/configs_model_type.py` before relying on it.

## The three paths

### Path A — Train v2 from scratch (the purist option)

Take `protenix-v2` as defined in `configs/configs_model_type.py:52`, feed it the full wwPDB preprocessed dataset (~1.5 TB), and run the AlphaFold 3 four-stage training recipe. You will end up with your own copy of the weights the ByteDance team did not publish.

**Cost**: on the order of **40–50k A100-hours** (roughly $60–100k at rented-cloud prices, or several weeks on a 64–128 GPU institutional cluster). Requires comfort with multi-node `torchrun`, mixed-precision debugging, and the ability to recover from OOM/NaN events over weeks of wall-clock time.

**When it's worth it**: you need the exact architecture and its full capability envelope; you're doing methods research; you have institutional compute; you want a reproducible baseline for your own future variants.

See [`03-from-scratch-v2.md`](./03-from-scratch-v2.md).

### Path B — Fine-tune v1.0.0 on your target (the pragmatic option)

Load the publicly released `protenix_base_default_v1.0.0` checkpoint (368M params), then do a short fine-tune on a subset curated for your problem — a specific protein family, a ligand class, an internal benchmark. A single A100-80G for a night is usually enough. Most of v2's advantage over v1 is architectural capacity that your narrow application may not actually need; fine-tuning typically closes most of the application-specific gap for a small fraction of the cost.

**Cost**: **tens to a few hundred GPU-hours**. Doable on a single A100-80G for small runs, or an 8×A100 node overnight for more ambitious ones.

**When it's worth it**: you care about performance on a specific domain (ligands, antibodies, RNA, a protein family), you already have a benchmark to beat, and you can afford to *lose* generalization outside that domain in exchange for sharper performance inside it.

See [`04-finetuning-v1.md`](./04-finetuning-v1.md).

### Path C — Distill v1.0.0 into a v2-shape student (the creative option)

Use `protenix_base_default_v1.0.0` as a teacher, instantiate a fresh v2-shape (464M) student from the same `protenix-v2` config, and train the student on the teacher's predictions over a very large unlabelled sequence set (MGnify, UniRef50, etc.). You get v2's architecture with training signal that is *free* except for the inference cost of running the teacher.

**Cost**: dominated by teacher inference over your sequence corpus — cheaper than real training, more expensive than fine-tuning. Expect 5–15k GPU-hours depending on corpus size.

**When it's worth it**: you want v2's capacity without the full PDB training pipeline, you have lots of unlabelled sequences, and you're okay with a performance ceiling below the real v2.

See [`08-distillation.md`](./08-distillation.md).

## Which path should you pick?

Rough decision tree:

```
Do you have >=64 A100/H100 GPUs for 2+ weeks?
├── Yes → Path A (from-scratch v2) is feasible. Read 01, 02, 03.
└── No  → Do you have a specific downstream problem (ligand, family, benchmark)?
         ├── Yes → Path B (fine-tune v1). Read 04, 05, and the relevant recipe.
         └── No  → Do you want the v2 architecture specifically?
                  ├── Yes → Path C (distillation). Read 08.
                  └── No  → Use v1.0.0 as released. You're done.
```

## File map

Core path documents:

| File | Purpose |
|---|---|
| [`01-compute-budget.md`](./01-compute-budget.md) | GPU-hours, dollar estimates, hardware choices |
| [`02-data-preparation.md`](./02-data-preparation.md) | Downloading / preprocessing wwPDB + custom CIFs |
| [`03-from-scratch-v2.md`](./03-from-scratch-v2.md) | Full four-stage v2 training recipe |
| [`04-finetuning-v1.md`](./04-finetuning-v1.md) | Fine-tune v1.0.0 on a subset |
| [`05-loss-weights.md`](./05-loss-weights.md) | What every `alpha_*` means, when to change it |
| [`06-monitoring-and-eval.md`](./06-monitoring-and-eval.md) | Reading loss curves, eval metrics, divergence signals |
| [`07-troubleshooting.md`](./07-troubleshooting.md) | OOM, NaN, slow-step debugging |
| [`08-distillation.md`](./08-distillation.md) | Teacher-student distillation into v2 shape |

Task-specific recipes:

| File | Purpose |
|---|---|
| [`recipes/ligand-docking.md`](./recipes/ligand-docking.md) | Fine-tune for protein–ligand pose quality |
| [`recipes/protein-family.md`](./recipes/protein-family.md) | Narrow to kinases, GPCRs, antibodies, etc. |
| [`recipes/posebusters.md`](./recipes/posebusters.md) | Optimize PoseBusters benchmark score |
| [`recipes/confidence-only.md`](./recipes/confidence-only.md) | Cheap stage-3-style PAE/pLDDT refinement |

## Conventions used in these docs

- **Code references** use `file:line` form (e.g. `configs/configs_base.py:400`) so you can jump to the line directly.
- **Command blocks** are copy-pasteable as written, but always read them before running — the training loop is expensive and mistakes compound.
- **"Stage N"** refers to the AF3-style stage taxonomy summarized in the training-cost table of `../training_inference_instructions.md:241-255`: initial, fine-tune 1, fine-tune 2, fine-tune 3.
- **Environment variables** assumed set: `PROTENIX_ROOT_DIR` (data root), `PYTHONPATH` (includes repo root), `CUTLASS_PATH` if using DeepSpeed kernels. See `../../CLAUDE.md` for defaults.

## External references worth reading first

- **AlphaFold 3 paper**, *Abramson et al., Nature 2024*. The Protenix training schedule is a faithful reproduction.
- **`docs/PX2.pdf`** — the Protenix-v2 technical report. Primary source for v2-specific architectural decisions.
- **`docs/PTX_V1_Technical_Report_202602042356.pdf`** — v1.0.0 report. Details the data cutoff and training curriculum.

If anything in these docs contradicts those sources, trust the source and open an issue.
