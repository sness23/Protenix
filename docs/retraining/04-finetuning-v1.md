# Fine-tuning v1.0.0 — The Pragmatic Path

This is Path B. Instead of training a 464M model from scratch, you start from the already-trained `protenix_base_default_v1.0.0` (368M) checkpoint and adapt it to your target. For most application goals this gets you 90%+ of what from-scratch v2 would deliver, at ~1% of the compute cost.

## When this is the right answer

Fine-tuning is the right call when **any** of the following hold:

- You care about performance on a specific domain (protein family, ligand class, benchmark) more than general capability.
- You have < 64 GPUs or < 2 weeks of wall-clock budget.
- You've already tried the released checkpoint on your problem and it's close but not quite good enough.
- You want to experiment with loss-weight recipes without burning real training compute.

Fine-tuning is the wrong call when:

- You need the larger v2 architecture specifically (then see [`08-distillation.md`](./08-distillation.md) or [`03-from-scratch-v2.md`](./03-from-scratch-v2.md)).
- Your downstream task's distribution is very far from wwPDB (then fine-tuning will catastrophically forget general knowledge; consider training from scratch on a merged dataset instead).
- The released checkpoint already works well enough — don't fine-tune for sport.

## The base recipe

`finetune_demo.sh` is a working baseline. Reproduced with annotations:

```bash
#!/bin/bash
# finetune-baseline.sh
set -euo pipefail
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PROTENIX_ROOT_DIR=/path/to/protenix_data

# One-time: grab the pretrained checkpoint
if [ ! -f "$PROTENIX_ROOT_DIR/checkpoint/protenix_base_default_v1.0.0.pt" ]; then
    wget -P "$PROTENIX_ROOT_DIR/checkpoint/" \
      https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_default_v1.0.0.pt
fi
ckpt="$PROTENIX_ROOT_DIR/checkpoint/protenix_base_default_v1.0.0.pt"

torchrun --nproc_per_node=8 runner/train.py \
  --model_name protenix_base_default_v1.0.0 \
  --run_name ft_mysubset \
  --seed 42 \
  --base_dir ./output/ft_mysubset \
  --dtype bf16 \
  --project protenix_ft \
  --use_wandb true \
  --load_checkpoint_path "$ckpt" \
  --load_ema_checkpoint_path "$ckpt" \
  --diffusion_batch_size 32 \
  --train_crop_size 384 \
  --max_steps 20000 \
  --warmup_steps 500 \
  --lr 5e-5 \
  --ema_decay 0.999 \
  --model.N_cycle 4 \
  --sample_diffusion.N_step 20 \
  --eval_interval 500 \
  --log_interval 50 \
  --checkpoint_interval 500 \
  --triangle_attention cuequivariance \
  --triangle_multiplicative cuequivariance \
  --data.train_sets weightedPDB_before2109_wopb_nometalc_0925 \
  --data.weightedPDB_before2109_wopb_nometalc_0925.base_info.pdb_list ./my_subset.txt \
  --data.test_sets recentPDB_1536_sample384_0925,posebusters_0925
```

Put `my_subset.txt` next to the script with one PDB ID per line.

## What's different from training from scratch

Three critical changes from `train_demo.sh`:

### 1. Lower learning rate

From-scratch: `--lr 0.001`. Fine-tune: `--lr 5e-5` (roughly 1/20). Rationale: the pretrained model is already near a good optimum, so you take small steps to adapt without destroying what it knows.

Common LRs by scenario:

| Situation | LR |
|---|---|
| Confidence-only (frozen diffusion) | 3e-4 |
| Small subset, stay close to pretrained | 3e-5 |
| Moderate subset, some adaptation needed | 5e-5 |
| Large subset, significant distribution shift | 1e-4 |
| Catastrophic — pretrained clearly wrong on your domain | 3e-4, consider from-scratch instead |

If you see eval metrics *regress* in the first 500 steps of fine-tuning, your LR is too high. Halve it.

### 2. Shorter warmup

From-scratch: `--warmup_steps 2000`. Fine-tune: `--warmup_steps 500` or less. The LR schedule is AF3-style (see `protenix/utils/lr_scheduler.py`) — warmup linearly ramps the LR from 0 to the target. For short runs, long warmup eats too much of your budget.

### 3. Shorter total run

From-scratch: `--max_steps 100000` for stage 1 alone. Fine-tune: typically `10000–30000` total.

Rule of thumb: fine-tune for ~5–10 passes through your subset. If your subset is 500 PDBs and effective batch is 32 (across data-parallel replicas), that's 500 × 8 / 32 = ~125 steps per pass. 5–10 passes = 600–1250 steps. Round up to at least 5000 to allow proper EMA accumulation and eval granularity.

## Preparing your subset

See the subset-building section of [`02-data-preparation.md`](./02-data-preparation.md). Briefly:

```bash
# Generate a candidate list from PDB (example: all kinase structures)
curl -s 'https://www.rcsb.org/search?...' > kinases_raw.txt
# Normalize to lowercase, strip whitespace
awk '{print tolower($1)}' kinases_raw.txt | sort -u > kinases.txt
# Intersect with training-set indices
zcat $PROTENIX_ROOT_DIR/indices/weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz \
  | cut -d, -f1 | sort -u > available.txt
comm -12 kinases.txt available.txt > my_subset.txt
wc -l my_subset.txt  # how many PDBs you actually have
```

**Hold out an eval split** — don't include your eval IDs in `my_subset.txt`. A 90/10 split is fine for most sizes; for < 200 PDBs consider 80/20.

## Single-GPU fine-tuning

If you have one A100-80G:

```bash
python3 runner/train.py \
  --model_name protenix_base_default_v1.0.0 \
  --load_checkpoint_path "$ckpt" \
  --load_ema_checkpoint_path "$ckpt" \
  --diffusion_batch_size 16 \
  --train_crop_size 256 \
  --max_steps 10000 \
  --warmup_steps 500 \
  --lr 5e-5 \
  --ema_decay 0.999 \
  --model.N_cycle 4 \
  --sample_diffusion.N_step 20 \
  --eval_interval 500 \
  --checkpoint_interval 500 \
  --triangle_attention cuequivariance \
  --triangle_multiplicative cuequivariance \
  --data.train_sets weightedPDB_before2109_wopb_nometalc_0925 \
  --data.weightedPDB_before2109_wopb_nometalc_0925.base_info.pdb_list ./my_subset.txt \
  --data.test_sets recentPDB_1536_sample384_0925 \
  --use_wandb false \
  --run_name ft_singlegpu \
  --base_dir ./output/ft_singlegpu
```

Key reductions: `diffusion_batch_size 16` (from 32), `train_crop_size 256` (from 384). Expect ~15 s/step; 10k steps = ~42 hours.

## Partial-freeze variants

You can freeze modules selectively. This is how the constraint model is trained — see `configs/configs_model_type.py:187-192`:

```python
"finetune_params_with_substring": [
    "constraint_embedder.substructure_z_embedder",
    "constraint_embedder.pocket_z_embedder",
    "constraint_embedder.contact_z_embedder",
    "constraint_embedder.contact_atom_z_embedder",
],
```

When this list is set, only parameters whose names contain one of these substrings are updated. Everything else is frozen. To use this pattern for your own fine-tune, add a similar entry to a new model config in `configs/configs_model_type.py`, or pass it from CLI:

```bash
--finetune_params_with_substring "confidence_head,distogram_head"
```

Common freeze patterns:

| Freeze everything except | When |
|---|---|
| `confidence_head` | Stage-4-style recalibration, cheap |
| `confidence_head,distogram_head` | Recalibrate both heads |
| `diffusion_module,confidence_head` | Keep representations, retune structure generation |
| `msa_module` | Adapt to different MSA domains (e.g. metagenomic vs. RCSB) |

Check `runner/train.py` for how `finetune_params_with_substring` is wired into the optimizer's param filter — if in doubt, start with a logging print to verify the right params are being updated.

## A loss-weight recipe per scenario

Which `alpha_*` weights to use depends on what you're fine-tuning for. The four stages from [`03-from-scratch-v2.md`](./03-from-scratch-v2.md) give you four recipes:

| Goal | Use stage | Crop | `alpha_bond` | `smooth_lddt` | `alpha_pae` | `train_confidence_only` |
|---|---|---|---|---|---|---|
| Broad adaptation, structure + confidence | Stage 1 | 384 | 0 | 1.0 | 0 | False |
| Tighten bond geometry for PoseBusters | Stage 2 | 640 | 1.0 | 0 | 0 | False |
| Longer complexes | Stage 3 | 768 | 1.0 | 0 | 0 | False |
| Better confidence calibration only | Stage 4 | 768 | 0 | 0 | 1.0 | True |

For full detail on what each weight does, see [`05-loss-weights.md`](./05-loss-weights.md).

## Sequential fine-tunes

A common pattern is **stage 1 style then stage 4 style**:

1. Run 10k steps with stage-1 loss weights and crop 384 to adapt the main model.
2. Run 5k steps with stage-4 loss weights (`--train_confidence_only true`) to recalibrate confidence on your new domain.

This is cheap (~20 hours total on 8× A100-80G) and produces a model with both adapted structure prediction and properly calibrated PAE/pLDDT for your target. Highly recommended over a single-stage fine-tune for most use cases.

## Evaluating your fine-tuned model

After fine-tuning, evaluate on:

1. **Your held-out subset** — did the target metric improve?
2. **The generic eval sets** (`recentPDB_1536_sample384_0925`, `posebusters_0925`) — how much general capability did you lose?
3. **If applicable, your benchmark of record** (e.g. PoseBusters, CASP ligand).

You should see:
- Target-domain metric improves — this is the point.
- Generic lDDT on `recentPDB` drops slightly (typically 0.5–2%). Acceptable as long as the target improvement is bigger.
- Generic lDDT drops *more than* 5% → you over-fit. Reduce LR, reduce steps, or use partial freezing.

Inference with the fine-tuned checkpoint:

```bash
protenix pred \
  -i examples/input.json \
  -o ./eval_output \
  -n protenix_base_default_v1.0.0 \
  --load_checkpoint_path ./output/ft_mysubset/checkpoint/last.pt \
  --use_default_params true
```

## Task-specific recipes

Concrete recipes for common goals:

- **Protein–ligand docking**: [`recipes/ligand-docking.md`](./recipes/ligand-docking.md)
- **Narrowing to a protein family**: [`recipes/protein-family.md`](./recipes/protein-family.md)
- **PoseBusters benchmark**: [`recipes/posebusters.md`](./recipes/posebusters.md)
- **Confidence recalibration only**: [`recipes/confidence-only.md`](./recipes/confidence-only.md)

Each one includes the full CLI command, subset construction notes, and expected metrics.
