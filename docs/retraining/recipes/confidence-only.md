# Recipe: Confidence-Only Fine-tuning

The cheapest meaningful fine-tune. Freeze the diffusion module and representations, train only the confidence head. This is the AF3/Protenix "stage 4" recipe, run as a standalone fine-tune.

## When this is the right thing to do

- You use the released v1.0.0 for predictions and the structures are good, but **the PAE / pLDDT values don't correlate well with your actual ligand-pose accuracy** (or any other downstream quality measure).
- You have a specific domain — kinase inhibitors, antibody complexes, RNA structures — where the generic confidence calibration is off.
- You want better pose ranking in an ensemble prediction workflow (generate many samples, pick the best by confidence).
- You don't have budget for full fine-tuning.

This recipe does **not** improve structure accuracy. It improves *your ability to tell which predictions are good*.

## Why it's cheap

- **~13 s/step** on A100-80G (compared to ~44 s/step at crop 768 for stage 3).
- **5k–7k steps is enough** (confidence heads converge quickly).
- **No training data preprocessing changes** — uses standard datasets.
- **Total**: ~20 GPU-hours, or about 3 hours on an 8× A100-80G node.

## What it modifies

Only the confidence-related modules. Specifically:

- `ConfidenceHead` (pLDDT, experimentally-resolved prediction, PDE, PAE heads).
- `DistogramHead` is typically *not* updated — it's frozen with the rest of the representation trunk.

The `--train_confidence_only true` flag in `runner/train.py` implements this freezing. Under the hood, it builds the optimizer's parameter list with only the confidence-head params.

## The recipe

```bash
#!/bin/bash
# recipes_confidence_only.sh
set -euo pipefail
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PROTENIX_ROOT_DIR=/path/to/protenix_data

ckpt="$PROTENIX_ROOT_DIR/checkpoint/protenix_base_default_v1.0.0.pt"

torchrun --nproc_per_node=8 runner/train.py \
  --model_name protenix_base_default_v1.0.0 \
  --run_name ft_confidence_only \
  --seed 42 \
  --base_dir ./output/ft_confidence_only \
  --dtype bf16 \
  --project protenix_ft \
  --use_wandb true \
  --load_checkpoint_path "$ckpt" \
  --load_ema_checkpoint_path "$ckpt" \
  --train_confidence_only true \
  --diffusion_batch_size 32 \
  --train_crop_size 768 \
  --max_steps 5000 \
  --warmup_steps 200 \
  --lr 3e-4 \
  --ema_decay 0.999 \
  --model.N_cycle 4 \
  --sample_diffusion.N_step 20 \
  --eval_interval 250 \
  --checkpoint_interval 250 \
  --log_interval 25 \
  --triangle_attention cuequivariance \
  --triangle_multiplicative cuequivariance \
  --loss.weight.alpha_pae 1.0 \
  --loss.weight.alpha_bond 0 \
  --loss.weight.smooth_lddt 0 \
  --loss.weight.alpha_confidence 1e-4 \
  --loss.weight.alpha_diffusion 0 \
  --loss.weight.alpha_distogram 0 \
  --data.train_sets weightedPDB_before2109_wopb_nometalc_0925 \
  --data.test_sets recentPDB_1536_sample384_0925,posebusters_0925
```

Key choices and why:

- **`--train_confidence_only true`** — the whole point.
- **`--lr 3e-4`** — higher than normal fine-tune LR because only confidence head is updated. The representation trunk's stability isn't at risk.
- **`--train_crop_size 768`** — larger crops help PAE calibration learn long-range error patterns.
- **`--loss.weight.alpha_pae 1.0`** — main objective.
- **`--loss.weight.alpha_diffusion 0`, `--loss.weight.alpha_distogram 0`** — don't contaminate with structure losses you can't update.
- **`--max_steps 5000`** — confidence heads converge faster than structure prediction.
- **`--eval_interval 250`** — fast feedback; cheap since we're running fewer total steps.
- **No `--data.<set>.base_info.pdb_list`** — use all available training data. Confidence calibration benefits from diversity, not specialization.

## Single-GPU variant

Easy on one A100-80G overnight:

```bash
python3 runner/train.py \
  --model_name protenix_base_default_v1.0.0 \
  --run_name ft_confidence_only_single \
  --base_dir ./output/ft_conf_single \
  --load_checkpoint_path "$ckpt" \
  --load_ema_checkpoint_path "$ckpt" \
  --train_confidence_only true \
  --dtype bf16 \
  --diffusion_batch_size 16 \
  --train_crop_size 512 \
  --max_steps 5000 \
  --warmup_steps 200 \
  --lr 3e-4 \
  --ema_decay 0.999 \
  --model.N_cycle 4 \
  --sample_diffusion.N_step 20 \
  --eval_interval 500 \
  --checkpoint_interval 500 \
  --triangle_attention cuequivariance \
  --triangle_multiplicative cuequivariance \
  --loss.weight.alpha_pae 1.0 \
  --loss.weight.alpha_diffusion 0 \
  --loss.weight.alpha_distogram 0 \
  --loss.weight.alpha_bond 0 \
  --loss.weight.smooth_lddt 0 \
  --data.train_sets weightedPDB_before2109_wopb_nometalc_0925 \
  --data.test_sets recentPDB_1536_sample384_0925 \
  --use_wandb false
```

~16 hours on one A100-80G.

## Specialized variants

### Confidence for a specific domain

If your domain is antibodies or RNA, restrict the training data to improve calibration specifically there:

```bash
--data.weightedPDB_before2109_wopb_nometalc_0925.base_info.pdb_list ./antibody_subset.txt \
--data.test_sets recentPDB_1536_sample384_0925
```

Make the subset 500–2000 entries. More is probably unnecessary for confidence training.

### Pair it with main-path fine-tuning

The most impactful sequence is:

1. Main fine-tune (family or ligand-focused) — 15k–25k steps.
2. Confidence-only fine-tune — 5k steps.

Starting checkpoint for step 2 is the *output* of step 1:

```bash
--load_checkpoint_path ./output/ft_mytarget/checkpoint/last.pt \
--load_ema_checkpoint_path ./output/ft_mytarget/checkpoint/ema.pt
```

This is already built into the `confidence-only` version of the [ligand docking](./ligand-docking.md) and [protein family](./protein-family.md) recipes.

### Calibration only, without PAE

If you only care about pLDDT (e.g. for ranking single predictions, not interpreting interfaces):

```bash
--loss.weight.alpha_pae 0 \
--loss.weight.alpha_confidence 1e-3   # up from 1e-4 for stronger pLDDT/resolved signal
```

This runs even faster because PAE loss is the most expensive confidence term (requires alignment).

## Evaluating confidence calibration

After training, the relevant metrics aren't lDDT — they're **calibration metrics** comparing predicted confidence to actual accuracy.

Main metrics to compute:

| Metric | Meaning | Target |
|---|---|---|
| pLDDT–lDDT correlation | How well does pLDDT predict lDDT? | > 0.80 |
| PAE mean error vs. actual error | Is PAE a tight predictor of residue-pair alignment error? | MAE < 5Å typical |
| Ranking accuracy | When you generate 25 candidates, is the top-pLDDT candidate the best? | > 60% top-1 |

Evaluation script sketch:

```python
# After generating N predictions per target with different seeds:
pred_plddt = [p["plddt"].mean() for p in predictions]
actual_lddt = [lddt(p["coords"], truth) for p in predictions]
from scipy.stats import pearsonr
r, p = pearsonr(pred_plddt, actual_lddt)
print(f"pLDDT–lDDT correlation: r = {r:.3f} (p = {p:.3g})")
```

Baseline v1.0.0 typically gets r ≈ 0.75. After this recipe on generic data, expect ~0.82. With domain-specific calibration, can hit ~0.88.

## Why this matters for ensemble prediction

A common inference workflow:

```bash
protenix pred ... --seeds 101,102,103,104,105,106,107,108,109,110 --sample_diffusion.N_sample 5
```

This generates 50 candidate structures. You want to pick the best. Without well-calibrated confidence, you're guessing; with it, you approach oracle performance (i.e. your top-1 pick is within 1% lDDT of the best in the ensemble).

The confidence-only fine-tune is often the **highest ROI** fine-tune you can do: small compute, big downstream impact.

## Limitations

- **Only as good as your ranking capacity.** If the released v1.0.0 makes bad predictions sometimes, this recipe improves your ability to *know* they're bad, but doesn't make them better. For that you need [structural fine-tuning](../04-finetuning-v1.md).
- **PAE calibration is domain-sensitive.** A model calibrated on monomeric proteins may be miscalibrated on multimers. Consider domain-specific confidence fine-tunes if you work across very different structural classes.
- **Confidence heads can overfit fast.** If you see `eval/pae_correlation` plateau or regress after step 3k, stop early.
