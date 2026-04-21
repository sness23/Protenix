# Loss Weights — What Every `alpha_*` Does

Protenix's loss is a weighted sum of several terms, each controlled by a coefficient in `configs/configs_base.py:400-408`. Understanding what each does and how they interact is the difference between a principled fine-tune and a random walk.

```python
# configs/configs_base.py:400-408
"weight": {
    "alpha_confidence": 1e-4,
    "alpha_pae": 0.0,          # or 1 in finetuning stage 3
    "alpha_except_pae": 1.0,
    "alpha_diffusion": 4.0,
    "alpha_distogram": 3e-2,
    "alpha_bond": 0.0,         # or 1 in finetuning stages
    "smooth_lddt": 1.0,        # or 0 in finetuning stages
},
```

And the sub-loss weights in `configs/configs_base.py:431-438`:

```python
"diffusion": {
    "mse": {
        "weight_mse": 1 / 3,
        "weight_dna": 5.0,
        "weight_rna": 5.0,
        "weight_ligand": 10.0,
    },
    ...
}
```

This page explains each one and the practical consequences of moving them.

## The top-level terms

### `alpha_diffusion` (default 4.0)

Weight on the diffusion-module training loss (denoising MSE). **This is the dominant structure-prediction signal during stages 1–3.**

- **Raise** → model focuses more on raw structural accuracy.
- **Lower** → model trades structure for other terms (confidence, distogram).
- **Set to 0** → stage-4-style mode: freeze diffusion, only train auxiliary heads. Also set `--train_confidence_only true` when you do this.

Rarely worth changing from 4.0 except to zero it out for confidence-only stages.

### `alpha_distogram` (default 0.03)

Weight on the distogram head loss (predicting pairwise distance histograms between residues). Provides a dense auxiliary signal that helps the pair representation learn useful geometry even when diffusion is noisy.

- **Raise** (e.g. to 0.1) → stronger pair-representation supervision, can help early training stability.
- **Lower** or zero → during late-stage fine-tuning when the pair representation is already well-trained.

In the AF3/Protenix recipe this stays at 0.03 through stages 1–3 and drops to 0 in stage 4.

### `alpha_confidence` (default 1e-4)

Weight on the confidence-head auxiliary losses (pLDDT, experimentally-resolved, PDE) — **excluding** PAE, which has its own weight.

The magnitude is tiny (`1e-4`) because these losses are noisy and their *gradient* into the representation trunk would otherwise destabilize diffusion training. During stage 4 when you've detached the trunk (`train_confidence_only true`), this weight's small magnitude doesn't matter — the confidence-head-internal gradients are what drive learning.

Don't change this unless you have a specific reason.

### `alpha_pae` (default 0.0 → 1.0 in stage 4)

PAE (Predicted Aligned Error) loss weight. PAE is a special confidence metric because it requires a comparison between the predicted and true structure *after* rigid-body alignment of one chain — that's expensive and noisy during early training.

- **Keep at 0 during stages 1–3**.
- **Set to 1.0 during stage 4** with `--train_confidence_only true`.

If you set `alpha_pae > 0` without freezing diffusion, the PAE loss will inject garbage gradients into the representation trunk and probably destabilize structure learning. Don't do this.

### `alpha_except_pae` (default 1.0)

Weight on the *non-PAE* portion of the confidence losses. Always 1.0 in the standard recipe. Don't change unless you're doing something unusual.

### `alpha_bond` (default 0.0 → 1.0 in stages 2+)

Weight on the bond-length penalty: L2 loss on how far predicted bond lengths deviate from ideal values.

- **0 during stage 1**: the initial training needs flexibility and bond penalties over-constrain it.
- **1.0 from stage 2 onward**: once the model knows approximate structure, sharpen the bond geometry for chemical plausibility.

This is the main knob for PoseBusters validity. If you're optimizing for PB score, `alpha_bond=1.0` is non-negotiable.

### `smooth_lddt` (default 1.0 → 0 in stages 2+)

Weight on a differentiable approximation to lDDT, averaged over atom pairs.

- **1.0 during stage 1**: smooth lDDT provides a useful global accuracy signal early in training.
- **0 from stage 2 onward**: by stage 2 the diffusion loss + bond loss are sufficient, and smooth lDDT starts to plateau or even conflict.

## The sub-weights inside diffusion MSE

`configs/configs_base.py:433-436`:

```python
"diffusion": {
    "mse": {
        "weight_mse": 1/3,
        "weight_dna": 5.0,
        "weight_rna": 5.0,
        "weight_ligand": 10.0,
    }
}
```

These control **per-molecule-type weighting** inside the MSE loss. The rationale: proteins are over-represented in PDB, so without up-weighting, DNA/RNA/ligands get drowned out.

- `weight_mse` (1/3) — base scalar applied to the whole MSE term.
- `weight_dna` (5.0), `weight_rna` (5.0) — multipliers applied to MSE contributions from DNA/RNA atoms.
- `weight_ligand` (10.0) — multiplier for ligand atoms.

**Practical tweaks**:

- **Focus on ligand quality**: raise `weight_ligand` to 20 or 30. Especially effective for PoseBusters-style tasks. Caution: at extreme values (>50) you over-fit to ligand atoms at the cost of protein backbone accuracy.
- **Focus on nucleic acids**: raise `weight_dna` / `weight_rna` to 10+.
- **Protein-only tasks**: you can set the others to 0 (not recommended unless your dataset has no nucleic acids or ligands at all).

Pass from CLI with dotted keys:

```bash
--loss.diffusion.mse.weight_ligand 20.0
```

## The resolution filter

`configs/configs_base.py:399`:

```python
"resolution": {"min": 0.1, "max": 4.0},
```

The loss is only computed on structures whose resolution falls in this range. Structures outside are still used for *sampling* but don't contribute to the loss. If you want to include lower-resolution structures (e.g. cryo-EM above 4Å), raise `max` — but know that noisy ground truth is a double-edged sword.

## Visualizing the loss composition

When you turn on WandB, you'll see separate scalar panels for each term. A rough guide to what normal-looking curves look like during stage 1:

| Metric | Typical trajectory |
|---|---|
| `loss/diffusion_total` | Drops steeply from ~5 to ~0.5 over first 10k steps, then slow decline |
| `loss/smooth_lddt` | Drops from ~0.4 to ~0.15 over 20k steps |
| `loss/distogram` | Drops from ~3.5 to ~1.5 over 10k steps |
| `loss/bond` (if enabled) | Starts near 0 (not penalized at step 0), rises during early learning, stabilizes |
| `loss/confidence_plddt` | Noisy around 0.1; mainly moves in stage 4 |
| `loss/confidence_pae` (stage 4 only) | Drops steadily through stage 4 |

If a loss is flat from step 0, its weight is probably 0 or its term is broken. If a loss is exploding, reduce its weight or check gradient clipping (`configs/configs_base.py` has `clip_grad_norm` — default 10.0).

## Aggressive recipes (use with care)

### Ligand-first

Heavy weighting toward ligands; good for docking problems:

```bash
--loss.weight.alpha_diffusion 4.0 \
--loss.weight.alpha_bond 1.0 \
--loss.weight.smooth_lddt 0 \
--loss.diffusion.mse.weight_ligand 25.0
```

### Confidence-rich

Amplifies confidence signal during a stage-4 recalibration:

```bash
--loss.weight.alpha_diffusion 0 \
--loss.weight.alpha_distogram 0 \
--loss.weight.alpha_confidence 1e-3 \
--loss.weight.alpha_pae 2.0 \
--loss.weight.alpha_bond 0
```

(Note the 10× increase in `alpha_confidence` is only safe because diffusion is frozen.)

### Structure-only

Strips away auxiliary terms; for methodology experiments:

```bash
--loss.weight.alpha_diffusion 4.0 \
--loss.weight.alpha_distogram 0 \
--loss.weight.alpha_confidence 0 \
--loss.weight.alpha_pae 0 \
--loss.weight.alpha_bond 0 \
--loss.weight.smooth_lddt 0
```

Not recommended for real training — you'll lose the auxiliary regularization — but useful for isolating the diffusion loss's contribution to a metric.

## Debugging unusual loss curves

- **Diffusion loss NaN within first 100 steps**: check `dtype bf16` is set (fp16 is known-broken for this model), check `warmup_steps` isn't 0.
- **Diffusion loss spikes after a checkpoint restart**: you forgot `--load_ema_checkpoint_path`. The non-EMA weights alone often restart in a shifted regime.
- **Distogram loss flat at 0 from step 0**: dataset is missing distance ground truth — check `resolution` filter isn't excluding everything.
- **Smooth lDDT going up, not down**: EMA is wrong or learning rate is way too high. Check the LR schedule printout.
- **Confidence losses all zero**: `alpha_confidence` is 0, or `train_confidence_only` is set without loading a checkpoint.

## When to change a loss weight vs. when to change something else

Rule of thumb: **the first thing to try when a loss curve looks wrong is usually not to change its weight**. Instead:

1. Check the LR schedule.
2. Check data loading (are you actually seeing the structures you think?).
3. Check dtype / kernel compatibility.
4. Check whether you loaded EMA properly.

If you've ruled out all of those, *then* start tuning weights.

See [`06-monitoring-and-eval.md`](./06-monitoring-and-eval.md) for what healthy curves look like side-by-side, and [`07-troubleshooting.md`](./07-troubleshooting.md) for symptom-to-fix mapping.
