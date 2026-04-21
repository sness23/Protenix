# Monitoring and Evaluation

Training Protenix is expensive enough that you want to know *as early as possible* whether a run is healthy. This page covers what to instrument, what to watch, and how to decide whether to keep going or kill the run.

## Turn on Weights & Biases

Non-optional. Set `--use_wandb true`. Training runs are long enough that scrolling through `nohup.out` is not a reasonable way to monitor them.

```bash
pip install wandb
wandb login
```

Then add `--use_wandb true --project protenix_v2` to your launch script. The Protenix training loop already wires up the common scalars — you get loss panels, learning rate, eval metrics, and throughput for free.

If you can't use WandB (air-gapped cluster, etc.), the alternative is tensorboard — check `runner/train.py` for the tb writer path, or fall back to grepping the stdout logs.

## The three phases of a healthy training curve

Every Protenix run has three recognizable phases on the diffusion loss:

1. **Initialization burn-in** (first ~1k steps). Loss drops very steeply as the model learns basic representations. Warmup ramps LR from 0 to target during this period.
2. **Fast learning** (steps ~1k–10k). Loss decline is exponential-ish. This is when most of the quality is acquired.
3. **Slow refinement** (step 10k onward). Loss declines logarithmically. Eval metrics continue improving slowly; each additional 10k steps buys progressively less.

If your run deviates from this shape, something is wrong. Specifically:

- **No burn-in drop**: warmup too long, LR too low, or data not loading.
- **Burn-in drop then flat**: LR too high and model got stuck in a bad basin. Restart with lower LR.
- **Oscillation during fast learning**: EMA decay too low, or batch size effectively too small.

## Scalars to watch

Organized by panel group. Add these as WandB "starred metrics" if you use that feature.

### Loss panels

| Metric | Watch for |
|---|---|
| `loss/total` | Steadily declining. Occasional spikes OK, sustained increase = problem. |
| `loss/diffusion_total` | Dominant term during stages 1–3. Should be the majority of `loss/total`. |
| `loss/diffusion_bond` | Only meaningful when `alpha_bond > 0`. Should stabilize around 0.1–0.3. |
| `loss/smooth_lddt` | Only during stage 1. Should drop from ~0.4 to ~0.15. |
| `loss/distogram` | Drops from ~3.5 to ~1.5 early, then flat. |
| `loss/confidence_plddt` | Mostly interesting during stage 4. |
| `loss/confidence_pae` | Stage 4 only. |
| `loss/confidence_pde` | Always small (1e-4 × base). |
| `grad_norm` | Should be O(1) — raw value depends on model, but if it drifts above 50 sustained, you're on the edge. |

### Eval panels

These appear at `--eval_interval` cadence. Since eval is expensive, keep the interval ≥500 steps for fine-tunes, ≥2000 for from-scratch.

| Metric | What it measures | Target |
|---|---|---|
| `eval/recentPDB/lDDT` | Overall structure quality on held-out recent PDB | 0.75+ for v1-size after stage 1; 0.78+ after stage 3 |
| `eval/recentPDB/DockQ` | Complex-interface quality | 0.35+ is solid |
| `eval/recentPDB/lDDT_chain` | Per-chain quality | Usually tracks global lDDT |
| `eval/posebusters/success_rate` | PoseBusters pass rate | 60%+ stage 1, 75%+ after stage 2 with alpha_bond=1 |
| `eval/posebusters/rmsd_below_2A` | Ligand pose accuracy | 30%+ stage 1, 50%+ after stage 2 |
| `eval/posebusters/rmsd_below_5A` | Easier ligand accuracy tier | 55%+ stage 1, 70%+ after stage 2 |

Exact metric names depend on `protenix/utils/metrics.py` — grep there if WandB panel names differ from the above.

### Throughput and system

| Metric | What it tells you |
|---|---|
| `step_time_s` | Seconds per step. Compare against the table in [`01-compute-budget.md`](./01-compute-budget.md). |
| `samples_per_second` | For multi-GPU runs, verifies good scaling. |
| `gpu_mem_peak_gb` | Should stabilize; sudden increase = bigger crop reached or memory leak. |

## When to stop a stage

There's no automated early-stopping in Protenix training — you make the call.

**Stage 1 (initial training, from scratch)**:
- Stop when `eval/recentPDB/lDDT` has been flat for 5–10 eval intervals. Typically around step 80k–100k for v2-sized models.
- If you're compute-constrained, stopping at step 50k still gives you a usable (if underdeveloped) model.

**Stages 2–3 (ongoing fine-tuning)**:
- These are shorter runs (20k and 10k steps respectively). Monitor PoseBusters rate — should climb measurably during stage 2. If it's flat after 5k steps of stage 2, your bond loss isn't engaged (check weights).

**Stage 4 (confidence)**:
- Monitor `eval/recentPDB/PAE_correlation` or equivalent calibration metric. This climbs steadily for ~5k steps then plateaus.

**Fine-tunes**:
- Evaluate on your target metric every 500 steps. Stop when target metric best-value hasn't improved in 3 evals.
- Watch generic `eval/recentPDB/lDDT` as a sanity check — if it's dropping fast, you're over-fitting; stop.

## What a divergence looks like

Early warning signs, in rough order of severity:

1. **`grad_norm` spikes**: individual batches producing large gradients. Gradient clipping usually keeps this contained; if clipping is being hit on every step, LR is too high.
2. **Loss spike then recovery**: usually a bad batch. One spike is fine; 3+ per 1000 steps is a problem.
3. **Loss NaN**: run is dead. Restart from last checkpoint with lower LR or different kernel. See [`07-troubleshooting.md`](./07-troubleshooting.md).
4. **Eval regressing while train loss improves**: classic overfit or eval set drift. For from-scratch runs, this shouldn't happen because the training set is huge. For fine-tunes it's expected past a certain point — stop at the eval peak.
5. **Eval frozen while train loss improves**: model is fitting noise, not signal. Check data loader is actually yielding varied batches.

## Cost of eval

Each eval pass through `recentPDB_1536_sample384_0925` runs ~384 samples through full inference with `N_step=20` diffusion and `N_cycle=4`. Per-GPU wall-time ~15 minutes at stage 1, ~30 minutes at stage 3.

This is why `--eval_interval` matters. Examples:

- At stage 1, `--eval_interval 2000` costs ~15min / (2000 × 12s) = ~4% overhead. Fine.
- At stage 3, `--eval_interval 500` costs ~30min / (500 × 44s) = ~8% overhead. Tolerable.
- `--eval_interval 100` on stage 3 = ~40% overhead. Too much.

## Sanity evals: don't trust eval numbers blindly

The eval metrics above are noisy, especially on small eval sets. Before celebrating a 2% bump in lDDT, verify:

1. **Evaluate with more samples**. `recentPDB_1536_sample384_0925` samples 384 structures per eval; this has enough variance that a 1–2% lDDT change can be noise.
2. **Evaluate with more diffusion steps**. During training eval, `sample_diffusion.N_step=20` is the fast approximation. For trustworthy final numbers, run a separate inference pass with `N_step=200`.
3. **Compare to same-seed baseline**. If in doubt, re-evaluate your *starting* checkpoint with the same eval protocol. If the difference against your fine-tuned model is within noise, you haven't improved anything.

## Custom metrics

For fine-tunes with a specific objective, add your domain metric to the eval pass. Easiest place to do this is in `protenix/utils/metrics.py` — add a new metric function, wire it into the eval loop via `runner/train.py`, and it'll appear in WandB.

Domain metrics worth adding:

- **Protein–ligand RMSD, specific to your HET codes**: filter PoseBusters-style metrics to the ligand types you care about.
- **Interface RMSD** for a specific chain pair: useful for antibody–antigen or protein–peptide.
- **Per-family breakdown**: stratify lDDT by your subset membership to see family-specific improvement.

## Recording the run for future reference

After a run completes, write a manifest to `$BASE_DIR/manifest.md`:

```markdown
# Run: v2_stage1 (protenix-v2 stage 1)

- Date: 2026-04-20 to 2026-05-02
- Git SHA: abc123def
- Data version: 2026.01.01
- Final step: 98000
- Final eval/recentPDB/lDDT: 0.771
- Final eval/posebusters/success_rate: 0.62
- Total GPU-hours: 1124 (64 GPUs × 17.5h × 1.0 util)
- Checkpoint SHA-256: ...
```

Six months from now you'll want this.

## Shutting down cleanly

At the end of a run, archive the best checkpoint(s) and drop the rest:

```bash
# Keep only best-by-eval and final
cd output/v2_stage1/checkpoint
ls *.pt | grep -v -E "^(step_98000|ema_98000|best_lDDT|best_lDDT_ema)\\.pt$" | xargs rm
```

Checkpoint directory can grow to 100s of GB — clean it before it becomes a problem.
