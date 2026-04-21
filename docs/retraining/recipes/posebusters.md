# Recipe: PoseBusters Benchmark Optimization

Fine-tune `protenix_base_default_v1.0.0` to maximize PoseBusters benchmark score.

## What PoseBusters measures

PoseBusters is a suite of chemical-validity checks applied to predicted protein–ligand complexes. A prediction "passes" if it satisfies **all** of:

- Reasonable bond lengths (within ~10% of ideal).
- Reasonable bond angles.
- No atom clashes (no pairs within 0.75× sum-of-vdW-radii).
- Chirality preserved relative to the reference ligand.
- Ring flatness/pyramidalization within expected ranges.
- RMSD to crystal pose below a threshold (typically 2Å).

So "beating PoseBusters" is really two things: **structurally accurate** AND **chemically valid**. Accuracy alone isn't enough — a structure can be the right shape but with invalid bond geometry.

This recipe pushes on both.

## What it costs you

- More compute than a generic fine-tune (need larger crops for full binding context).
- Slightly worse generic lDDT on non-ligand structures (because you weight ligands heavily).
- Longer run (PoseBusters validity takes many steps to really improve).

## Step 1 — Training subset

Unlike the [ligand docking recipe](./ligand-docking.md), here you want training data that **specifically has chemically plausible ligand geometry** — so don't just filter by "has a HET group". Prefer:

- X-ray structures with resolution ≤ 2.5Å (high-resolution ligand geometry is reliable).
- HET codes that are drug-like (molecular weight 150–700, no metals, no modified residues).
- Structures where the ligand has ≥ 8 heavy atoms (eliminates buffer ions mistaken for ligands).

Filtering script (sketch):

```bash
# Start with ligand-containing PDBs (from the ligand-docking recipe)
# Then filter by resolution and HET code
python3 scripts/filter_posebusters_training.py \
  --input ligand_containing_pdbs.txt \
  --max_resolution 2.5 \
  --min_ligand_heavy_atoms 8 \
  --exclude_het_codes HOH,SO4,PO4,NA,CL,K,MG,CA,ZN,FE,CU,MN \
  --output pb_training_subset.txt

# Intersect with Protenix training indices
comm -12 pb_training_subset.txt \
  <(zcat $PROTENIX_ROOT_DIR/indices/weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz | cut -d, -f1 | sort -u) \
  > pb_train.txt
```

(`scripts/filter_posebusters_training.py` is something you'd write — parse PDB metadata or the local mmCIF files to check resolution and HET code.)

**Crucial**: exclude all PDB IDs that appear in the PoseBusters eval set. They're in `posebusters_indices_mainchain_interface.csv` in your indices directory:

```bash
cut -d, -f1 $PROTENIX_ROOT_DIR/indices/posebusters_indices_mainchain_interface.csv \
  | sort -u > posebusters_eval_ids.txt
comm -23 pb_train.txt posebusters_eval_ids.txt > pb_train_clean.txt
```

Your subset should be ~20k–40k entries. Too small = under-training; too large = losing signal specificity.

## Step 2 — The two-phase fine-tune

PoseBusters optimization benefits from the stage-2/stage-3-style recipe adapted to fine-tuning. Run two phases:

### Phase A — Structure + bond geometry

```bash
#!/bin/bash
# recipes_posebusters_phase_a.sh
set -euo pipefail
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PROTENIX_ROOT_DIR=/path/to/protenix_data

ckpt="$PROTENIX_ROOT_DIR/checkpoint/protenix_base_default_v1.0.0.pt"

torchrun --nproc_per_node=8 runner/train.py \
  --model_name protenix_base_default_v1.0.0 \
  --run_name ft_posebusters_phaseA \
  --seed 42 \
  --base_dir ./output/ft_posebusters_phaseA \
  --dtype bf16 \
  --project protenix_pb \
  --use_wandb true \
  --load_checkpoint_path "$ckpt" \
  --load_ema_checkpoint_path "$ckpt" \
  --diffusion_batch_size 24 \
  --train_crop_size 640 \
  --max_steps 25000 \
  --warmup_steps 1000 \
  --lr 5e-5 \
  --ema_decay 0.999 \
  --model.N_cycle 4 \
  --sample_diffusion.N_step 20 \
  --eval_interval 500 \
  --checkpoint_interval 500 \
  --log_interval 50 \
  --triangle_attention cuequivariance \
  --triangle_multiplicative cuequivariance \
  --loss.weight.alpha_pae 0 \
  --loss.weight.alpha_bond 1.0 \
  --loss.weight.smooth_lddt 0 \
  --loss.weight.alpha_confidence 1e-4 \
  --loss.weight.alpha_diffusion 4.0 \
  --loss.weight.alpha_distogram 0.03 \
  --loss.diffusion.mse.weight_ligand 20.0 \
  --data.train_sets weightedPDB_before2109_wopb_nometalc_0925 \
  --data.weightedPDB_before2109_wopb_nometalc_0925.base_info.pdb_list ./pb_train_clean.txt \
  --data.test_sets recentPDB_1536_sample384_0925,posebusters_0925 \
  --data.posebusters_0925.base_info.max_n_token 768
```

Key choices:

- **`--train_crop_size 640`** — big enough for most full binding-site contexts.
- **`--diffusion_batch_size 24`** — fits in 80GB with crop 640.
- **`--loss.weight.alpha_bond 1.0`** — the key PB-validity knob.
- **`--loss.weight.smooth_lddt 0`** — stage-2 pattern.
- **`--loss.diffusion.mse.weight_ligand 20.0`** — modest up-weighting.
- **`--max_steps 25000`** — long enough for bond loss to really improve validity.

Budget: ~9 wall-clock days on 8× A100-80G. If you can't afford that, reduce to crop 384 + 15k steps, which runs in ~2 days — you'll lose a few points on large complexes but still see good PB improvement.

### Phase B — Confidence recalibration (stage-4 style)

```bash
ckpt=./output/ft_posebusters_phaseA/checkpoint/last.pt
ema=./output/ft_posebusters_phaseA/checkpoint/ema.pt

torchrun --nproc_per_node=8 runner/train.py \
  [same flags as Phase A, except:] \
  --load_checkpoint_path "$ckpt" \
  --load_ema_checkpoint_path "$ema" \
  --run_name ft_posebusters_phaseB \
  --base_dir ./output/ft_posebusters_phaseB \
  --train_confidence_only true \
  --max_steps 7000 \
  --lr 3e-4 \
  --loss.weight.alpha_pae 1.0 \
  --loss.weight.alpha_bond 0 \
  --loss.weight.alpha_diffusion 0 \
  --loss.weight.alpha_distogram 0
```

~1.5 days on 8× A100-80G. Fixes PAE/pLDDT calibration on the fine-tuned model.

## Step 3 — Inference at high quality

For the final PoseBusters evaluation, use inference settings tuned for accuracy rather than speed:

```bash
for pdb in $(cat $PROTENIX_ROOT_DIR/indices/posebusters_eval_ids.txt); do
    protenix pred \
      -i "./posebusters_inputs/${pdb}.json" \
      -o "./pb_eval_output/${pdb}" \
      -n protenix_base_default_v1.0.0 \
      --load_checkpoint_path ./output/ft_posebusters_phaseB/checkpoint/last.pt \
      --use_default_params false \
      --use_template true \
      --model.N_cycle 10 \
      --sample_diffusion.N_step 200 \
      --sample_diffusion.N_sample 25
done
```

Key inference flags:

- `--model.N_cycle 10` (not 4) — more refinement cycles at inference.
- `--sample_diffusion.N_step 200` (not 20) — full diffusion schedule.
- `--sample_diffusion.N_sample 25` — generate many candidates, pick best by confidence.

Run the PoseBusters validation suite on the top-confidence prediction for each target.

## Expected improvement

Baseline v1.0.0 PoseBusters numbers (from `../model_1.0.0_benchmark.md`, approximately):

| Metric | Baseline v1.0.0 | After this recipe |
|---|---|---|
| PB overall pass rate | ~73% | 82–87% |
| Bond-length validity | ~92% | 97–99% |
| Clash-free | ~85% | 92–95% |
| Chirality preserved | ~96% | 97–98% |
| Ligand RMSD < 2Å | ~50% | 60–68% |

Gains depend on training subset quality. If your subset is smaller or less clean, expect the lower end of each range.

## Known failure modes

### "My PB pass rate dropped during fine-tuning"

Usually means the bond loss is over-tightening structure in early training, producing clashes elsewhere. Mitigations:

- Longer warmup: `--warmup_steps 2000`.
- Lower `alpha_bond` initially: start at 0.3, raise to 1.0 after 5k steps.

### "Validity is good but RMSD is bad"

Bond loss is doing its job, but structural loss isn't keeping up. Raise `weight_ligand` higher (25–30) or train longer.

### "RMSD is good but validity is bad"

Opposite — need more bond loss. Raise `alpha_bond` to 2.0 or extend Phase A.

### "Confidence metrics are worse after Phase B"

You probably forgot to load the Phase A checkpoint into Phase B. Verify `--load_checkpoint_path` points at the Phase A output, not back at v1.0.0.

## Combining with covalent-aware methods

For covalent inhibitors, PoseBusters has a separate scoring track. You'll want:

- Include covalent-ligand-containing PDBs in your training subset (they're often filtered out by "drug-like" criteria, so add them back manually).
- Increase `--loss.weight.alpha_bond` to 2.0 — covalent bonds between ligand and protein need accurate geometry.
- Use the constraint model (`protenix_base_constraint_v0.5.0`) as your starting checkpoint instead of v1.0.0 if you need explicit covalent bond constraints.
