# Recipe: Protein–Ligand Docking

Fine-tune `protenix_base_default_v1.0.0` to maximize protein–ligand pose quality. This is the recipe you want if your use case is docking — CASP ligand targets, internal compound screens, PoseBusters-style benchmarks.

This recipe is tuned for the `/cdock` / DynamicBind-style workflow in the user's docking pipeline (see `~/data/vaults/casp/` references), but generalizes to any ligand-focused problem.

## What this recipe optimizes for

- **Ligand RMSD ≤ 2Å** against crystal poses — the primary metric.
- **PoseBusters chemical validity** — no clashing atoms, reasonable bond geometry.
- **Confidence calibration on ligand atoms** specifically, so you can rank poses.

What it trades off:

- Slight loss on protein-only lDDT vs. the released v1.0.0.
- Less well-calibrated confidence on non-ligand-containing complexes.

## Step 1 — Build a ligand-focused subset

Restrict training to PDB entries that *contain ligands*. You can generate this list from the PDB API:

```bash
# Pull PDB IDs containing small-molecule ligands (HET codes not in stdres/water/metals)
curl -s 'https://search.rcsb.org/rcsbsearch/v2/query' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {
      "type": "terminal",
      "service": "text",
      "parameters": {
        "attribute": "rcsb_entry_info.deposited_nonpolymer_entity_instance_count",
        "operator": "greater",
        "value": 0
      }
    },
    "return_type": "entry",
    "request_options": {"paginate": {"start": 0, "rows": 100000}}
  }' \
  | jq -r '.result_set[].identifier' \
  | tr '[:upper:]' '[:lower:]' \
  | sort -u > ligand_containing_pdbs.txt

# Intersect with what's actually in the Protenix training indices
zcat $PROTENIX_ROOT_DIR/indices/weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz \
  | cut -d, -f1 | sort -u > available.txt
comm -12 ligand_containing_pdbs.txt available.txt > ligand_subset.txt
wc -l ligand_subset.txt  # expect 30k–60k entries
```

If you want to further narrow to a specific ligand class (kinase inhibitors, peptide ligands, fragments), filter by HET code or by deposited ligand SMILES.

## Step 2 — Hold out a clean eval set

Do **not** train on PoseBusters structures — they're already in the Protenix eval set. Also hold out 200–500 of your own entries for domain-specific eval:

```bash
shuf ligand_subset.txt > shuffled.txt
head -n 300 shuffled.txt > ligand_eval.txt
tail -n +301 shuffled.txt > ligand_train.txt
```

## Step 3 — Fine-tune with ligand-weighted loss

Key settings:

- `--loss.diffusion.mse.weight_ligand 25.0` (up from default 10.0) — ligand atoms dominate the MSE.
- `--loss.weight.alpha_bond 1.0` — turn on bond geometry loss. Essential for PoseBusters validity.
- `--loss.weight.smooth_lddt 0` — turn off, since we're in stage-2 mode.
- `--train_crop_size 640` — ligand binding sites + full protein context.
- `--lr 5e-5` — gentle fine-tune.

```bash
#!/bin/bash
# recipes_ligand_docking.sh
set -euo pipefail
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PROTENIX_ROOT_DIR=/path/to/protenix_data

ckpt="$PROTENIX_ROOT_DIR/checkpoint/protenix_base_default_v1.0.0.pt"

torchrun --nproc_per_node=8 runner/train.py \
  --model_name protenix_base_default_v1.0.0 \
  --run_name ft_ligand_docking \
  --seed 42 \
  --base_dir ./output/ft_ligand_docking \
  --dtype bf16 \
  --project protenix_ft \
  --use_wandb true \
  --load_checkpoint_path "$ckpt" \
  --load_ema_checkpoint_path "$ckpt" \
  --diffusion_batch_size 32 \
  --train_crop_size 640 \
  --max_steps 20000 \
  --warmup_steps 500 \
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
  --loss.diffusion.mse.weight_ligand 25.0 \
  --data.train_sets weightedPDB_before2109_wopb_nometalc_0925 \
  --data.weightedPDB_before2109_wopb_nometalc_0925.base_info.pdb_list ./ligand_train.txt \
  --data.test_sets recentPDB_1536_sample384_0925,posebusters_0925
```

Budget: 20k steps at ~30 s/step on 8× A100-80G = ~5 wall-clock days. If that's too long, drop crop to 384 and steps to 10k — you lose some quality on large complexes but finish in under 2 days.

## Step 4 — Confidence recalibration

After the main fine-tune, run a short stage-4-style pass to fix confidence calibration on ligands. See [`confidence-only.md`](./confidence-only.md) for the full recipe; the short version:

```bash
# Load the output of step 3 as the starting checkpoint
ckpt=./output/ft_ligand_docking/checkpoint/last.pt
ema=./output/ft_ligand_docking/checkpoint/ema.pt

torchrun --nproc_per_node=8 runner/train.py \
  [same flags as above, except:] \
  --load_checkpoint_path "$ckpt" \
  --load_ema_checkpoint_path "$ema" \
  --run_name ft_ligand_docking_conf \
  --base_dir ./output/ft_ligand_docking_conf \
  --train_confidence_only true \
  --max_steps 5000 \
  --lr 3e-4 \
  --loss.weight.alpha_pae 1.0 \
  --loss.weight.alpha_diffusion 0 \
  --loss.weight.alpha_distogram 0 \
  --loss.weight.alpha_bond 0
```

~1 day on 8× A100-80G.

## Step 5 — Evaluate

After both steps, run inference on your held-out ligand eval set AND on PoseBusters:

```bash
# Your held-out set
for pdb in $(cat ligand_eval.txt); do
    protenix pred \
      -i "./input_jsons/${pdb}.json" \
      -o "./eval_output/${pdb}" \
      -n protenix_base_default_v1.0.0 \
      --load_checkpoint_path ./output/ft_ligand_docking_conf/checkpoint/last.pt \
      --use_default_params true \
      --use_template true
done

# Aggregate RMSD vs. ground truth
python3 scripts/aggregate_ligand_rmsd.py \
  --pred_dir ./eval_output \
  --truth_dir ./input_jsons \
  --out ./eval_output/rmsd_summary.csv
```

(The `aggregate_ligand_rmsd.py` script is one you'll need to write — it parses predicted and ground-truth ligand coordinates and computes symmetry-corrected RMSD. Use RDKit's `GetBestRMS` or `rdkit.Chem.AllChem.GetBestRMS`.)

Expected results on a well-curated subset:

| Metric | Released v1.0.0 | After this recipe |
|---|---|---|
| PoseBusters pass rate | ~73% | 80–85% |
| Ligand RMSD < 2Å | ~50% | 60–68% |
| Ligand RMSD < 5Å | ~73% | 80–85% |
| PAE correlation on ligand atoms | 0.45 | 0.58–0.65 |

Exact gains depend heavily on how well your training subset covers the eval distribution.

## Common gotchas

- **Over-weighting ligand MSE**: at `weight_ligand > 40`, the model starts over-fitting ligand positions at the cost of protein backbone accuracy. Stay in the 20–30 range.
- **Too-short fine-tune**: 10k steps isn't enough for the ligand-weighted loss to really bite. Either commit to 20k+ or lower `weight_ligand` to 15.
- **Forgetting to filter training subset**: if you accidentally include PoseBusters structures in your training subset (they won't automatically be excluded by `--data.train_sets weightedPDB_before2109_wopb_nometalc_0925` if you've replaced the default), your PoseBusters eval numbers will be meaningless.
- **Mixing data versions**: if you're using `weightedPDB_before250701_v20260101` instead of `weightedPDB_before2109_wopb_nometalc_0925`, update the `pdb_list` flag name accordingly.

## Variations

### Covalent ligands

If your target class is covalent inhibitors (e.g. some cysteine protease ligands), also bump:

```bash
--loss.weight.alpha_bond 2.0  # from 1.0
```

Covalent geometry is even more dependent on bond-length accuracy.

### Metal-coordinated ligands

The default training subset excludes metal-containing structures (`nometalc` in the name). If you want to handle metals (Zn, Mg coordination), switch to a training set that includes them — but you'll need to generate it yourself or find one in the configs.

### Fragments (small ligands, <20 heavy atoms)

Fragments are harder because there are fewer atoms for the loss to constrain. Consider:

```bash
--loss.diffusion.mse.weight_ligand 35.0   # higher
--train_crop_size 384                     # smaller is OK, fragment binding sites are local
--max_steps 30000                         # need more steps because signal is weaker per structure
```
