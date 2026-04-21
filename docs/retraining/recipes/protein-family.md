# Recipe: Narrowing to a Protein Family

Fine-tune `protenix_base_default_v1.0.0` to perform exceptionally well on a specific protein family — kinases, GPCRs, antibodies, serine proteases, etc. — at the cost of some generalization.

This is the most common form of fine-tune in practice: you have a project focused on one family, and you'd rather have a specialist model than a generalist.

## When to use this vs. the ligand recipe

- **This recipe**: when the family itself is what you care about (e.g. antibody structures, regardless of what's bound).
- **[Ligand recipe](./ligand-docking.md)**: when the protein doesn't matter much but ligand quality does.
- **Combined**: if you want "kinase inhibitors specifically", start with this recipe, then follow with the ligand recipe — or merge loss weights.

## Pick your family

Common targets and example approaches:

| Family | PDB filter heuristic | Typical subset size |
|---|---|---|
| Kinases | InterPro IPR000719 (protein kinase domain); UniProt family cross-refs | 3k–5k PDB entries |
| GPCRs | GPCRdb.org releases a curated PDB list; or filter by Pfam 7tm_1/7tm_2/7tm_3 | 300–800 entries |
| Antibodies | SAbDab (http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/) | 7k–10k entries |
| Serine proteases | Pfam PF00089 (trypsin-like) | 1k–2k entries |
| Nuclear receptors | NuclearDB or Pfam PF00104 | 500–1000 entries |
| Viral glycoproteins | PDB advanced search on virus taxonomy | 1k–3k entries |
| Ubiquitin ligases | Pfam PF00569, PF00097, PF13923 (multiple zinc finger families) | 1k+ entries |

The subset-size sweet spot is **500–5000 entries**. Too few → the model overfits and eval numbers mean nothing. Too many → you're barely narrowing and the fine-tune doesn't buy much.

## Step 1 — Build the subset

Let's use kinases as a worked example. First, get a canonical list:

```bash
# Option A: from Pfam via InterPro
curl -s 'https://www.ebi.ac.uk/interpro/api/entry/pfam/PF00069/protein/reviewed/?page_size=200' \
  | jq -r '.results[].metadata.accession' > kinase_uniprots.txt

# Map UniProt → PDB via UniProt REST (use your favorite approach)
# Easiest: download SIFTS pdb_chain_uniprot.tsv from EBI and join
wget https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_uniprot.tsv.gz
zcat pdb_chain_uniprot.tsv.gz \
  | awk 'NR>2 {print tolower($1)"\t"$4}' \
  | sort -u \
  | awk -v f=kinase_uniprots.txt 'BEGIN{while((getline l < f)>0)uni[l]=1} uni[$2]' \
  | cut -f1 | sort -u > kinase_pdbs.txt
```

Then intersect with the Protenix training set:

```bash
zcat $PROTENIX_ROOT_DIR/indices/weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz \
  | cut -d, -f1 | sort -u > available.txt
comm -12 kinase_pdbs.txt available.txt > kinase_subset.txt
wc -l kinase_subset.txt
```

Hold out ~10–20% for eval:

```bash
shuf kinase_subset.txt > shuffled.txt
head -n 300 shuffled.txt > kinase_eval.txt
tail -n +301 shuffled.txt > kinase_train.txt
```

## Step 2 — Check for homology overlap with recentPDB eval

The generic eval set `recentPDB_1536_sample384_0925` is pre-filtered for low homology against training data. But if your *family subset* is very large within recentPDB, you're implicitly testing on similar structures.

Quick check (rough): how many of your family IDs appear in the recentPDB index?

```bash
comm -12 kinase_subset.txt <(cut -d, -f1 $PROTENIX_ROOT_DIR/indices/recentPDB_low_homology_maxtoken1536.csv | sort -u)
```

If there's heavy overlap, your generic eval signal will be biased. Either:

1. Rely on your family-specific eval set (`kinase_eval.txt`) instead.
2. Build a custom low-homology eval by clustering your family at 30% sequence identity and holding out cluster representatives.

## Step 3 — Fine-tune

For a moderate-size family (1k–3k PDBs), this config is a reasonable starting point:

```bash
#!/bin/bash
# recipes_protein_family.sh
set -euo pipefail
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PROTENIX_ROOT_DIR=/path/to/protenix_data

ckpt="$PROTENIX_ROOT_DIR/checkpoint/protenix_base_default_v1.0.0.pt"

torchrun --nproc_per_node=8 runner/train.py \
  --model_name protenix_base_default_v1.0.0 \
  --run_name ft_kinases \
  --seed 42 \
  --base_dir ./output/ft_kinases \
  --dtype bf16 \
  --project protenix_ft \
  --use_wandb true \
  --load_checkpoint_path "$ckpt" \
  --load_ema_checkpoint_path "$ckpt" \
  --diffusion_batch_size 32 \
  --train_crop_size 384 \
  --max_steps 15000 \
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
  --loss.weight.alpha_bond 0 \
  --loss.weight.smooth_lddt 1.0 \
  --loss.weight.alpha_confidence 1e-4 \
  --loss.weight.alpha_diffusion 4.0 \
  --loss.weight.alpha_distogram 0.03 \
  --data.train_sets weightedPDB_before2109_wopb_nometalc_0925 \
  --data.weightedPDB_before2109_wopb_nometalc_0925.base_info.pdb_list ./kinase_train.txt \
  --data.test_sets recentPDB_1536_sample384_0925,posebusters_0925
```

Notes:

- `--max_steps 15000` gives roughly 5 passes through a 2000-entry subset at effective batch ~32 (across 8 GPUs).
- Using stage-1 loss weights because we're doing broad adaptation, not PoseBusters tuning.
- Crop 384 is enough for most single-domain proteins; bump to 512 or 640 if your family tends toward large multi-domain complexes.

Budget: ~2 days on 8× A100-80G.

## Step 4 — Evaluate

```bash
# Eval on family-specific holdout
for pdb in $(cat kinase_eval.txt); do
    protenix pred \
      -i "./input_jsons/${pdb}.json" \
      -o "./eval_output/${pdb}" \
      -n protenix_base_default_v1.0.0 \
      --load_checkpoint_path ./output/ft_kinases/checkpoint/last.pt \
      --use_default_params true \
      --use_template true
done
```

Compute lDDT on your held-out family set. You should see:

- **+2 to +5% lDDT** on the family-specific eval (vs. baseline v1.0.0).
- **-0.5 to -2% lDDT** on generic `recentPDB` eval.

If you see less family improvement or more generic regression, adjust the LR or step count.

## Variations

### Antibodies

Antibodies benefit from crop 640 (because of Fv–antigen complexes) and slightly longer runs:

```bash
--train_crop_size 640 \
--diffusion_batch_size 24 \
--max_steps 25000
```

Also consider training with explicit CDR-focused sampling — the default weighted sampler treats all chains uniformly, but antibody CDR loops are the hard part. See `configs/configs_data.py:45-67` for sampler config.

### Membrane proteins (GPCRs, channels)

Include relevant template structures and use MSAs generated from metagenomic databases (many membrane proteins have sparse close homologs in UniRef). Enable templates explicitly:

```bash
--use_template true \
--model.N_cycle 8
```

The extra cycles help the model use template information more thoroughly.

### Small families (<500 entries)

When your subset is small, risk of overfit is high. Consider:

- **Reduce LR**: `--lr 2e-5`.
- **Reduce steps**: `--max_steps 5000`.
- **Partial freezing**: add `--finetune_params_with_substring confidence_head,pairformer.block_23` to update only late layers.
- **Data augmentation**: use MSA subsampling (`--msa.sample_size` — check the config) to artificially increase diversity.

### Large families (>10k entries)

Treat it as a smaller full fine-tune:

- **Longer run**: `--max_steps 40000`.
- **Optionally higher LR**: `--lr 1e-4` (closer to from-scratch).
- **Keep `alpha_bond` off during main run, add a stage-2 pass later** — same multi-stage logic as Path A but scoped to the family.

## Combining family + ligand

If you want "kinases with good ligand accuracy specifically":

1. Run this family recipe for 10k steps to shift the model toward kinase backbones.
2. Then run the [ligand docking recipe](./ligand-docking.md) for 10k steps with `pdb_list` still restricted to kinases — but now with `weight_ligand=25`, `alpha_bond=1.0`.
3. Optionally, [confidence-only](./confidence-only.md) pass at the end.

Total: ~4 days on 8× A100-80G, and you get a kinase-ligand specialist that outperforms v1.0.0 on both kinase structure and kinase-ligand pose quality.
