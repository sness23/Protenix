# Data Preparation

Training and fine-tuning both require the preprocessed wwPDB training data. This page walks through what to download, what it contains, how the on-disk layout maps to config names, and how to build your own subset or dataset if you need one.

Everything here assumes `PROTENIX_ROOT_DIR` is set to the directory where you want the data to live:

```bash
export PROTENIX_ROOT_DIR=/path/to/protenix_data
mkdir -p "$PROTENIX_ROOT_DIR"
```

If you skip this step, `PROTENIX_ROOT_DIR` defaults to `$HOME`, which you almost certainly don't want for a 1.5 TB dataset.

## Which data version do you want?

The download script in `scripts/database/download_protenix_data.sh` supports two data versions:

| `--version` | wwPDB cutoff | Use for |
|---|---|---|
| `2024.05.22` (default) | 2021-09-30 | AF3-equivalent `protenix_base_default_v1.0.0`; cleanest comparison to published benchmarks |
| `2026.01.01` | 2025-06-30 | `protenix_base_20250630_v1.0.0`; broader/newer PDB coverage for practical use |

The `2026.01.01` version is what you'd want for training v2 from scratch — it's the most recent supported snapshot. The `2024.05.22` version is what you'd want for methodological comparisons against published v1.0.0 numbers.

**For path A (from-scratch v2)**: use `2026.01.01`. For path B (fine-tuning), match the version to whichever checkpoint you're fine-tuning from.

## The download command

```bash
# Full download (training + inference) — ~1.5 TB
bash scripts/database/download_protenix_data.sh --full --version 2026.01.01

# Inference-only (much smaller) — for reproducing eval only
bash scripts/database/download_protenix_data.sh --inference_only --version 2026.01.01
```

The script is idempotent on already-present files — if it's interrupted, just re-run. For large downloads, consider running inside `tmux` or with `nohup`.

## On-disk layout (after `--full`)

From `../training_inference_instructions.md:151-181`:

```
$PROTENIX_ROOT_DIR/
├── common/                             # shared metadata
│   ├── clusters-by-entity-40.txt       # 40% identity clusters for antibody sampling
│   ├── components.cif                  # CCD source definitions
│   ├── components.cif.rdkit_mol.pkl    # RDKit Mol cache
│   ├── obsolete_release_date.csv
│   ├── obsolete_to_successor.json
│   ├── release_date_cache.json
│   └── seq_to_pdb_index.json           # sequence → PDB entity lookup
├── indices/                            # train/eval index files
│   ├── posebusters_indices_mainchain_interface.csv
│   ├── recentPDB_low_homology_maxtoken1024_sample384_pdb_id.txt
│   ├── recentPDB_low_homology_maxtoken1536.csv
│   └── weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz
├── mmcif/                              # raw mmCIF + template search database
├── mmcif_bioassembly/                  # preprocessed wwPDB training data (.pkl.gz)
├── mmcif_msa_template/                 # per-chain MSA and template features
├── posebusters_bioassembly/
├── posebusters_mmcif/
├── recentPDB_bioassembly/
├── rna_msa/
│   ├── msas/
│   └── rna_sequence_to_pdb_chains.json
└── search_database/
    ├── nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta
    ├── pdb_seqres_2022_09_28.fasta
    ├── rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta
    └── rnacentral_active_seq_id_90_cov_80_linclust.fasta
```

Do not rename or reorganize these directories — the data loader uses the top-level names as defaults (see `configs/configs_data.py:209-235` for example `mmcif_dir` / `bioassembly_dir` paths).

**Disk pressure note**: `mmcif_bioassembly/` is the largest single directory (~800 GB). If you're tight on space you can partially download by commenting out specific sections of `download_protenix_data.sh` — but any dataset you reference in `--data.train_sets` needs its full bioassembly directory present.

## Mapping configs to on-disk datasets

Dataset names that appear in training CLI flags (`--data.train_sets`, `--data.test_sets`) are registered in `configs/configs_data.py:128`. The training-relevant names:

| Config name | What it is |
|---|---|
| `weightedPDB_before2109_wopb_nometalc_0925` | Full wwPDB training set, cutoff 2021-09-30, excluding PoseBusters structures and metal-containing. Compatible with data version `2024.05.22`. |
| `weightedPDB_before250701_v20260101` | wwPDB training set with 2025-07-01 cutoff. Compatible with data version `2026.01.01`. |
| `weightedPDB_before210930_v20260101` | 2021-09-30 cutoff but using the `2026.01.01` preprocessing. Use this if you want the v1.0.0 data cutoff but with newer MSA/template features. |
| `recentPDB_1536_sample384_0925` | Standard eval set — recent low-homology PDB entries, sampled to 1536 token max. |
| `posebusters_0925` | PoseBusters eval set. |

**Important**: you must pick a training set whose suffix (`_0925` vs `_v20260101`) matches the data version you downloaded. The pipeline doesn't auto-detect this; you'll get a file-not-found on the first batch otherwise.

## Configuring custom dataset paths

If your data lives somewhere other than `$PROTENIX_ROOT_DIR`, override the paths in the config:

```bash
python3 runner/train.py \
  --data.weightedPDB_before250701_v20260101.base_info.mmcif_bioassembly_dir /custom/path/to/bioassembly \
  --data.weightedPDB_before250701_v20260101.base_info.msa_dir /custom/path/to/msa \
  ...
```

The per-dataset `base_info` dict contains `mmcif_dir`, `bioassembly_dir`, `indices_fpath`, `msa_dir`, `template_dir`. Grep for them in `configs/configs_data.py`.

## Building a PDB subset for fine-tuning

Fine-tuning lets you restrict training to a specific list of PDB IDs via `--data.<dataset>.base_info.pdb_list`. The list file is plain text, one PDB ID per line, lowercase:

```text
6hvq
5mqc
5zin
3erk
1m17
```

The file needs to be accessible at runtime (relative paths are resolved from the run directory). Conventional place to put it is `examples/finetune_subset.txt` or next to your launch script.

**How to build a good subset**:

1. **Start with a canonical source**. For a protein family: PDB advanced search, or pull from UniProt's family cross-refs. For a ligand class: filter PDB by HET code.
2. **Intersect with what's in the training data**. Your subset must be a subset of the IDs that appear in `indices/weightedPDB_*`. Otherwise the data loader skips them silently:
   ```bash
   zcat $PROTENIX_ROOT_DIR/indices/weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz \
     | cut -d, -f1 | sort -u > available_pdbs.txt
   comm -12 <(sort my_family.txt) available_pdbs.txt > my_subset.txt
   ```
3. **Hold out an eval split**. Reserve 10–20% of your subset for evaluation and exclude it from `pdb_list` — otherwise your eval numbers are meaningless.
4. **Check homology overlap with eval sets**. The shipped `recentPDB_1536_sample384_0925` is chosen to be low-homology against pre-2021 PDB. If your subset overlaps heavily with it at sequence level, evaluation signal is compromised.

## Preparing your own CIF files as training data

If you have structures that aren't in wwPDB (predictions, in-house structures, etc.), see `../prepare_training_data.md` for the full pipeline. Summary:

1. **Put CIF files in a folder** or list them in a text file.
2. **Update the CCD cache** if your CIFs use HET codes newer than 2024-06-08:
   ```bash
   python3 scripts/gen_ccd_cache.py -c $PROTENIX_ROOT_DIR/common -n 8
   ```
3. **Run the preprocessing script**:
   ```bash
   python3 scripts/prepare_training_data.py \
     -i /path/to/cifs_or_list.txt \
     -o /path/to/output.csv \
     -b /path/to/bioassembly_output/ \
     -c $PROTENIX_ROOT_DIR/common/clusters-by-entity-40.txt \
     -n 16
   ```
4. **Generate MSA and template features** for the new chains:
   - Proteins: see `../msa_template_pipeline.md`. You'll run either ColabFold-style server-mode MSA or local HMMER search, then template search, per sequence.
   - RNA: use `runner/rna_msa_search.py`.
5. **Register the new dataset** in `configs/configs_data.py` by adding an entry that points `base_info` at your directories.

This pipeline is slow — budget a day of wall-time per 1000 structures on a 16-core machine. MSA search is the bottleneck.

## CCD cache staleness

The released `components.cif.rdkit_mol.pkl` is cut at 2024-06-08. If your training set includes PDB entries newer than that with novel HET codes (ligands, modified residues), you'll get warnings or silent drops. Regenerate with:

```bash
python3 scripts/gen_ccd_cache.py -c $PROTENIX_ROOT_DIR/common -n 16
```

This takes ~30 minutes and downloads the latest CCD from RCSB. Safe to run periodically.

## Validating your data setup

Before launching a long training run, sanity-check with one step:

```bash
python3 runner/train.py \
  --model_name protenix_base_default_v1.0.0 \
  --max_steps 1 \
  --eval_interval 1 \
  --train_crop_size 256 \
  --diffusion_batch_size 4 \
  --data.train_sets weightedPDB_before250701_v20260101 \
  --data.test_sets recentPDB_1536_sample384_0925 \
  --use_wandb false
```

If this completes without data-loader errors, your paths are wired up. If it fails:
- **`FileNotFoundError: .../mmcif_bioassembly/...`** — dataset version mismatch. Check `--data.train_sets` suffix matches your `--version` download.
- **`KeyError: 'components.cif.rdkit_mol.pkl'`** — CCD cache missing; re-run the download or regenerate.
- **`IndexError` or empty batch** — your `pdb_list` filter excludes everything in the dataset; widen it or drop the filter temporarily.

Once that one-step run passes, you can proceed to [`03-from-scratch-v2.md`](./03-from-scratch-v2.md) or [`04-finetuning-v1.md`](./04-finetuning-v1.md).
