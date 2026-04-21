# Troubleshooting

A catalog of common failure modes in Protenix training and fine-tuning, organized by symptom. If your problem doesn't appear here, the first places to dig are:

- `runner/train.py` — main loop, often the clearest source of what a given flag does
- `protenix/utils/torch_utils.py` — AMP / dtype handling
- `protenix/model/layer_norm/` — custom LayerNorm kernel, occasional source of numerical issues

## Out-of-memory (OOM)

### Symptom: OOM on the very first forward pass

Most likely causes:

1. **Crop size too large for your GPU.** Stage 3 (crop 768) + `diffusion_batch_size 32` needs ~48 GB per GPU. On A100-40G you must shrink one.
2. **Kernel choice.** `triattention` is memory-heavier than `cuequivariance`. Try `--triangle_attention cuequivariance --triangle_multiplicative cuequivariance`.
3. **Compile + AMP overhead.** If you're running with `--dtype fp32` on memory-tight hardware, switch to `bf16`.

Remediations, in order of preference:

```bash
# 1. Switch to lighter kernels
--triangle_attention cuequivariance --triangle_multiplicative cuequivariance

# 2. Reduce diffusion batch size
--diffusion_batch_size 16   # from 32

# 3. Reduce crop size (affects recipe fidelity)
--train_crop_size 512       # from 768

# 4. Reduce pairformer blocks (distorts model — last resort)
--model.pairformer.nblocks 36   # from 48
```

Never reduce `N_cycle` below 2 — it breaks recycling assumptions.

### Symptom: OOM partway through training

Training OOMs that appear after N successful steps are usually from:

1. **Variable-size batches.** Some complexes in the dataset are larger than others. If your `max_n_token` / `max_n_atom` filters aren't strict enough, an unusually large batch can blow you up.
   ```bash
   # Cap per-batch size more tightly
   --data.weightedPDB_before250701_v20260101.base_info.max_n_token 768
   --data.weightedPDB_before250701_v20260101.base_info.max_n_atom 6000
   ```

2. **Memory fragmentation.** Sometimes the allocator holds onto peak-size buffers after a large batch. Mitigation: `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`.

3. **Checkpoint eval pass.** Eval uses larger crops than training (via `recentPDB_1536_sample384`, cropped to 1536). If you OOM at eval time but not train time, lower `--data.recentPDB_1536_sample384_0925.base_info.max_n_token` to 1024.

### Symptom: OOM only during inference

`runner/inference.py::update_inference_configs` automatically downgrades precision for large inputs. If you're still OOMing:

- Reduce `sample_diffusion.N_sample` (default 5 → 1 for debugging).
- Set `--dtype bf16` explicitly (should be default but worth verifying).
- Check `N_token` of your failing input — at 4000+ tokens, even H100 struggles.

## NaN losses

### Symptom: loss becomes NaN within the first 1000 steps

Almost always a numerical-stability issue in mixed precision.

1. **Verify `--dtype bf16`, not `fp16`.** Protenix is not tested with fp16; use bf16 or fp32. The `fp16` path exists in the code but is effectively unsupported.
2. **Warmup too short or skipped.** With `--warmup_steps 0` and a high LR, early steps can produce huge gradients. Use `--warmup_steps 2000` for from-scratch, 500+ for fine-tune.
3. **LR too high.** Halve it and restart.
4. **Custom LayerNorm kernel.** Set `LAYERNORM_TYPE=torch` env var and retry. If the NaN disappears, the fast kernel has a numerical edge case with your hardware; file an issue.

### Symptom: NaN appears deep into training (e.g. step 40k+)

Usually not a bug in the kernel; usually a *data* issue:

1. **Corrupt PDB entry.** One entry with bizarre coordinates (e.g. chain with all zeros, clash score in the millions) can produce a garbage loss. The `resolution` filter (default `min: 0.1, max: 4.0`) catches most of these but not all.
2. **Ligand geometry outlier.** A HET code with invalid coordinates in its CCD definition.

Finding the culprit:

```bash
# Log the PDB ID of the batch right before the NaN
# (hack: add a print in the data loader — grep runner/train.py for the iterator)
```

Then add that ID to an exclude list.

### Symptom: loss is NaN at step 0 (before first update)

- Checkpoint-loading order wrong. If `--load_checkpoint_path` is a stage-4 weights file and `--model_name` is the base model, you'll get state-dict drift. Verify `--model_name` matches the checkpoint's origin.
- EMA-only checkpoint loaded as main (or vice versa). Both flags should usually point at the same run's matched pair.

## Speed problems

### Symptom: step time is 2–3× the documented number

Expected per-step times from `../training_inference_instructions.md:252`:

| Stage | s/step on A100-80G |
|---|---|
| 1 | 12 |
| 2 | 30 |
| 3 | 44 |
| 4 | 13 |

If your times are much higher:

1. **Wrong kernel.** `torch` triangle kernels are ~2× slower than `cuequivariance`. Check your flags.
2. **CPU bottleneck.** Data loading can't keep up with GPUs. Increase `--num_workers` (if exposed) or preload more aggressively.
3. **Distributed-training overhead.** `NCCL_DEBUG=INFO` to see if there's communication stalling. Big payoff from proper NCCL topology config on multi-node.
4. **Disk I/O.** If `mmcif_bioassembly/` lives on network-attached storage instead of local NVMe, expect 2–5× slowdown. Move to local disk if possible.
5. **Wrong GPU.** A100-40G is ~0.85× A100-80G throughput. V100 is ~0.3×. Adjust expectations.
6. **`LAYERNORM_TYPE=torch`** is set, disabling the fast LayerNorm kernel. Check your env.

### Symptom: single-node works fine, multi-node slow

- **NCCL topology**. Cross-node NCCL without InfiniBand / NVLink is very slow. Try `NCCL_IB_DISABLE=0`, `NCCL_P2P_DISABLE=0`. On AWS EFA, use the AWS NCCL plugin.
- **Uneven data distribution**. If per-rank batch varies widely, you pay for the slowest rank. Consider a length-bucketed sampler (requires code change).
- **Gradient sync bandwidth**. 464M params × 4 bytes (bf16 gradients accumulated in fp32) = ~2 GB per all-reduce. Test with `torch.distributed.barrier` timing.

## Kernel / compile issues

### Symptom: `cuequivariance` fails to import or build

```bash
pip install --upgrade cuequivariance-torch
# If that fails:
pip install cuequivariance-torch --no-build-isolation
```

- Requires CUDA 12+ and matching PyTorch build. Check `nvcc --version` matches `torch.version.cuda`.
- For DeepSpeed-based kernels, set `CUTLASS_PATH=/opt/cutlass` (needs a manual CUTLASS install).

If you can't get CUDA kernels working, fall back to `--triangle_attention torch --triangle_multiplicative torch`. Slower (~2×) but always works.

### Symptom: `fast_layernorm` kernel crashes

```bash
LAYERNORM_TYPE=torch python3 runner/train.py [...]
```

Run with the stock PyTorch LayerNorm to confirm the issue is the kernel. File the issue at https://github.com/bytedance/Protenix/issues if so.

## Checkpoint / resume issues

### Symptom: `state_dict mismatch` when loading checkpoint

The `--model_name` you're using doesn't match the model that produced the checkpoint. Architectures differ between `protenix-v2`, `protenix_base_default_v1.0.0`, `protenix_mini_*`, etc. — they are not interchangeable.

If the mismatch is small (e.g. one module added), you can use `"load_strict": False` in the model config (see `configs/configs_model_type.py:185` for an example). This loads what it can and reinitializes the rest.

### Symptom: resume starts from step 0 instead of where you left off

The step counter is optionally restored. Check for a `--resume` flag or `--resume_from` in `runner/train.py`. Absent that, you just reduce `--max_steps` by however many you've already done.

### Symptom: EMA weights are much worse than non-EMA right after resume

You loaded the same file into both `--load_checkpoint_path` and `--load_ema_checkpoint_path`, but the checkpoint only contains non-EMA weights. Use the `ema.pt` file for the EMA slot.

## Data-loader issues

### Symptom: `FileNotFoundError` on first batch

Usually a version/dataset mismatch. See the mapping table in [`02-data-preparation.md`](./02-data-preparation.md): you need the `_0925` suffix for data version `2024.05.22` and `_v20260101` for data version `2026.01.01`.

### Symptom: training is IO-starved despite plenty of RAM

The preprocessed bioassembly files are compressed `.pkl.gz` — decompression is CPU-bound. If your CPU is the bottleneck:

- Pre-decompress to `.pkl` (costs disk but saves CPU): trivial script, but keep originals around.
- Increase `num_workers` to saturate CPUs.
- Move the dataset to local NVMe if it's on network storage.

### Symptom: empty batches, training skips forward steps

Your `pdb_list` filter excludes everything currently in the sampler's candidate set, or your `max_n_token` cap is too low for all entries.

Quick check:

```bash
wc -l my_subset.txt
# Intersect with actual training indices to see how many survive
zcat $PROTENIX_ROOT_DIR/indices/weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz \
  | cut -d, -f1 | sort -u > available.txt
comm -12 <(sort my_subset.txt) available.txt | wc -l
```

## Distributed training fails to start

### Symptom: `torchrun` hangs at init

- Missing `MASTER_ADDR` / `MASTER_PORT` env vars in multi-node.
- Firewall blocking the rendezvous port (default 29500).
- One rank failed silently before rendezvous; check `SLURM_JOB_ID` or `torchrun --rdzv_*` flags.

### Symptom: `NCCL timeout` mid-training

One rank is stuck, usually on a bad batch or an OOM that didn't crash cleanly. Check per-rank logs; restart from last checkpoint.

## Unexpected eval regressions

### Symptom: lDDT drops after switching to stage 2

Normal in the first 500 steps of stage 2 — the loss landscape changes with the new bond term. Recovers within 1–2k steps. If it's still below the stage-1 level after 3k steps of stage 2, something's wrong with `alpha_bond` or with checkpoint loading.

### Symptom: PoseBusters success rate crashes

If you enabled `alpha_bond=1.0` with a high LR, the model can overshoot and produce distorted backbones. Reduce LR or increase `warmup_steps`.

## When to open an issue vs. debug locally

Open an issue at https://github.com/bytedance/Protenix/issues when:

- You see NaN losses that reproduce across restarts with identical config.
- Fast kernels crash consistently with clean stack traces.
- Documented throughput numbers are off by >2×.

Debug locally first when:

- Your data doesn't match published data versions.
- You're using custom PDB entries or custom subsets.
- You're using a kernel combo that isn't in `inference_demo.sh`'s tested set.
