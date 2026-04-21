# Training Protenix-v2 From Scratch

This is Path A — the full reproduction of the `protenix-v2` (464M-parameter) weights that ByteDance did not publish. If you've read [`01-compute-budget.md`](./01-compute-budget.md) and still want to proceed, this page walks you through the four-stage recipe step by step.

## Prerequisites

- Data downloaded and validated per [`02-data-preparation.md`](./02-data-preparation.md). Use `--version 2026.01.01`.
- A multi-GPU machine or cluster. Minimum viable: 8× A100-80G. Reasonable: 32–64 GPUs. Ambitious: 128+.
- `PROTENIX_ROOT_DIR` and `PYTHONPATH` set.
- If you plan to use cuEquivariance kernels (recommended), `CUTLASS_PATH` set and `pip install cuequivariance-torch` completed. See `../kernels.md`.
- Weights & Biases account (optional but *strongly* recommended — you'll want the loss curves).

## What "v2" actually is

`protenix-v2` is defined in `configs/configs_model_type.py:52-88`. Relative to `protenix_base_default_v1.0.0`, it has:

| Field | v1 | v2 |
|---|---|---|
| `c_z` (pair repr dim) | 128 (default) | 256 |
| `diffusion_batch_size` | 48 | 64 |
| `template_embedder.hidden_scale_up` | false | true |
| `msa_module.hidden_scale_up` | false | true |
| `pairformer.hidden_scale_up` | false | true |
| `confidence_head.hidden_scale_up` | false | true |
| `N_cycle` (default) | 10 | 10 |

`hidden_scale_up` roughly doubles the hidden dimension in that module. The cumulative effect is 368M → 464M params. Nothing else is architecturally different — v2 is a scale-up, not a redesign. This means the training recipe is essentially the AF3 recipe with minor adjustments for the larger model.

## The four-stage recipe

AF3 trains in four stages that progressively increase crop size and shift loss weights. Summary from `../training_inference_instructions.md:241-255`:

| Hyperparameter | Stage 1 (Initial) | Stage 2 (FT-1) | Stage 3 (FT-2) | Stage 4 (FT-3) |
|---|---|---|---|---|
| `train_crop_size` | 384 | 640 | 768 | 768 |
| `diffusion_batch_size` | 48 | 32 | 32 | 32 |
| `loss.weight.alpha_pae` | 0 | 0 | 0 | 1.0 |
| `loss.weight.alpha_bond` | 0 | 1.0 | 1.0 | 0 |
| `loss.weight.smooth_lddt` | 1.0 | 0 | 0 | 0 |
| `loss.weight.alpha_confidence` | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| `loss.weight.alpha_diffusion` | 4.0 | 4.0 | 4.0 | 0 |
| `loss.weight.alpha_distogram` | 0.03 | 0.03 | 0.03 | 0 |
| `train_confidence_only` | False | False | False | True |

See [`05-loss-weights.md`](./05-loss-weights.md) for what each weight does and why it changes between stages.

## Stage 1 — Initial training

This is where the model learns the bulk of what it knows. Most of the compute budget lives here.

```bash
#!/bin/bash
# stage1-initial.sh
set -euo pipefail
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PROTENIX_ROOT_DIR=/path/to/protenix_data

torchrun --nproc_per_node=8 --nnodes=8 --rdzv_backend=c10d \
  runner/train.py \
  --run_name v2_stage1 \
  --model_name protenix-v2 \
  --seed 42 \
  --base_dir ./output/v2_stage1 \
  --dtype bf16 \
  --project protenix_v2 \
  --use_wandb true \
  --diffusion_batch_size 48 \
  --train_crop_size 384 \
  --max_steps 100000 \
  --warmup_steps 2000 \
  --lr 0.001 \
  --ema_decay 0.999 \
  --model.N_cycle 4 \
  --sample_diffusion.N_step 20 \
  --eval_interval 2000 \
  --log_interval 50 \
  --checkpoint_interval 2000 \
  --triangle_attention cuequivariance \
  --triangle_multiplicative cuequivariance \
  --loss.weight.alpha_pae 0 \
  --loss.weight.alpha_bond 0 \
  --loss.weight.smooth_lddt 1.0 \
  --loss.weight.alpha_confidence 1e-4 \
  --loss.weight.alpha_diffusion 4.0 \
  --loss.weight.alpha_distogram 0.03 \
  --data.train_sets weightedPDB_before250701_v20260101 \
  --data.test_sets recentPDB_1536_sample384_0925,posebusters_0925 \
  --data.posebusters_0925.base_info.max_n_token 768
```

Notes:

- **`--nproc_per_node=8 --nnodes=8`** → 64 GPUs. Adjust for your cluster. Also set `--rdzv_endpoint` and the `MASTER_ADDR`/`MASTER_PORT` env vars per your launcher.
- **`--model.N_cycle 4`** during training. Inference uses 10. This is a deliberate compute/quality trade from AF3: training with fewer cycles is cheaper, and the model generalizes to more cycles at inference.
- **`--sample_diffusion.N_step 20`** is the *mini-rollout* used during training eval. Inference uses 200. Again, cheaper during training.
- **`--eval_interval 2000`** gives you eval signal every ~6–8 hours at stage-1 speed. Resist dropping below 1000 — eval is expensive.
- **`--checkpoint_interval 2000`** so you can recover from crashes without losing more than a few hours.
- **`--data.posebusters_0925.base_info.max_n_token 768`** — caps PoseBusters complex size for eval memory safety.

Expect this to run ~10–14 days of wall-clock on 64 A100-80G GPUs. Monitor loss curves per [`06-monitoring-and-eval.md`](./06-monitoring-and-eval.md) and stop when eval lDDT on `recentPDB` plateaus.

### What to watch during stage 1

- **`loss/diffusion_total`** should drop steeply for the first ~10k steps, then slowly continue decreasing. If flat after 5k, your LR is too low or your data is broken.
- **`eval/recentPDB/lDDT`** is the north-star metric. Should climb to ~0.75–0.80 over the run.
- **NaN losses**: if they appear, see [`07-troubleshooting.md`](./07-troubleshooting.md). Usually caused by FP16/BF16 edge cases in attention.

At the end of stage 1 you should have:
- A `last.pt` checkpoint and an `ema.pt` checkpoint in `./output/v2_stage1/checkpoint/`.
- `eval/recentPDB/lDDT` plateau around 0.75.
- Confidence metrics will still be poor (that's what stage 4 fixes).

## Stage 2 — Fine-tune 1 (bond geometry)

Loads stage-1 checkpoint. Ups the crop to 640, introduces the bond-length loss, disables smooth lDDT.

```bash
#!/bin/bash
# stage2-ft1.sh
set -euo pipefail
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PROTENIX_ROOT_DIR=/path/to/protenix_data

ckpt=./output/v2_stage1/checkpoint/last.pt
ema=./output/v2_stage1/checkpoint/ema.pt

torchrun --nproc_per_node=8 --nnodes=8 runner/train.py \
  --run_name v2_stage2 \
  --model_name protenix-v2 \
  --seed 42 \
  --base_dir ./output/v2_stage2 \
  --dtype bf16 \
  --project protenix_v2 \
  --use_wandb true \
  --load_checkpoint_path "$ckpt" \
  --load_ema_checkpoint_path "$ema" \
  --diffusion_batch_size 32 \
  --train_crop_size 640 \
  --max_steps 20000 \
  --warmup_steps 500 \
  --lr 0.0005 \
  --ema_decay 0.999 \
  --model.N_cycle 4 \
  --sample_diffusion.N_step 20 \
  --eval_interval 1000 \
  --log_interval 50 \
  --checkpoint_interval 1000 \
  --triangle_attention cuequivariance \
  --triangle_multiplicative cuequivariance \
  --loss.weight.alpha_pae 0 \
  --loss.weight.alpha_bond 1.0 \
  --loss.weight.smooth_lddt 0 \
  --loss.weight.alpha_confidence 1e-4 \
  --loss.weight.alpha_diffusion 4.0 \
  --loss.weight.alpha_distogram 0.03 \
  --data.train_sets weightedPDB_before250701_v20260101 \
  --data.test_sets recentPDB_1536_sample384_0925,posebusters_0925 \
  --data.posebusters_0925.base_info.max_n_token 768
```

Key deltas vs. stage 1:
- `--load_checkpoint_path` / `--load_ema_checkpoint_path` — crucial, don't reinitialize.
- `--lr 0.0005` — halved. You're fine-tuning now.
- `--train_crop_size 640` — larger crops for better long-range interactions.
- `--diffusion_batch_size 32` — reduced to fit memory at the larger crop.
- `--loss.weight.alpha_bond 1.0` — turn on bond-length loss (PoseBusters-style validity).
- `--loss.weight.smooth_lddt 0` — turn off.

~3 days on 64× A100-80G.

## Stage 3 — Fine-tune 2 (larger crops, keep refining)

Same loss mix as stage 2, crop goes to 768.

```bash
#!/bin/bash
# stage3-ft2.sh
set -euo pipefail
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PROTENIX_ROOT_DIR=/path/to/protenix_data

ckpt=./output/v2_stage2/checkpoint/last.pt
ema=./output/v2_stage2/checkpoint/ema.pt

torchrun --nproc_per_node=8 --nnodes=8 runner/train.py \
  --run_name v2_stage3 \
  --model_name protenix-v2 \
  --seed 42 \
  --base_dir ./output/v2_stage3 \
  --dtype bf16 \
  --project protenix_v2 \
  --use_wandb true \
  --load_checkpoint_path "$ckpt" \
  --load_ema_checkpoint_path "$ema" \
  --diffusion_batch_size 32 \
  --train_crop_size 768 \
  --max_steps 10000 \
  --warmup_steps 500 \
  --lr 0.0003 \
  --ema_decay 0.999 \
  --model.N_cycle 4 \
  --sample_diffusion.N_step 20 \
  --eval_interval 1000 \
  --log_interval 50 \
  --checkpoint_interval 1000 \
  --triangle_attention cuequivariance \
  --triangle_multiplicative cuequivariance \
  --loss.weight.alpha_pae 0 \
  --loss.weight.alpha_bond 1.0 \
  --loss.weight.smooth_lddt 0 \
  --loss.weight.alpha_confidence 1e-4 \
  --loss.weight.alpha_diffusion 4.0 \
  --loss.weight.alpha_distogram 0.03 \
  --data.train_sets weightedPDB_before250701_v20260101 \
  --data.test_sets recentPDB_1536_sample384_0925,posebusters_0925 \
  --data.posebusters_0925.base_info.max_n_token 768
```

If you're memory-bound here (peak ~48 GB per GPU), reduce `--diffusion_batch_size` to 16 before reducing `--train_crop_size`.

~2 days on 64× A100-80G.

## Stage 4 — Fine-tune 3 (confidence head only)

This stage trains *only* the confidence head with frozen diffusion. Cheap (~13 s/step) and essential for well-calibrated PAE and pLDDT.

```bash
#!/bin/bash
# stage4-ft3.sh
set -euo pipefail
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PROTENIX_ROOT_DIR=/path/to/protenix_data

ckpt=./output/v2_stage3/checkpoint/last.pt
ema=./output/v2_stage3/checkpoint/ema.pt

torchrun --nproc_per_node=8 --nnodes=8 runner/train.py \
  --run_name v2_stage4 \
  --model_name protenix-v2 \
  --seed 42 \
  --base_dir ./output/v2_stage4 \
  --dtype bf16 \
  --project protenix_v2 \
  --use_wandb true \
  --load_checkpoint_path "$ckpt" \
  --load_ema_checkpoint_path "$ema" \
  --train_confidence_only true \
  --diffusion_batch_size 32 \
  --train_crop_size 768 \
  --max_steps 10000 \
  --warmup_steps 500 \
  --lr 0.0003 \
  --ema_decay 0.999 \
  --model.N_cycle 4 \
  --sample_diffusion.N_step 20 \
  --eval_interval 1000 \
  --log_interval 50 \
  --checkpoint_interval 1000 \
  --triangle_attention cuequivariance \
  --triangle_multiplicative cuequivariance \
  --loss.weight.alpha_pae 1.0 \
  --loss.weight.alpha_bond 0 \
  --loss.weight.smooth_lddt 0 \
  --loss.weight.alpha_confidence 1e-4 \
  --loss.weight.alpha_diffusion 0 \
  --loss.weight.alpha_distogram 0 \
  --data.train_sets weightedPDB_before250701_v20260101 \
  --data.test_sets recentPDB_1536_sample384_0925,posebusters_0925
```

Key deltas:
- `--train_confidence_only true` — freezes the diffusion and representation modules. Only confidence head and friends update.
- `--loss.weight.alpha_pae 1.0` — PAE is the whole point of this stage.
- `--loss.weight.alpha_diffusion 0` and `--loss.weight.alpha_distogram 0` — no structure-quality loss since you're not updating structure.

~1.5 days on 64× A100-80G.

## Multi-node launcher template

If your cluster uses Slurm:

```bash
#!/bin/bash
#SBATCH --job-name=protenix_v2_stage1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/v2_stage1_%j.out

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN

srun torchrun \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --nproc_per_node=8 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  runner/train.py [your args...]
```

## Resuming after a crash

If training dies mid-stage, the loop auto-checkpoints at `--checkpoint_interval` intervals. Resume by pointing a fresh invocation at the last saved checkpoint:

```bash
--load_checkpoint_path ./output/v2_stage1/checkpoint/last.pt \
--load_ema_checkpoint_path ./output/v2_stage1/checkpoint/ema.pt
```

The step counter restarts from zero by default — check `runner/train.py` for the `--resume` flag if you want the step index restored. Otherwise just subtract the consumed steps from `--max_steps` and relaunch.

## Verifying your final weights

After stage 4, run inference on the released evaluation sets and compare against `../model_1.0.0_benchmark.md`. You should see numbers at least as good as v1.0.0 (you trained with more params and a larger data cutoff), and likely slightly better on most metrics.

```bash
protenix pred \
  -i examples/examples_with_template/example_mgyp004658859411.json \
  -o ./output/v2_eval \
  -n protenix-v2 \
  --load_checkpoint_path ./output/v2_stage4/checkpoint/last.pt \
  --use_default_params true \
  --use_template true
```

If inference crashes with a state-dict mismatch, your `--model_name` and the checkpoint don't agree on architecture — check you're using `protenix-v2` and not `protenix_base_default_v1.0.0`.

## Publishing

Congratulations, you have your own v2 weights. A few housekeeping notes:

- The checkpoint is Apache-2.0-licensable since you trained it. Feel free to publish.
- Name it differently from `protenix-v2` to avoid confusion with ByteDance's eventual release. Suggest e.g. `protenix-v2-repro-<your-handle>-<date>`.
- Include a small training manifest: data version, step counts per stage, final eval metrics, checkpoint SHA.

Proceed to [`06-monitoring-and-eval.md`](./06-monitoring-and-eval.md) to learn what to watch during the run, or [`07-troubleshooting.md`](./07-troubleshooting.md) for debugging.
