# Production service deployment guide

**Status:** Research / planning document
**Date:** 2026-04-28
**Goal:** Turn Protenix from a CLI-per-request tool into a multi-tenant prediction service for paying customers, "kept resident in GPU memory like an LLM," with low latency and high throughput.

---

## TL;DR

1. **Protenix is the right model to use** — it's an Apache-2.0 PyTorch reimplementation of AlphaFold 3 by ByteDance AI4Science. AF3's own weights are non-commercial; Protenix and Boltz-1/Boltz-2 (MIT) are the only commercially-clean AF3-class options as of April 2026.
2. **The "like an LLM" intuition is half right.** Keep weights resident in a long-lived process — yes. But most modern LLM-serving cleverness (continuous batching, PagedAttention, prefill/decode disaggregation) does **not** apply to AF3-class models because there's no autoregressive token loop.
3. **MSA search is 90%+ of end-to-end time** for AF3-class models. The single highest-ROI change is an MSA cache layer + GPU-MMseqs2, not anything inside the model itself.
4. **The repo has clean hooks** for wrapping `runner/inference.py::InferenceRunner` in a long-lived service. No model code needs to change.
5. **Realistic envelope per H100 with everything optimized:** 30–100 predictions/hour at full quality, 300–800/hour with Protenix-Mini. Multi-GPU scales near-linearly via data parallelism.

---

## 1. Is Protenix the same as AlphaFold 3?

Same architecture, different weights, very different license.

| Dimension | DeepMind AF3 | Protenix |
|---|---|---|
| Architecture | Pairformer + Diffusion | Same, re-implemented in PyTorch |
| Weights | Trained by DeepMind | Trained from scratch by ByteDance AI4Science |
| Code license | Open source | Apache-2.0 |
| **Weights license** | **Non-commercial only** ([prohibited use policy](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_PROHIBITED_USE_POLICY.md)) | **Apache-2.0 — commercial use OK** |
| Hosted endpoint | alphafoldserver.com (no commercial use) | protenix-server.com (ByteDance) |

### The AF3-clone landscape (April 2026)

| Model | Author | License | Notes |
|---|---|---|---|
| AlphaFold 3 | DeepMind | Code OS; **weights non-commercial** | Reference architecture |
| **Protenix-v1 / -v2** | ByteDance | **Apache-2.0** (code + weights) | Best on antibody-antigen and peptide-protein per FoldBench |
| **Boltz-1 / Boltz-2** | MIT | **MIT** | Boltz-2 adds binding-affinity head; ~20s/prediction on A100; >1000× faster than OpenFE |
| Chai-1 / Chai-2 | Chai Discovery | Non-commercial for self-hosting; commercial only via their API | Strong all-rounder |
| HelixFold3 / 3.2 | Baidu | CC BY-NC-SA (non-commercial) | RNA + covalent modification focus |

**Punchline for this project:** Protenix or Boltz-2. Avoid AF3, Chai-1, HelixFold3 weights if you're charging for predictions.

### Protenix variants in this repo (selected via `--model_name`)

| Variant | Params | MSA / Template | Notes |
|---|---|---|---|
| `protenix-v2` | 464M | ✅ / ✅ | Latest; strongest on antibody-antigen |
| `protenix_base_default_v1.0.0` | 368M | ✅ / ✅ | AF3-equivalent cutoff (2021-09-30) |
| `protenix_base_20250630_v1.0.0` | 368M | ✅ / ✅ | 2025-06-30 cutoff, practical use |
| `protenix_base_default_v0.5.0` | 368M | ✅ / ❌ | Legacy, no template |
| `protenix_mini_{esm,ism,default}_v0.5.0` | small | — | Lightweight; few-step diffusion; `_esm` / `_ism` skip MSA via PLM |
| `protenix_tiny_default_v0.5.0` | tiny | — | Ultra-light |
| `protenix_base_constraint_v0.5.0` | 368M | ✅ / — | Contact / pocket constraints |

The **Mini variants exist precisely for the latency-sensitive tier** — distilled, 1–2 step diffusion, ~85% FLOP reduction with ~1–5% accuracy hit per the Protenix-Mini paper ([arXiv 2507.11839](https://arxiv.org/abs/2507.11839)).

---

## 2. Why "serve it like an LLM" is half right

### The right half — yes, do this

- Long-lived process holds weights resident in GPU VRAM.
- Requests come in over HTTP / queue → forward pass → response.
- Never reload the checkpoint per request.

Cold load is 10–30 s; warm forward is seconds-to-minutes. The gap is two orders of magnitude. Getting to "weights resident, process alive" is by far the biggest single win, before any kernel-level optimization.

### The wrong half — most LLM-serving tricks don't apply

The modern LLM-serving stack (vLLM, TensorRT-LLM, SGLang, TGI) is built around two things AF3-class models don't have:

1. **Continuous / iteration-level batching.** vLLM's superpower is that LLMs decode token-by-token, so the scheduler can interleave many requests at every step. A structure prediction is one big forward pass — there's no per-token "step" to interleave at. (Diffusion sampling has 200 denoising steps, but they're a tiny fraction of total cost; not worth scheduling around.)
2. **PagedAttention / KV cache.** Same reason — there is no per-token KV cache to page in and out.

### What does carry over from LLM-land

- Long-lived weights-resident worker processes.
- **Dynamic batching** — Triton-style: collect a few requests up to `max_queue_delay`, run them together. Useful but capped because activation memory is huge; natural batch size for AF3 is 1–4.
- **Bucketing by sequence length** — much more important than for LLMs because triangle attention is O(N³). Padding a 200-residue protein up to a 4000-token max is an 8000× compute waste.
- **Mixed precision** — BF16 throughout, with FP32 pinned for numerically-touchy regions (Protenix already does this via `autocasting_disable_decorator`).
- **CUDA Graphs** — capture each fixed-shape forward once and replay. Combine with bucketing: one CUDA Graph per bucket.
- **`torch.compile(mode="reduce-overhead")`** — adds CUDA Graphs automatically. Expect 1.3–2× on the trunk.
- **Kernel fusion** and FlashAttention-style fused attention.
- **Multi-GPU data parallelism** (N replicas behind a queue) for throughput. **Tensor parallelism** only for very large complexes (>3000 tokens).
- **Snapshot / warmup on startup** so the first request isn't slow.
- **Autoscaling on queue depth** (KServe + KEDA, or Ray Serve).

### What the big labs / serving platforms do (2026)

- **Triton Inference Server** (NVIDIA) — framework-agnostic; best fit for Protenix because it natively handles "preprocess → big GPU forward → postprocess" pipelines via ensemble models.
- **Ray Serve** — Python-native, programmable, fractional GPUs. Good fit when each request orchestrates several components (MSA → embed → Pairformer → diffusion).
- **vLLM / SGLang / TensorRT-LLM** — LLM-only. Skip.
- **NVIDIA Dynamo** — new (GTC 2025) orchestration layer above the engines, datacenter-scale.
- **KServe v0.15** (June 2025) — Kubernetes-native CRD with KEDA-based LLM autoscaler, scale-to-zero, prefill/decode disaggregation. Use as orchestrator on top of Triton or Ray Serve.
- **Modal / Fireworks / Replicate / Baseten** — serverless GPU platforms competing on cold-start. Modal's GPU memory snapshots cut Parakeet ASR cold boot from ~20 s to ~2 s by checkpointing the full CUDA process state including VRAM.

---

## 3. What's actually slow in Protenix today

Three bottlenecks, in priority order.

### Bottleneck 1: MSA search (≈90% of end-to-end time)

OPIG measured AF3 spending **58 of every 60 minutes on CPU MSA**, with only 1–2 minutes on GPU ([blopig.com Sep 2025](https://www.blopig.com/blog/2025/09/accelerating-alphafold-3-for-high-throughput-structure-prediction/)). This isn't a "Protenix the model" problem — it's the pre-flight pipeline.

Speedup options:
- **AlphaFast** (Romero Lab, [bioRxiv 2026.02.17.706409](https://www.biorxiv.org/content/10.64898/2026.02.17.706409v1)) — replaces JackHMMER with GPU-MMseqs2 → **22.8× end-to-end on 1 GPU, 71.2× on 4 GPUs**, per-target time from ~15 minutes to **8 seconds**.
- **MMseqs2-GPU NIM** (NVIDIA, 2025) — 22× over CPU MMseqs2.
- **Per-sequence MSA cache** keyed on canonicalized sequence hash. Hit rate for repeat customers is near 100% (e.g., re-folding same protein with different ligands).
- **Skip MSA entirely** — Protenix `_esm` / `_ism` Mini variants substitute a protein language model for the alignment, eliminating the bottleneck.
- **Local ColabFold MMseqs2 server** — already running locally at `127.0.0.1:8080/api/`.

### Bottleneck 2: Pairformer triangle attention is O(N³)

Triangle attention attends over triplets of tokens. A 200-residue protein and a 2000-residue antibody-antigen complex differ by ~1000× in compute, not 10×.

- **NVIDIA cuEquivariance kernels** — reduce memory from O(N³) to O(N²) with up to **5× kernel speedup**. Already default in Protenix (`--triangle_attention cuequivariance --triangle_multiplicative cuequivariance`).
- **DeepSpeed `DS4Sci_EvoformerAttention`** — alternative backend, CUTLASS-based. **13× peak memory cut, 2–3× inference speedup** in OpenFold deployment. Wired into Protenix as `--triangle_attention deepspeed`.
- **MegaFold** ([arXiv 2506.20686](https://arxiv.org/abs/2506.20686)) — Triton-based memory-efficient EvoAttention + DeepFusion. 1.23× peak memory cut, 1.73× per-iteration training time, 1.35× longer sequences. Cross-platform (NVIDIA + AMD).
- **Triangle Multiplication is All You Need** ([arXiv 2510.18870](https://arxiv.org/html/2510.18870v1)) — drops triangle attention entirely; **4× faster long-sequence inference, 34% lower training cost**. Architectural change.

### Bottleneck 3: Diffusion sampling does N_step × N_sample forward passes

Defaults are 200 steps × 5 samples = **1000 diffusion forwards per seed**.

- **Protenix-Mini** ([arXiv 2507.11839](https://arxiv.org/abs/2507.11839)) — distilled to 1–2 ODE steps, Pairformer pruned 48→16 blocks, diffusion DiT 24→8 blocks. ~85% FLOP cut, 1–5% accuracy loss.
- **Protenix v0.7.0+** added `--enable_cache --enable_fusion` flags. The shared-variable cache reuses Pairformer trunk output across the N_sample diffusion samples (since they all condition on the same trunk). Free win — turn it on.
- **Reduce N_cycle / N_step** — `--use_default_params true` auto-fills recommended values; can be overridden lower for faster preview-tier predictions.

### Published end-to-end numbers (model warm, MSA cached)

- AF3 reference: 300-residue + ligand → **~30 s on A100**; 1500-residue antibody-antigen → **~90 s**.
- AF3 5120-token complex: **5+ minutes on 16 A100s** (DeepMind shards heavily for large inputs).
- Boltz-2 + cuEquivariance: **~20 s on one A100**.
- NVIDIA Boltz-2 NIM (TensorRT): **1.45–6.44× over OSS Boltz-2 on H100**.
- AlphaFast: **8 s per target on 4 GPUs** including MSA via GPU-MMseqs2; $0.035/prediction serverless cost.
- No published Protenix-specific server numbers; no Protenix NIM exists yet (open opportunity).

---

## 4. Code map: where to plug in

The code is clean enough that you do **not** need to rewrite anything to get a service running. The hooks are there.

### Entry points and one-time-vs-per-request work

| Component | File:line | Cost | When |
|---|---|---|---|
| `InferenceRunner.__init__` | `runner/inference.py:73` | Env init, model construction | **Once per process** |
| `load_checkpoint` | `runner/inference.py:144` | `torch.load` (~1–2 GB) | **Once per process** |
| `download_inference_cache` | `runner/inference.py:291` | CCD, RDKit mol cache, ESM weights | **Once per environment** (cached on disk after first run) |
| `infer_predict` | `runner/inference.py:418` | Main loop over JSONs | Replace with queue consumer for service |
| `runner.predict` (called from line 486) | `runner/inference.py` | Single batch forward | **Per request** |
| `update_inference_configs` | `runner/inference.py:385` | AMP toggle based on N_token to avoid OOM | Per request — keep this |
| `Protenix.forward` | `protenix/model/protenix.py:850` (calls `main_inference_loop` at line 904) | Pure neural-net forward | Per request, no changes needed |
| `sample_diffusion` | `protenix/model/generator.py:123` | 200×5 diffusion passes | Per request |

### MSA / template / RNA-MSA — **already separable from inference**

| Step | File | Where called |
|---|---|---|
| MSA search | `runner/msa_search.py:125` | `preprocess_input` in `batch_inference.py:113` |
| Template search | `runner/template_search.py` | `preprocess_input:125` |
| RNA MSA | `runner/rna_msa_search.py` | `preprocess_input:135` |

These are invoked from `protenix prep`, not from the GPU inference path. Good — the architecture already has a natural seam between "expensive CPU prep" and "GPU forward."

### Configuration — service-friendly

| File | Role |
|---|---|
| `configs/configs_base.py` | Base hyperparams |
| `configs/configs_inference.py` | Inference-only overrides |
| `configs/configs_model_type.py` | Per-model presets (loaded by name) |
| `protenix/config/config.py::parse_configs` | CLI flag → config-dict merger |

For a service: load config once at process start, reuse across requests. Compatible with current structure.

### Constraints today

- **Internal batch_size hardcoded to 1** in `protenix/data/inference/infer_dataloader.py:62`. The name `batch_inference.py` means "batch of input JSONs," not GPU batching. For a service this is fine — natural batch is 1–4 anyway because activations are huge; just keep replicas data-parallel.
- **No persistent server, no queue, no FastAPI/Triton wrapper.** You build that.
- **`protenix/web_service/` exists but is just utilities** (request parsing, viz). Not a server. Don't be misled by the directory name.
- **Kernel × model variant compatibility is not uniform.** `inference_demo.sh` shows tested combos; lock the kernel choice per model and validate up front.

---

## 5. Recommended optimization stack

In order of ROI. Stop when latency is acceptable.

| Step | Action | Expected impact |
|---|---|---|
| 1 | **Long-lived process** holding `InferenceRunner`. FastAPI + uvicorn + queue, or Ray Serve, or Triton with custom Python backend. | Eliminate per-request 10–30 s checkpoint load |
| 2 | **MSA cache layer.** Hash sequences (canonicalized) → store `.a3m` on local SSD or S3. Point at local MMseqs2 server. | **~90% of total latency savings for repeat queries** |
| 3 | **Pre-warm on startup**: synthetic forward at each token-bucket size (256/512/1024/2048/4096). Populates cuDNN autotuner and any compiled graphs. | First real request is fast |
| 4 | **Bucket inputs by token count.** Round each request up to next bucket and pad. | Caps padding waste at ~2× per bucket vs. unbounded |
| 5 | **`--enable_cache --enable_fusion`** (Protenix v0.7.0+). Shared trunk cache across diffusion samples. | Free; multiplies through all subsequent gains |
| 6 | **cuEquivariance triangle kernels** (already default). Validate per-model variant. | Already on |
| 7 | **`torch.compile(mode="reduce-overhead")` on trunk.** Adds CUDA Graphs. | 1.3–2× |
| 8 | **Tiered service**: Protenix-v1/v2 premium tier, Protenix-Mini fast/preview tier. | 5–10× cost gap on Mini |
| 9 | **Multi-tenant scaling**: data-parallel replicas (N H100s, KEDA-autoscaled on queue depth). | Linear throughput |
| 10 | **Very large complexes (>3000 tokens)**: tensor-parallel pair representation across 4–8 GPUs on one node with NVLink. | Mirrors DeepMind's 16-A100 setup |

### Things to defer or skip

- **Continuous batching, PagedAttention, prefill/decode disaggregation** — LLM-specific, don't apply to AF3 architecture.
- **INT4/INT8 weight-only quantization** — strong for memory-bound LLM decode, weaker for compute-bound triangle attention. Skip unless memory becomes the bottleneck.
- **TensorRT** — possible 1.5–6× win (per Boltz-2 NIM) but substantial engineering cost. Try `torch.compile` first.

---

## 6. Production architecture sketch

```
                 Customer HTTP/API
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  API gateway: auth, rate limits, billing,    │
   │              request validation              │
   └──────────────────────────────────────────────┘
                        │
                        ▼
       ┌────────────────────────────────┐
       │   Async job queue              │
       │   (Redis / SQS / RabbitMQ)     │
       │   Predictions are sec-to-min;  │
       │   do NOT make this synchronous │
       └────────────────────────────────┘
            │                       │
            ▼                       ▼
   ┌─────────────────┐    ┌─────────────────────────┐
   │ MSA worker pool │    │ GPU inference workers   │
   │ (CPU or         │    │ (H100, model resident,  │
   │  GPU-MMseqs2)   │    │  KEDA-autoscaled on     │
   │                 │    │  queue depth)           │
   └─────────────────┘    └─────────────────────────┘
            │                       │
            ▼                       ▼
   ┌──────────────────────────────────────────┐
   │  Shared MSA cache                        │
   │  (S3 or SSD, sequence-hash keyed)        │
   └──────────────────────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────┐
   │  Object storage for inputs/outputs (S3)  │
   └──────────────────────────────────────────┘
                        │
                        ▼
            Webhook / poll for results
```

This is essentially what Tamarind Bio and Neurosnap (the existing commercial AF3-clone hosting services) appear to run. **NVIDIA's Boltz-2 NIM is the closest published reference architecture** if you want a copyable blueprint — it ships TensorRT-optimized weights and reports 1.45–6.44× speedup over OSS Boltz-2 on H100.

---

## 7. Realistic envelope

After all the above, model warm and MSA cached, on a single H100:

| Workload | Full Protenix-v1/v2 | Protenix-Mini |
|---|---|---|
| 200-residue monomer + small ligand | ~15–30 s | ~3–6 s |
| 500-residue dimer + ligand | ~45–90 s | ~6–18 s |
| 1500-residue antibody-antigen | ~90–180 s | ~15–35 s |

Per-H100 throughput: roughly **30–100 predictions/hour** at full quality, **300–800/hour** in Mini mode. Multi-GPU scales near-linearly because requests are independent.

---

## 8. Existing production AF3-class services (competitive landscape)

| Service | Models | Commercial use? | Notes |
|---|---|---|---|
| DeepMind AlphaFold Server | AF3 only | ❌ Non-commercial only | 30 jobs/user/day; restricted ligand list |
| Tamarind Bio | 200+ models incl. AF2, Chai-1, Boltz, Protenix-style | ✅ Pharma SaaS | $13.6M Series A 2025; web UI + API |
| Neurosnap | AF2, Chai-1, Boltz-1, design tools | ✅ | Pay-as-you-go credits |
| Chai Discovery | Chai-1, Chai-2 (proprietary) | ✅ via API only | $130M Series B at $1.3B (Dec 2025) |
| NVIDIA BioNeMo / NIM | Boltz-2 (TensorRT), AF2 | ✅ DGX Cloud / AWS / Azure | **No Protenix NIM yet** |
| Protenix Web Server | All Protenix variants | ByteDance-hosted, free, T&Cs unclear | Reference deployment |
| Modal / Replicate / Baseten | DIY | ✅ your infra | Serverless GPU; cents per prediction at scale |

**Differentiation candidates:** Mini-tier pricing, antibody-antigen specialization (Protenix's strength), MSA caching across customer base (network effect), specialty workflows (e.g., docking integration).

---

## 9. Reality check / risks

- **MSA-search infrastructure is the actual hard part of this product.** Model serving is a solved problem. The MSA cache, database refresh schedule, GPU-MMseqs2 farm, cache hit rate optimization — that's where most engineering time will go.
- **Competitive market.** Tamarind, Neurosnap, Chai (commercial API), NVIDIA BioNeMo. "We host Protenix" alone is not a product.
- **Benchmark per use-case.** Protenix-v1/v2 leads on antibody-antigen and protein-peptide; AF3 still leads on some all-atom benchmarks; Boltz-2 leads on binding affinity. If you have a target customer segment, pick the model that wins on their workload.
- **Kernel × model variant compatibility is not uniform.** Test cuEquivariance + each Protenix variant before committing.

---

## 10. Open implementation tasks (when ready to build)

1. Wrap `InferenceRunner` in a long-lived FastAPI worker; verify weights stay resident, no per-request reload.
2. Build MSA cache: sequence-hash keyed, backed by local MMseqs2 server.
3. Implement bucketing layer (round token count to nearest of 256/512/1024/2048/4096; pad).
4. Pre-warm on startup at each bucket size.
5. Add job queue (Redis to start) with webhook-on-complete for async client UX.
6. Benchmark `torch.compile(mode="reduce-overhead")` on the trunk per bucket size.
7. Stand up Protenix-Mini as separate worker pool for the cheap/fast tier.
8. Multi-replica deployment with Ray Serve or KEDA-autoscaled K8s.
9. (Stretch) Investigate TensorRT export, mirroring NVIDIA's Boltz-2 NIM pattern.

---

## Sources

### Protenix
- [Protenix repo](https://github.com/bytedance/Protenix)
- [Protenix-v1 paper](https://www.biorxiv.org/content/10.64898/2026.02.05.703733v3.full)
- [Protenix-v2 paper](https://www.biorxiv.org/content/early/2026/04/11/2026.04.10.717613)
- [Protenix-Mini](https://arxiv.org/abs/2507.11839), [Protenix-Mini+](https://arxiv.org/abs/2510.12842)
- Local: `docs/kernels.md`, `docs/training_inference_instructions.md`, `inference_demo.sh`

### AlphaFold 3
- [AF3 repo](https://github.com/google-deepmind/alphafold3)
- [Weights prohibited use policy](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_PROHIBITED_USE_POLICY.md)
- [Performance docs](https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md)
- [AlphaFold Server](https://alphafoldserver.com/)

### Other AF3 reproductions
- [Boltz repo](https://github.com/jwohlwend/boltz), [Boltz-2 announcement](https://jclinic.mit.edu/boltz-2-towards-accurate-and-efficient-binding-affinity-prediction/)
- [Chai-1 repo](https://github.com/chaidiscovery/chai-lab), [Chai $130M Series B (TechCrunch)](https://techcrunch.com/2025/12/15/openai-backed-biotech-firm-chai-discovery-raises-130m-series-b-at-1-3b-valuation/)
- [HelixFold3 (PaddleHelix)](https://github.com/PaddlePaddle/PaddleHelix)

### Acceleration / systems papers
- [AlphaFast (GPU-MMseqs2 for AF3)](https://www.biorxiv.org/content/10.64898/2026.02.17.706409v1), [github](https://github.com/RomeroLab/alphafast)
- [MegaFold](https://arxiv.org/abs/2506.20686), [github](https://github.com/Supercomputing-System-AI-Lab/MegaFold)
- [FastFold](https://arxiv.org/abs/2203.00854), [ScaleFold](https://arxiv.org/abs/2404.11068)
- [Triangle Multiplication is All You Need](https://arxiv.org/html/2510.18870v1)
- [OPIG accelerating-AF3 blog](https://www.blopig.com/blog/2025/09/accelerating-alphafold-3-for-high-throughput-structure-prediction/)
- [DeepSpeed DS4Sci_EvoformerAttention](https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/)

### NVIDIA stack
- [cuEquivariance blog](https://developer.nvidia.com/blog/accelerated-molecular-modeling-with-nvidia-cuequivariance-and-nvidia-nim-microservices/)
- [Boltz-2 NIM overview](https://docs.nvidia.com/nim/bionemo/boltz2/latest/overview.html), [performance](https://docs.nvidia.com/nim/bionemo/boltz2/latest/performance.html)
- [MMseqs2-GPU NIM](https://developer.nvidia.com/blog/accelerated-sequence-alignment-for-protein-design-with-mmseqs2-and-nvidia-nim/)

### LLM serving (for technique reference)
- [Inside vLLM](https://www.aleksagordic.com/blog/vllm), [torch.compile + vLLM](https://blog.vllm.ai/2025/08/20/torch-compile.html)
- [Continuous batching from first principles (HF)](https://huggingface.co/blog/continuous_batching)
- [BucketServe (length-bucketed serving)](https://arxiv.org/html/2507.17120v1)
- [Triton dynamic batcher](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)
- [Modal GPU memory snapshots](https://modal.com/blog/gpu-mem-snapshots)
- [Fireworks: CUDA Graphs in Python](https://fireworks.ai/blog/speed-python-pick-two-how-cuda-graphs-enable-fast-python-code-for-deep-learning)
- [KServe v0.15](https://medium.com/@simardeep.oberoi/building-production-llm-infrastructure-with-kserve-v0-15-a5eecb2311bc)
- [NVIDIA Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)

### Hosted services / market
- [Tamarind Bio](https://www.tamarind.bio/), [Series A coverage](https://www.genengnews.com/topics/artificial-intelligence/tamarind-bio-secures-13-6m-series-a-to-make-ai-more-accessible-for-biology/)
- [Neurosnap](https://neurosnap.ai/)
- [Modal Boltz example](https://modal.com/docs/examples/boltz_predict)
- [ABCs of AF3, Boltz, Chai-1 (Boolean Biotech)](https://blog.booleanbiotech.com/alphafold3-boltz-chai1)
- [FoldBench (Nature Comms)](https://www.nature.com/articles/s41467-025-67127-3)
