# Phase7 Punica Dual-LoRA Plan

Phase7 currently treats the base weight as read-only and routes ZO updates
through LoRA-style overlays. The next serving-training milestone needs one
Phase7 scoring request to apply two logical LoRA overlays:

```text
output = base(x) + shared_update(x) + perturbation_slot(x)
```

The top-level request should still carry a single LoRA id. If that id belongs to
the Phase7 perturbation range, the lower LoRA/Punica path should automatically
apply the shared update overlay in addition to the requested perturbation slot.

## Current Baseline

- Foreground serving requests use normal vLLM scheduling and priority `0`.
- Phase7 scoring requests use the same scheduler with priority `1000`.
- Phase7 serving does not use full direct-worker scoring RPCs in the hot path.
- Slot writes are allowed while foreground load is active, as long as a
  plus/minus pair is not overwritten before its scoring requests return.
- `SLOT_WRITE_STREAM=background_sync` is the conservative default slot copy path.
- The vLLM scheduler remains unmodified; Phase7 relies on normal priority
  scheduling plus idle-gap score admission.
- `VLLM_ZO_FORCE_EAGER_SCORING=1` remains a diagnostic safety setting for
  serving ZO compact-NLL batches until the dual-LoRA graph path is stable.

The current LoRA bank implementation can represent plus/minus as full bank
slots, but that duplicates the accumulated update into every probe slot. The
dual-LoRA Punica path is intended to remove that duplication.

## Why The Current Punica Path Is Not Enough

The current vLLM/Punica metadata model maps each token to one LoRA index:

```text
token_lora_mapping[token] = lora_index
```

`PunicaWrapperGPU.add_lora_linear(...)` then applies one LoRA pass:

```text
output += x @ A[lora_index] @ B[lora_index]
```

This means a single request cannot currently say "use perturbation slot K and
also use the shared update slot". Multiple logical ids can alias the same
physical tensor storage, but aliasing alone only solves storage ownership. It
does not change the forward semantics from one overlay to two overlays.

## Target Semantics

LoRA ids are divided into normal ids and Phase7 perturbation ids:

```text
normal LoRA id:
  output = base + LoRA[id]

Phase7 perturbation id:
  output = base + shared_update + perturbation[id]
```

The scheduler still sees a normal request-level LoRA id. The runner/Punica path
detects whether any active id belongs to the Phase7 perturbation range and builds
an additional shared-update mapping:

```text
perturb_mapping:
  Phase7 tokens -> perturbation_slot_index
  other tokens  -> -1

update_mapping:
  Phase7 tokens -> shared_update_slot_index
  other tokens  -> -1
```

The shared update must not apply to foreground tokens in the same GPU batch
unless that serving mode is explicitly requested later. The default Part2
semantics keep foreground no-training requests on the base model.

## First Implementation Shape

The first version should not modify Triton kernels. It should add a Phase7
dual-pass path around the existing Punica wrapper:

```text
normal batch:
  output = base(x)
  output += normal_lora_pass(x)

Phase7-active batch:
  output = base(x)
  output += perturbation_lora_pass(x)
  output += shared_update_lora_pass(x)
```

Required pieces:

- Add Phase7 perturbation-id range metadata to the worker-side LoRA runtime.
- Keep one physical shared-update slot per layer.
- Keep a perturbation slot ring for ready/submitted/done scoring requests.
- Add a second `LoRAKernelMeta` instance for shared-update tokens.
- Build `phase7_update_token_mapping` from the normal token mapping.
- Add a wrapper-level method such as `add_phase7_update_lora_linear(...)`.
- Keep normal foreground batches on the existing LoRA path.

This version trades one extra LoRA pass for much lower update storage
duplication and a simpler correctness story.

## CUDA Graph Strategy

Do not put per-token branching inside Triton kernels. The branch should happen at
the batch/wrapper level.

The practical first version should use two execution shapes:

```text
normal graph class:
  base + normal LoRA/no-LoRA

Phase7 graph class:
  base + perturbation LoRA + shared update LoRA
```

The Phase7 graph shape should be fixed once selected. Within a Phase7-active
batch, the update pass is always present. The mapping controls which tokens are
affected:

```text
phase7 token     -> shared_update_slot_index
non-phase7 token -> -1
```

This avoids changing Python forward structure inside the Phase7 path. It also
keeps foreground-only batches from paying the shared-update pass overhead.

Until the Phase7 graph is verified, Phase7-active compact-NLL batches may keep
the existing eager fallback while foreground-only batches continue to use the
server's normal compile/CUDA graph path.

## Fused Kernel Follow-Up

After the dual-pass wrapper path is numerically stable, a fused Punica path can
be considered:

```text
output += update_A/update_B contribution
output += perturb_A/perturb_B contribution
```

That would require extending metadata and Triton kernels to accept two mappings
or a pair of LoRA indices per token. This is a performance optimization, not the
first correctness milestone.

## Correctness Checks

- A Phase7 perturbation request equals `base + shared_update + perturbation`.
- A normal foreground request remains `base` or `base + normal_lora`.
- Mixed foreground plus Phase7 batches do not leak shared update into foreground
  tokens.
- Shared update physical tensors are not duplicated across perturbation slots.
- Perturbation slots are not overwritten while corresponding scoring requests
  are in flight.
- Base weight tensors remain unchanged before and after Phase7 scoring.
- Compact NLL matches the direct-worker numerical reference on an idle smoke
  path.

## Open Questions

- Exact representation of the Phase7 id range: environment config, engine
  config, or LoRA manager-owned metadata.
- Whether Phase7-active batches should initially force eager, use an independent
  graph bucket, or depend on the existing vLLM graph specialization.
- Whether `specialize_active_lora` should be disabled for Phase7-active batches
  to reduce graph churn.
- Whether shared update should ever apply to foreground requests after training
  has progressed. The current Part2 serving-training default says no.
