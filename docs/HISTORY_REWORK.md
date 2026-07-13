# History Rework Map

The repository history was reorganized by semantic responsibility on
2026-07-13. The old long-lived branches and both pre-rework dirty source trees
remain reachable through annotated archive tags.

The rebuilt commits are a many-to-many semantic recomposition of the old
`packaging`, `zo-trainer-hf-integration`, and dirty-worktree changes. They are
not one-to-one cherry-picks, so there is intentionally no old-hash-to-new-hash
lookup table.

No archive tag or rebuilt branch has been pushed. They remain local references
until explicitly backed up.

## Preserved Main-Repository States

| Archive tag | Target | Tree | Original target date | Meaning |
|---|---|---|---|---|
| `archive/main-a9da4cb-before-rework` | `a9da4cbc7628150c10fe1eda5ed7929343536eb3` | `c00af2e4dab01b55ac12b9ad5331a48ca627a385` | `2026-05-27T20:57:29-05:00` | Public `main` before the rebuild |
| `archive/packaging-7db92c8` | `7db92c82a23b1c91966543aad19bb3f7e7f0a9e4` | `264e7fd0761b5468670ed867fcffecf14690b7be` | `2026-07-03T04:17:10-05:00` | Original packaging branch tip |
| `archive/hf-integration-509bc4b` | `509bc4b55f737b1f72603ba36596fd37eddd0bcc` | `2e1544ecebb7fdbbbd4ff3e9f699bcc31c2c9f81` | `2026-07-10T15:51:13-05:00` | Original HF-integration branch tip |
| `archive/main-dirty-before-rework-20260713` | `11541631f5f986fc1c46da15b745803df5152b2f` | `173ee13b5596683af5d717d9c3993e5e72dfa1a5` | `2026-07-13T03:08:38-05:00` | Synthetic snapshot of tracked and nonignored untracked pre-rework files |

The dirty snapshot has `509bc4b55f737b1f72603ba36596fd37eddd0bcc`
as its parent. It records vLLM at
`7c7a59dbf9fdb6afe860e3a49879b7a28db6e9f2` and LOZO at
`eaeb1c9ab7b302d37f1dc7c1c5d8ea7d8b7535c8`.

The original commit objects retain their hashes, author dates, committer dates,
messages, and trees. Only the newly composed commits have new hashes and dates.
The annotated archive tags themselves naturally have a new tag-creation date.

## Rebuilt Main-Repository Topology

The named branch roles are:

| Branch | Tip before this history-only document | Responsibility |
|---|---|---|
| `codex/rework/runtime-foundation` | `5c9ec25f4d5fd11ea142620c4688e45a70d6a545` | Packaged runtime, estimator, checkpoint, update, and phase foundations |
| `codex/rework/hf-native` | `c52a54abe85e4da233a45e62246c229beb3c1efb` | Native Hugging Face Trainer, preprocessing, checkpoint, and Phase 3/4 integration |
| `codex/rework/es-rollout` | `c0a7d0511ffeee7140e3f49bd3eb51a87c7c8141` | Generation-reward ES model boundary |
| `codex/rework/phase7-serving` | `1679554d7c078a5806beb127042c59b1ba74b61f` | Scheduled serving backend, HF thread bridge, QoS workflows, and benchmark |
| `codex/integration-rework` | `74bdb3f441f290a0e08fa4c538f6ddb156f917b7` | Integration content before this map was added |

The runtime-foundation sequence is:

```text
1eb817d Package reusable ZO-vLLM runtime
9ca6045 Unify ZO direction and estimator layers
4dbe43d Add task-neutral objective adapters
e0c44b5 Refactor ZO experiment infrastructure
c77011b Add resumable ZO checkpoint policies
6168050 Add worker-resident LoRA update banks
c429314 Migrate Phase 2 to shared runtime
a48851a Refocus Phase 3 on scaling benchmarks
6c07df6 Add Phase 4 convergence workflows
c7b36b3 Add OPT BOS and MeZO scope alignment
73828cb Add GPU runner smoke coverage
56e30b3 Make ZO estimates executor neutral
5c9ec25 Route ZO updates through optimizer state
```

The HF-native sequence is:

```text
795a1d7 Add the native Hugging Face ZO Trainer
f5d7168 Expose HF-native task preprocessing
e327651 Enforce strict ZO checkpoint artifacts
9b84c99 Migrate Phase 3 benchmarks to HF Trainer
e9bbaaf Migrate Phase 4 evaluation to HF Trainer
ae01b50 Support generic and effective HF outputs
c52a54a Add shared configuration regression tests
```

The topic branches add:

```text
c0a7d05 Add HF rollout reward models

61da7c2 Add scheduled serving engine backend
8f193f6 Run serving ZO through HF Trainer
51e4c81 Restore Phase 7 QoS workflows
1679554 Benchmark scheduled scoring overhead
```

Integration merge `d7ab2626adc73918e484919763775db3c1d58020`
joins rollout support. Merge
`1faafed4ba1ea351e7e1d4373e181b6d95633f98` joins Phase 7.

Before this history-only file was added, integration tree
`1f25e9f1ddca85d7890042b5a75990a900aa81af` and the archived dirty tree differed
only by blank-line cleanup in three package `__init__.py` files and trailing
whitespace cleanup in
`zo_vllm/experiment/scoring/generate_scorer.py`. A diff with whitespace and
blank lines ignored is empty.

## vLLM and LOZO Submodule Anchors

The vLLM runtime archive tags preserve both the old integration pointer and the
final rebuilt pointer:

| Archive tag | Target | Tree | Meaning |
|---|---|---|---|
| `archive/zo-runtime-61943f076` | `61943f076da0f8313ceb21a38da7130cda46e046` | `1b8e0c54936481dee605c30d04db885726c9d641` | Runtime referenced by the old HF-integration HEAD |
| `archive/zo-runtime-final-7c7a59dbf` | `7c7a59dbf9fdb6afe860e3a49879b7a28db6e9f2` | `7392ac9f199b9d091e90d4ff017c6158c7304e33` | Final runtime pinned by every rebuilt main branch |

The final runtime sequence is:

```text
61943f076 Optimize direct scoring outputs
1a51dd2fa Signal device tensor readiness
ebbc29cb3 Skip NLL during AGZO activation
7c7a59dbf Vectorize scheduled prompt NLL
```

`archive/lozo-eaeb1c9` preserves LOZO commit
`eaeb1c9ab7b302d37f1dc7c1c5d8ea7d8b7535c8` and tree
`d18101119a9ad1b0179ac9547ed79888190b09cb`.

## zo-post Rebuild

The sibling repository has the same two-level archive policy:

| Archive tag | Target | Tree | Meaning |
|---|---|---|---|
| `archive/zo-post-bd8902c` | `bd8902c5897ad647141343e9551a4eed3dc63afa` | `452eba372610a85314192d007c87a62621892bf7` | Original committed tip |
| `archive/zo-post-dirty-before-rework-20260713` | `443e0b7c8ce4422c91a659ecc94749d1eb3ee9f5` | `b4b89cec469cc3b5dc0f38aa14ca84b32fd331cc` | Synthetic pre-rework dirty snapshot |

Branch `codex/rework/zo-post-migration` contains:

```text
1735c3c Migrate Countdown ES to HF Trainer
d6413b8 Require typed artifacts in zo-post studies
1bfebc7 Normalize convergence monitor logs
d0ba1de Clean AGZO script lint
c86d931 Document migrated zo-post boundaries
```

Its final tree is exactly
`b4b89cec469cc3b5dc0f38aa14ca84b32fd331cc`, byte-for-byte identical to the
archived dirty tree.

## Recovery and Storage

Inspect an archived state without changing branches:

```bash
git show --stat archive/hf-integration-509bc4b
git show --stat archive/main-dirty-before-rework-20260713
```

Create an explicit recovery branch only when needed:

```bash
git switch -c recovery/pre-rework archive/main-dirty-before-rework-20260713
```

Archive tags do not duplicate a full repository. Git stores commits and trees
as small metadata objects and reuses identical blobs. New storage is consumed
only for genuinely new file contents, plus negligible commit/tree/tag metadata.
