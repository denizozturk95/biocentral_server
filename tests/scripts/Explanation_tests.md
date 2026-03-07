## Experiments

| # | Script | Invariant / MR | What it tests |
|---|--------|---------------|---------------|
| 1 | `test_idempotency.py` | **Idempotency** | Same sequence → same embedding (repeated calls) |
| 2 | `test_batch_invariance.py` | **Batch invariance** | Embedding A+B together == embedding A, B separately (pooled + per-residue) |
| 3 | `test_projection_determinism.py` | **Projection determinism** | PCA/UMAP/t-SNE on same data → same (or similar) coordinates |
| 4 | `test_training_determinism.py` | **Training determinism** | Re-embedding with same model + seed → identical embeddings |
| 5 | `test_progressive_x_masking.py` | **MR: X-masking** | Progressively replace residues with 'X'; measure divergence |
| 6 | `test_reversed_sequence.py` | **MR: Reversed sequence** | Reverse order; measure sensitivity to position (ESM2 + one-hot baseline) |
| 7 | `test_random_baseline.py` | **Baseline: Random seqs** | X-masking on randomly generated (non-biological) sequences |
| 8 | `test_aa_mutation_sensitivity.py` | **MR: AA mutation** | Replace residues with each of the 20 standard amino acids |

### Baselines & comparisons

| Script | Purpose |
|--------|---------|
| `fetch_uniref50_sequences.py` | Download & cache 250 seqs/bin from UniRef50 (7 length bins) |
| `plot_x_masking.py` | Cosine similarity + L2 distance plots, baseline comparison |
| `plot_aa_mutation.py` | Heatmap, line overlay, bar ranking for mutation sensitivity |

All experiments use ESM2-T6-8M for meaningful results. CSV reports are written to `tests/reports/`.

## Experiment Details

### 1. Idempotency

Calling `embed(seq)` multiple times must return identical results (for a deterministic embedder) or results within a very tight tolerance (for GPU models with non-deterministic kernels).

- **ESM2**: tolerates cosine distance ≤ 1e-5
- Tested on first 10 extended sequences × 4 repeats

### 2. Batch Invariance

Embedding sequences A and B *together* in one batch must yield the same per-sequence embedding as embedding them *individually*. Transformer attention operates over the single sequence, not across batch members, so batch composition must be invisible to the output.

- Tests batch sizes: 2, 5, 10, 20
- Target sequence placed at random positions within the batch
- **Pooled variant**: compares pooled (mean-reduced) vectors via cosine distance
- **Per-residue variant**: compares full (L, D) embedding matrices via max absolute difference

### 3. Projection Determinism

- **PCA**: deterministic → expects bit-identical coordinates across runs
- **UMAP / t-SNE**: stochastic → checks structural similarity via Procrustes distance

### 4. Training Determinism

Verifies that embedding computation during training preparation is deterministic: same sequences, same model, same seed → identical embeddings across repeated runs.

- Uses 8 short sequences with binary labels
- 3 repeated runs, compared against the first
- Tolerance: cosine distance ≤ 1e-5

### 5. Progressive X-Masking (Metamorphic Relation)

Replace progressively more amino acids with `X` (the standard placeholder for "not resolved / unknown") and measure how the embedding diverges from the original.

**Questions answered:**
- Is divergence monotonically increasing with masking ratio?
- At what masking ratio does the embedding differ *significantly*?
- Critical ratio `r*` where cosine distance > 0.1

Masking ratios tested: 0%, 5%, 10%, 15%, 20%, 25%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%

**Variants:**
- `TestProgressiveXMaskingESM2` — 4 hardcoded sequences (15, 76, 400, 300 aa), 5 runs each
- `TestXMaskingUniRef50` — 250 real sequences per length bin (7 bins: 50, 100, 200, 400, 600, 800, 1000), 2 runs each
- `TestXMaskingOneHot` — one-hot encoding baseline on 4 basic sequences
- `TestXMaskingOneHotUniRef50` — one-hot encoding baseline at UniRef50 scale

### 6. Reversed Sequence (Metamorphic Relation)

Reverse the amino acid order and compare embeddings.

- **ESM2**: expects significant cosine distance (model is order-sensitive via positional encoding)
- **One-hot baseline**: expects zero cosine distance (bag-of-residues, order-insensitive)
- Double-reversal identity check: reverse(reverse(seq)) == seq → same embedding

### 7. Random Sequence Baseline

Generate random amino acid sequences (uniform sampling from 20 standard AAs) and run the same X-masking experiments. This baseline shows whether the masking divergence profile differs for "meaningless" sequences versus real proteins.

- Same 7 length bins as UniRef50: 50, 100, 200, 400, 600, 800, 1000 aa
- 50 random sequences per bin, 2 runs each

### 8. Amino Acid Mutation Sensitivity

Instead of replacing residues with `X`, replace with each of the 19 *other* standard amino acids. This measures how sensitive the embedding model is to specific substitutions.

**Questions answered:**
- Which amino acid substitutions cause the largest embedding change?
- Is the model more sensitive to substitutions involving charged/hydrophobic AAs?
- How does AA-replacement compare to X-masking (structured vs unknown token)?

**Variants:**
- `TestAAMutationSensitivity` — 4 basic sequences, all 20 AAs, 3 runs
- `TestAAMutationUniRef50` — 250 seqs/bin from UniRef50, 6 representative AAs (A, D, G, K, L, W), 1 run

**Visualisations** (`plot_aa_mutation.py`):
- Heatmap: replacement AA × masking ratio → mean distance
- Line overlay: 20 AA curves + X-masking reference on same axes
- Bar chart: AAs ranked by embedding change at 10% mutation rate, color-coded by physicochemical group

### Plotting

- `plot_x_masking.py` produces:
  - **1×2 cosine similarity** plot → `x_masking_similarity_plot`
  - **2×2 cosine + L2** plot (top: cosine similarity, bottom: L2 distance) → `x_masking_cosine_and_l2_plot`
  - **Baseline comparison** plot (ESM2 vs one-hot vs random sequences, cosine + L2) → `x_masking_baseline_comparison`
  - **UniRef50 per-bin** plot (mean±std per length bin, cosine + L2) → `x_masking_uniref50`
  - **UniRef50 vs baselines** plot (aggregated comparison, cosine + L2) → `x_masking_uniref50_vs_baselines`
- `plot_aa_mutation.py` produces:
  - **Heatmap** (AA × ratio → distance) → `aa_mutation_heatmap`
  - **Line overlay** (all 20 AAs + X-masking reference) → `aa_mutation_line_overlay`
  - **Bar ranking** (AAs at 10% mutation rate) → `aa_mutation_bar_ranking`
  - **UniRef50 heatmap** (aggregated across all bins) → `aa_mutation_heatmap_uniref50`
  - **UniRef50 per-bin** (per-length AA sensitivity at ~10%) → `aa_mutation_per_bin_uniref50`
  - **Mutation vs X-masking** (mean AA distance + range vs X-masking reference) → `aa_mutation_vs_xmasking`

### CSV Reports

All CSV reports are written to `tests/reports/`. Key files:

| File | Source Test | Description |
|------|-----------|-------------|
| `idempotency_esm2_pooled.csv` | `test_idempotency.py` | Repeat-call consistency |
| `batch_invariance_esm2.csv` | `test_batch_invariance.py` | Batch vs individual (pooled) |
| `batch_invariance_per_residue_esm2.csv` | `test_batch_invariance.py` | Batch vs individual (per-residue) |
| `training_determinism_esm2.csv` | `test_training_determinism.py` | Seed-controlled embedding reproducibility |
| `reversed_sequence_esm2.csv` | `test_reversed_sequence.py` | Order sensitivity (ESM2) |
| `reversed_sequence_one_hot.csv` | `test_reversed_sequence.py` | Order sensitivity (one-hot baseline) |
| `reversed_sequence_esm2_diverse.csv` | `test_reversed_sequence.py` | Reversal + double-reversal |
| `x_masking_progressive_esm2.csv` | `test_progressive_x_masking.py` | Basic 4-seq progressive masking |
| `x_masking_random_esm2.csv` | `test_progressive_x_masking.py` | Basic 4-seq random masking |
| `x_masking_progressive_one_hot.csv` | `test_progressive_x_masking.py` | One-hot baseline (basic) |
| `x_masking_random_one_hot.csv` | `test_progressive_x_masking.py` | One-hot baseline (random, basic) |
| `x_masking_progressive_one_hot_uniref50.csv` | `test_progressive_x_masking.py` | One-hot baseline (UniRef50 scale) |
| `x_masking_progressive_uniref50.csv` | `test_progressive_x_masking.py` | UniRef50 large-scale progressive |
| `x_masking_random_uniref50.csv` | `test_progressive_x_masking.py` | UniRef50 large-scale random |
| `x_masking_progressive_random_seqs.csv` | `test_random_baseline.py` | Random sequences baseline |
| `x_masking_random_random_seqs.csv` | `test_random_baseline.py` | Random sequences baseline (random masking) |
| `aa_mutation_sensitivity_esm2.csv` | `test_aa_mutation_sensitivity.py` | Basic 4-seq AA mutation |
| `aa_mutation_sensitivity_uniref50.csv` | `test_aa_mutation_sensitivity.py` | UniRef50 AA mutation |
