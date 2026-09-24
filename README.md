# Matryoshka Attribution on Llama-3.1-8B, from first principles

A from-scratch walk through **Matryoshka Attribution (MAttr)**, run on **Llama-3.1-8B** split across **2 × Tesla T4** (Kaggle backend, driven from Colab).

* Paper: Arora, Acharya, Hu, Zhang, Goodman, Jurafsky, Potts. *Matryoshka attribution: Learning to attribute language model outputs to representations and weights.* arXiv:2609.25518 (2026).
* Official code: [aryamanarora/matryoshka-attribution](https://github.com/aryamanarora/matryoshka-attribution)
* This repo: one notebook, `matryoshka_attribution_llama8b.ipynb`. It rebuilds the method in about 300 lines (sigmoid top-k, the soft interchange hooks, the training loop, the evaluation metrics). It mirrors the official `sigmoid_topk.py`, `trainer.py` and `models/llama.py`, but it does not import them.

## The question

Llama-3.1-8B reads `57+38=` and says `95`. It has 1,024 attention heads, 32 MLP blocks and 458,752 MLP neurons. Which of them actually produced the `95`?

MAttr answers with one ranking of all components. It learns one score per component. At every training step it draws a random budget `k`, turns the scores into a soft mask that keeps exactly `k` components (sigmoid top-k), swaps every other component to its value on a different prompt (`12+71=`), and pushes the scores so the kept set still says `95`. Because `k` changes every step, one ranking has to work at every size. The top 5 sits inside the top 10, which sits inside the top 20, like nested Russian dolls.

## What the notebook shows

| Part | What | Result |
|---|---|---|
| 1 | Sigmoid top-k built by hand | Mask sums to `k` exactly, hand-derived gradient matches finite differences to 1e-10, masks are nested in `k` |
| 2 | 16-component toy with a brute-forced optimum | I×G misses two saturated backup components entirely. IG finds them but keeps the redundant one. MAttr learns that the backup is redundant and pushes it down. Total gap to the optimum: I×G 63.5, IG 4.85, MAttr 1.98 |
| 3 | Theorem 1 of the paper, checked numerically | The expected first MAttr step equals the centred, Beta(2,2)-weighted IG integral to 1.7e-14 |
| 4 | Llama-3.1-8B, 1,057 components (input + heads + MLPs), `a+b=` | Table below |
| 5 | 2,293,760 (layer, position, neuron) units | Compared with 28 layer-18 addition neurons that Feucht et al. (2026) found by hand. Three optimiser arms (Adam eps 1e-2, Adam eps 1e-8, SGD) |
| 6 | Months (`What month is three months after July?`) | Does MAttr trained only on months find the same neurons? |

### Node level (1,057 components, 399 held-out pairs)

| method | CPR | Compactness | cost on 2 × T4 |
|---|---|---|---|
| **MAttr** (500 steps) | **1.73** | 0.496 | 142 s |
| Activation patching (1,057 forward passes) | 1.20 | 0.495 | 375 s |
| IG (m = 10, mask path) | 1.18 | 0.497 | 45 s |
| I×G | 0.25 | 0.050 | 15 s |
| Random | 0.25 | 0.050 | 0 s |

The ordering matches the paper's MIB node-level average (MAttr 2.06, IG about 1.29, causal patching 1.14).
* MAttr ranks the two known arithmetic heads `a15.h13` and `a16.h21` (Nikankin et al. 2025) at #4 and #15. Activation patching and IG also put both in their top 5, so this confirms the known heads rather than being unique to MAttr.
* I×G falls to random because it ranks the input embedding 1,049th of 1,057. Every kept head recomputes from its inputs, so no circuit works without the input.
* Much of MAttr's CPR lead comes from overshoot. At `k = 528` its circuit has faithfulness 2.31. That means it moves the logit difference 2.31 times as far from the fully swapped run as the full model does. The paper itself notes that CPR rewards this, which is why it adds Compactness, where the top three methods tie.

**One knob at a time (node level).** Three seeds give CPR 1.73, 1.74, 1.73, so the curve is stable, although individual ranks wobble by a few places. Switching to a log-uniform `k` schedule costs a little CPR (1.68) and moves the two known arithmetic heads to #1 and #2, as the paper's §4 Result 2 predicts.

### Neuron level (2,293,760 units, 5,000 steps each)

Where do the 28 layer-18 addition neurons found by hand by Feucht et al. (2026) land, among all 458,752 neurons of the model?

| method | top 10 | top 100 | top 1,000 | median rank |
|---|---|---|---|---|
| IG (m = 10) | 3 | 15 | 28 / 28 | 89 |
| MAttr SGD (log `k`) | 1 | 6 | 27 | 186 |
| I×G | 0 | 8 | 25 | 193 |
| MAttr Adam eps 1e-2 | 0 | 3 | 22 | 276 |
| MAttr Adam eps 1e-8 | 0 | 1 | 11 | 1,538 |
| Random | 0 | 0 | 1 | 235,923 |

Faithfulness of the neuron circuits on held-out pairs:

| method | CPR | Compactness | faithfulness with 0.1% of units |
|---|---|---|---|
| MAttr Adam eps 1e-2 | 5.20 | 0.968 | 0.60 |
| MAttr Adam eps 1e-8 | 6.29 | 0.904 | 0.22 |
| MAttr SGD | 1.12 | 0.948 | 0.82 |
| IG (m = 10) | 1.06 | 0.978 | 0.87 |
| I×G | 1.03 | 0.587 | 0.10 |

* Every method is hundreds of times better than chance at finding the hand-found neurons, from a search over the whole model with no hint about layer 18.
* The Adam eps effect reproduces. Changing only eps from 1e-8 to 1e-2 improves the median rank of the known neurons 5.6 times and raises Compactness. The eps 1e-8 run has the higher CPR, which matches the paper's Fig. 8.
* The official code's claim that MAttr with SGD beats IG at recovering these neurons does not reproduce in our single run. IG ranks them best. The neurons were selected by overlap with a linear DAS subspace, which may favour gradient methods, and MAttr is trained to keep the answer, not to find them.
* MAttr's CPR lead at neuron level comes from overshoot. With 5% of units kept, its circuit pushes the logit difference up to 7 times past the full model. On Compactness and at the smallest circuits, IG is as good or better.

### Months (5,963,776 units, trained on months prompts only)

`Q: What month is three months after July?\nA:` (13 tokens, 2,724 training pairs, 300 test pairs).

| method | top 10 | top 100 | top 1,000 | median rank of Feucht's 16 months neurons |
|---|---|---|---|---|
| MAttr Adam eps 1e-2 | 2 | 8 | 15 / 16 | 140 |
| IG (m = 10) | 3 | 10 | 16 / 16 | 44 |
| Random | 0 | 0 | 0 | 217,117 |

Overlap between the top-500 neurons found on addition and on months, two independent runs on two datasets: MAttr 139, IG 184, chance 0.55. MAttr trained only on months puts `L18/N1712` and `L18/N10099`, which are in both of Feucht's sets, at #5 and #6. This is Feucht et al.'s central claim, that Llama reuses its addition neurons for calendar arithmetic, recovered without supervision. Again, IG does it slightly better.

## Scorecard

| paper claim | our result | verdict |
|---|---|---|
| Sigmoid top-k keeps exactly `k` and is nested in `k` | exact to 1e-6, gradient matches finite differences to 1e-10 | reproduced |
| Theorem 1: first MAttr step is centred, Beta(2,2)-weighted IG | matches the exact integral to 1.7e-14 | reproduced |
| MAttr beats causal patching, IG and I×G on node-level CPR | 1.73 vs 1.20 / 1.18 / 0.25 | reproduced (ordering) |
| Known arithmetic heads rank high | #4 and #15, or #1 and #2 with log `k`. Patching and IG also find them | reproduced, not unique to MAttr |
| Log-uniform `k` sharpens the small-circuit end | known heads to #1 and #2, input node #21 to #8, CPR 1.73 to 1.68 | reproduced |
| Adam eps matters at neuron level | median rank 276 vs 1,538, Compactness 0.968 vs 0.904 | reproduced |
| MAttr matches or beats IG at recovering published neurons | IG best in our run, on both addition and months | not reproduced (single run) |
| Addition and months share neurons | 139 of top-500 shared, about 250 times chance | reproduced |
| Parameter attribution with RL (refusal) | not run | not tested |

Scope: one task family, one seed per neuron-level arm, batch 8, fp16 on T4, our own implementation of the metrics.

## Setup notes

* Model: `unsloth/Meta-Llama-3.1-8B`, an ungated copy of `meta-llama/Llama-3.1-8B` (same weights). Downloads in about 2 minutes on Kaggle.
* fp16, not bf16. The T4 has no bf16 tensor cores.
* Layers 0 to 15 on GPU 0, layers 16 to 31 and the output head on GPU 1 (7.5 GB each).
* Weights stay frozen. Only the per-component scores are trained. The backward pass goes through the frozen model to reach them.
* Loss scaling for the fp16 backward. Started at 1024. Lowered to 64 partway through the first neuron run, when overshooting circuits (logit difference around 90) began to overflow fp16 on about 6% of steps. The scale is divided out before the optimiser step, so this changes only the numerical range, not the maths.

## Differences from the paper

* Batch 8 instead of batch 1 (same wall-clock cost on a T4, less noise).
* Our own evaluation code (same definitions of faithfulness, CPR, IIA and Compactness as paper eqs. 4, 5, 8, 9), not MIB's harness. Our addition task is not a MIB task, so numbers are comparable in ordering, not in absolute value.
* IG is integrated along the mask path `alpha = t * 1` (the IG of the paper's Theorem 1), not along the input-embedding path used by MIB's IG baseline.
* Sigmoid top-k bisection bracket widened from `±10T` (official code) to `±T(log N + 10)`. With 2.3 million units and all scores at zero, the official bracket cannot reach small `k`. At init the mask for `k = 1` sums to about 104 instead of 1.
* Parameter attribution with RL (paper Section 5, removing refusal from Llama-3.1-8B-Instruct) is not reproduced here.
