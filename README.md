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

Neuron-level and months results are added when the runs finish.

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
