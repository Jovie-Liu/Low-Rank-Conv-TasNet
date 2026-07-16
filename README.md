# Artificial-Neuroscience-Metrology-and-Engineering-for-Deep-Learning-using-Linear-Algebra

- The project titled *[Artificial Neuroscience: Metrology and Engineering for Deep Learning Using Linear Algebra](https://gtr.ukri.org/projects?ref=EP%2FZ535448%2F1)* is a UKRI (UK Research and Innovation) EPSRC (Engineering and Physical Sciences Research Council) Discipline Hopping in ICT Grant featuring collaborations between the School of Electronic Engineering and Computer Science and the School of Mathematical Sciences at Queen Mary University of London, United Kingdom. Associated with the Centre for Digital Music ([C4DM](https://www.c4dm.eecs.qmul.ac.uk/people/)) and the [Centre for Fundamentals of AI and Computational Theory](https://www.seresearch.qmul.ac.uk/cfcs/people/jiliu/), I work as a Postdoctoral Research Associate co-supervised by [Prof. Mark Sandler](https://www.seresearch.qmul.ac.uk/cmai/people/msandler/#grants) and [Prof. Boris Khoruzhenko](https://www.seresearch.qmul.ac.uk/cpsd/people/bkhoruzhenko/#grants).

- To gain a deeper understanding of the **correlation structure in audio processing neural networks**, we apply Empirical Spectral Density (ESD) techniques to weight matrices to investigate the dynamics of training.

- To build more **efficient, compact, and less energy-consuming neural networks**, we employ tensor decomposition on the weight matrices and examine the effects of low-rank approximation on the neural network behaviors.

---

## Research Phase I: Low-Rank Structure in Linear Weights

In the first phase of this project, we investigate the low-rank structure of the linear weights in [**Conv-TasNet**](https://arxiv.org/pdf/1809.07454), focusing on the pointwise convolution layers ([Video Demo](https://www.youtube.com/watch?v=fL-FDF-Iojk&list=PLWSd-mlbNCAWjovFmisi1asUd0StPzdPc&index=2)). Specifically, we study whether singular value decomposition (SVD) can reveal redundant directions in these learned weights, allowing us to compress the model while preserving its source separation performance. We compress **all 74 pointwise $1\times1$ convolution layers** in Conv-TasNet using **layer-wise truncated SVD**, where each layer’s rank is chosen relative to its own **effective rank**. All results below use **Libri2Mix (8 kHz, min mode)** and report **validation loss = negative SI-SNR** (**lower is better**).

### Quick takeaways

- **Effective rank is the sweet spot.**  
  At **$1.0\,k_{\mathrm{eff}}$**, the model is still **1.10× smaller** and slightly **better** than the baseline after fine-tuning.
- **Conv-TasNet is surprisingly compressible.**  
  Even at **2.64× compression**, the model recovers most of its performance after fine-tuning.
- **Factorization is not only post-hoc compression.**  
  If introduced early in training, the model can still learn effectively.

---

### Experiment 1 — Post-hoc factorization + fine-tuning

Start from the **best vanilla Conv-TasNet checkpoint**, replace all pointwise linear layers with factorized versions, then fine-tune.

**Baseline validation loss:** `-14.2096`

| Effective Rank Ratio | Model Compression Ratio | Initial Factored Loss | Best Loss after Fine-tuning |
|:---:|:---:|:---:|:---:|
| Baseline | 1.00× | -- | -14.2096 |
| 0.1 | 8.71× | 0.7177 | -9.8351 (ep +56) |
| 0.2 | 4.93× | 1.3163 | -12.2977 (ep +51) |
| 0.3 | 3.44× | -2.1353 | -13.3362 (ep +52) |
| 0.4 | 2.64× | -8.2283 | -13.7470 (ep +38) |
| 0.5 | 2.14× | -10.7592 | -13.9976 (ep +38) |
| 0.6 | 1.80× | -12.1045 | -14.0856 (ep +15) |
| 0.7 | 1.56× | -12.9411 | -14.1430 (ep +25) |
| 0.8 | 1.37× | -13.4710 | -14.2181 (ep +19) |
| 0.9 | 1.22× | -13.8072 | -14.2162 (ep +11) |
| 1.0 ($k_{\mathrm{eff}}$) | 1.10× | -14.0033 | **-14.2339** (ep +7) |
| 1.1 | 1.01× | -14.1314 | -14.2247 (ep +13) |
| 1.2 | 0.94× | -14.1845 | -14.2241 (ep +9) |
| 1.3 | 0.90× | -14.2017 | -14.2350 (ep +19) |
| 1.4 | 0.88× | -14.2096 | -14.2390 (ep +15) |
| 1.5 | 0.86× | -14.2101 | **-14.2742** (ep +13) |
| full | 0.81× | -14.2097 | -14.2507 (ep +16) |

**Summary:**  
- Below **$k_{\mathrm{eff}}$**: more compression, but steadily worse final performance.  
- Around **$k_{\mathrm{eff}}$**: best **compression/performance trade-off**.  
- Above **$k_{\mathrm{eff}}$**: only tiny gains, but compression quickly disappears.

<img src="experiment1_plot.jpg" style="width:700px">

**This plot is a visualization of the results in the above table, and it shows:**
- Aggressive truncation hurts immediately, but fine-tuning recovers much of the loss.  
- The most interesting region is around **$k_{\mathrm{eff}}$**, where the model stays smaller **and** performs as well as or slightly better than the baseline.

---

### Experiment 2 — Early-epoch factorization + continued training

Instead of waiting for full convergence, we factorize the model at **epoch 1, 5, or 10**, then continue training. Here $k_{\mathrm{eff}}$ refers to the effective rank at each layer; the range for ascending and descending ranks is 20% - 100% of the effective rank.

| Setting | Compression Ratio | Best Loss after Training | Checkpoint / Initial Factored Loss |
|---|---:|---:|---:|
| Epoch 10 ($k_{\mathrm{eff}}$) | 1.24× | **-13.9449** (ep +81) | -10.8078 / -10.7734 |
| Epoch 10 ($0.5 \ k_{\mathrm{eff}}$) | 2.39× | -13.6216 (ep +83) | -10.8078 / -10.1671 |
| Epoch 10 ($0.25\ k_{\mathrm{eff}}$) | 4.49× | -12.9179 (ep +69) | -10.8078 / -7.7510 |
| Epoch 10 ($k_{\mathrm{asc}}$) | 2.06x | -13.4640 (ep +93) | -10.8078 / -9.2964 |
| Epoch 10 ($k_{\mathrm{desc}}$) | 1.98x | -13.6017 (ep +73) | -10.8078 / -9.7609 |
| Epoch 5 ($k_{\mathrm{eff}}$) | 1.34× | -13.7195 (ep +88) | -9.6479 / -9.6290 |
| Epoch 5 ($0.5\ k_{\mathrm{eff}}$) | 2.59× | -13.3547 (ep +81) | -9.6479 / -9.3568 |
| Epoch 5 ($0.25\ k_{\mathrm{eff}}$) | 4.84× | -12.2960 (ep +85) | -9.6479 / -7.8804 |
| Epoch 5 ($k_{\mathrm{asc}}$) | 2.24x | -13.3755 (ep +95) | -9.6479 / -9.2429 | 
| Epoch 5 ($k_{\mathrm{desc}}$) | 2.12x | -13.4953 (ep +76) | -9.6479 / -9.0872 | 
| Epoch 1 ($k_{\mathrm{eff}}$) | 1.39× | -13.5912 (ep +95) | -5.8962 / -5.8926 |
| Epoch 1 ($0.5\ k_{\mathrm{eff}}$) | 2.68× | -13.3419 (ep +89) | -5.8962 / -5.8578 |
| Epoch 1 ($0.25\ k_{\mathrm{eff}}$) | 4.99× | -12.5440 (ep +85) | -5.8962 / -5.7438 |
| Epoch 1 ($k_{\mathrm{asc}}$) | 2.37x | -13.3551 (ep +96) | -5.8962 / -5.8610 |
| Epoch 1 ($k_{\mathrm{desc}}$) | 2.17x | -13.4938 (ep +76) | -5.8962 / -5.8267 |

**Summary:**  
- **$k_{\mathrm{eff}}$ remains the best checkpoint-based rank rule**, giving the strongest final performance at epochs 10, 5, and 1.  
- **Later factorization is consistently better**: epoch 10 outperforms epoch 5 and epoch 1 across all rank schedules.  
- **Descending rank schedules are consistently slightly stronger and converge faster than ascending ones**, showing that layer-wise rank allocation matters, not just overall compression.
- The overall pattern suggests that **$k_{\mathrm{eff}}$ is best for preserving learned structure**, while **more structured or more constrained rank schedules may be better for training factorized models from scratch**.

---

### Experiment 3 — Random Initialization

We train the factorized model from scratch with **random initialization**, without warm-up pretraining. Here $k_{\mathrm{eff}}$ is the effective rank distribution of the best vanilla Conv-TasNet checkpoint. We experiment with various rank structures, including **uniform rank and ascending/descending ranks with different shapes and rates (linear, convex, concave)**.

| Rank | Compression Ratio | Best Loss after Training |
|---|---:|---:|
| $k_{\mathrm{eff}}$ | 1.10× | -8.4764 (ep 100) |
| Uniform $80$ | 1.27x | -9.5967 (ep 95) |
| Uniform $40$ | 2.46x | -13.1355 (ep 97) |
| Uniform $20$ | 4.62x | -12.7622 (ep 96) |
| Uniform $10$ | 8.22x | -11.1319 (ep 81) |
| Uniform $5$ | 13.48x | -8.9363 (ep 94) |
| Linear $20 - 100$ | 1.67x | -9.5110 (ep 100) |
| Linear $100 - 20$ | 1.69x | -13.2841 (ep 97) |
| Linear $10 - 60$ | 2.77x | -12.3025 (ep 95) |
| Linear $60 - 10$ | 2.81x | -13.0842 (ep 85) |
| Linear $5 - 30$ | 5.16x | -12.2210 (ep 99) |
| Linear $30 - 5$ | 5.22x | -12.0690 (ep 99) |
