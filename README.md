# Optimizers from scratch on a looped transformer

I wrote SGD, momentum, Adam and a simple Muon-style optimizer in plain PyTorch (no `torch.optim`), checked them against PyTorch's versions, and used them to train a tiny transformer that reuses one block N times (1, 4, 8 loops) on modular addition, (a + b) mod 17. I then compared the looped model with a model that has N separate blocks.

Motivated by Zhang, Lu and Zhao, *On Enhancing Expressive Power via Compositions of Single Fixed-Size ReLU Network* (ICML 2023, [arXiv 2301.12353](https://arxiv.org/abs/2301.12353)). Built and run on a phone using Google Colab.

Full write-up (method, results, comparison with the paper, limitations): [Project write-up](https://docs.google.com/document/d/1fZBr5tCavJHRc8d9ScSkijZIvPvcmeAM/edit)

## Main findings

- Muon was best at 1 loop but became unstable at 4 and 8 loops, in both the looped and the separate-block models.
- Adam was the most stable and changed little with loop count.
- SGD and momentum improved as loops increased.
- One shared block matched the training loss of separate blocks with 4 to 8 times fewer parameters.
- All models mostly memorized the training data (test accuracy 0.00 to 0.59, chance about 0.06).

## Limitations

Small model and task, 3 seeds, untuned learning rates, 400 steps, simplified Muon, and a phone-based setup. Explanations for the results are hypotheses, not tested.

## Run it

Upload `looped_optimizers_v2.ipynb` to Google Colab (or open it from GitHub in Colab) and run all cells. CPU is enough.

The code was written with AI assistance (Claude). I ran it, produced the results and wrote the analysis.
