# JEN-1-COMPOSER-pytorch(wip)
![model architecture](https://github.com/0417keito/JEN-1-COMPOSER-pytorch/blob/main/JEN1-Composer.jpg)

the unofficial implementation of JEN-1-COMPOSER(https://arxiv.org/abs/2310.19180v2)
this arch is very interesting.
The approach uses demixed audio as input, modelling each individual track independently, but also using each as a condition.
I believe that this approach can be applied beyond music generation.
The following may be of interest in terms of the input being demixed audio.
https://github.com/mir-aidj/all-in-one

## Citations
```bibtex
@misc{2310.19180,
Author = {Yao Yao and Peike Li and Boyu Chen and Alex Wang},
Title = {JEN-1 Composer: A Unified Framework for High-Fidelity Multi-Track Music Generation},
Year = {2023},
Eprint = {arXiv:2310.19180},
}
```
