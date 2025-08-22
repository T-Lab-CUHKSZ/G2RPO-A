# G2RPO-A
[Preprint] G2RPO-A: Guided Group Relative Policy Optimization with Adaptive Guidance
## Installation
The basic framework we use is Open-R1, the main installation can follow the instruction of its repository. 

To avoid version conflict, we directly provide the local open-r1 repository we use.

For additional changes, we change the following version of packages:
- Use the trl of version 0.15.2
- Update the hugggingface package so that Qwen3 series can be recognized. 

## Main programs
To run our agorithm, there are two main programs.
### Trainer
The trainer is the core implement of our algorithm, you can find it in [here](./trainer/grpo_trainer_adaptive_final.py).

Replace the trainer used in trl into our proposed trainer. 
### Training script
Because we change the random sampler that open-r1 uses to a sequence sampler, we provide our own training script in [here](./src/grpo.py).

