# Info
Source: reddit

Date: 2021/11/21

# Post header
Cedille, the largest French language model (6b), released in open source  
We have spent the last 3 months of our lives, teraFLOPs of compute and gone through 300gb of text to bring you Cedille:

>Ce que j'aime quand je mange une baguette, c'est quand celle-ci est craquante. Je ne saurais définir le terme "craquant" mais je sais que lorsque c'est le cas, je peux être sûre que la baguette est bonne.

The entirety of French spirit captured in measly 6B parameters! 🇫🇷🥖

More seriously, we are super excited to share Cedille, the so far largest French language model: https://en.cedille.ai

# Comments section

>We mostly stuck to the finetuning recommendations provided by GPT-J: https://github.com/kingoflolz/mesh-transformer-jax/blob/master/howto_finetune.md
>We trained on 300gb of uncompressed text (= 78B tokens), which took 12.5 days for the final model on a v3-128 TPU.
>The Google TRC program provided us with the TPU compute. There's no public price for renting v3-128 TPU.

>There are only few available benchmarks in French language (especially for zero-shot tasks and the like). Machine translated benchmarks were of low quality unfortunately :/

>Existing models usually did some amount of fine-tuning on training data. We wanted to avoid this (for many different reasons). But it makes it much harder to compare models with each other.

>For zero-shot and few-shot tasks many different ways of splitting/processing output are commonly used. Additionally, non-greedy sampling may introduce variance to the results.

>We ran into pretty high costs of running benchmarks for GPT-3 (~$1000).

# Notes
None

# Though:
Very useful for fench based NLP projects.