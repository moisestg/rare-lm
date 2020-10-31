# Broad Discourse Context for language modeling

## About the code: 
This repo features a simple implementation of a [pointer sentinel mixture](https://arxiv.org/abs/1609.07843) language model (PSMM) in Tensorflow v1.4 (`psmm` folder) which may serve you as a starting point for your own projects. It also includes a vanilla RNNLM (`vanilla` folder).

## About the thesis:
The work is focused on analyizing the nature of the [LAMBADA dataset](https://arxiv.org/abs/1606.06031) and exploring techniques that may increase the performance on this task. Some key points of our work:
- LAMBADA focuses on probing the ability of language models to handle long-range dependencies. However, rare words (e.g. named entities) play an important role on the overall performance that is usually overlooked when all results are averaged.
- Pointer-based models yield state-of-the-art performance on LAMBADA. PSMM produces comparable results while having a high degree of adaptability. 

For more details and results, check the full thesis [here](../master/report/MasterThesis_MTorres.pdf) or [there](https://doi.org/10.3929/ethz-b-000223923).
