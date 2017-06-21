# LSTM + CONTINUOUS CACHE 
1 layer, 512 hidden size, 1 epoch + theta = 0.3, lambda (linear interpolation) = 0.7  
All perplexity and accuracy results are computed on lambada_test

### Target word PoS tag

![perp_pos](perp_pos.png)

![perp_pos](acc_pos.png)

### Target word appears in tag context

![perp_context](perp_context.png)

![acc_context](acc_context.png)

### Distance (in #words) to previous target word mentions in the context

![perp_distance](perp_distance.png)

![acc_distance](acc_distance.png)

### Number of mentions of the target word in the context

![perp_repetition](perp_repetition.png)

![acc_repetition](acc_repetition.png)

