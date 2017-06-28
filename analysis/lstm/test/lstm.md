# LSTM
1 layer, 512 hidden size, 1 epoch  
All perplexity and accuracy results are computed on lambada_test  
Perplexity: 7286.43, Accuracy: 0.1 %  

### Target word (or its lemma) PoS tag

|  PN  |  CN  | V   | ADJ | ADV | O  |
|:----:|:----:|-----|-----|-----|----|
| 2248 | 2215 | 367 | 222 | 66  | 35 | 

![perp_pos](perp_pos.png)

![perp_pos](acc_pos.png)

### Target word appears in the context

|  Yes |  No |
|:----:|:---:|
| 4363 | 790 | 

![perp_context](perp_context.png)

![acc_context](acc_context.png)

### Distance (in #words) to previous target word mentions in the context

|  No | [1,10] | (10,20] | (20,30] | (30,40] | (40,50] | (50,60] | (60,70] | (70,80] | +80 |
|:---:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:---:|
| 790 |   13   |   504   |   678   |   797   |   863   |   728   |   461   |   189   |  69 | 

![perp_distance](perp_distance.png)

![acc_distance](acc_distance.png)

### Number of mentions of the target word in the context

|  No |   1  |  2  |  3  |  4 | 5 |
|:---:|:----:|:---:|:---:|:--:|:-:|
| 790 | 3367 | 837 | 135 | 23 | 1 |

![perp_repetition](perp_repetition.png)

![acc_repetition](acc_repetition.png)

