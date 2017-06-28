# LSTM
1 layer, 512 hidden size, 1 epoch  
All perplexity and accuracy results are computed on lambada_control  
Perplexity: 134.22, Accuracy: 21.9 %  

### Target word (or its lemma) PoS tag

|  CN  |  O   | V   | ADV | ADV | PN |
|:----:|:----:|-----|-----|-----|----|
| 2400 | 829  | 728 | 440 | 308 | 295| 

![perp_pos](perp_pos.png)

![perp_pos](acc_pos.png)

### Target word appears in the context

|  Yes |  No |
|:----:|:---:|
| 837  | 4163| 

![perp_context](perp_context.png)

![acc_context](acc_context.png)

### Distance (in #words) to previous target word mentions in the context

|  No | (30,40]| (20,30] | (40,50] | (10,20] | (50,60] | (60,70] | [1,10]  | (70,80] | +80 |
|:---:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:---:|
| 4163|   155  |   141   |   137   |   110   |   97    |   83    |   47    |   43    |  24 | 

![perp_distance](perp_distance.png)

![acc_distance](acc_distance.png)

### Number of mentions of the target word in the context

|  No |   1  |  2  |  3  |  4 | 5 | +5 |
|:---:|:----:|:---:|:---:|:--:|:-:|:--:|
| 4163| 572  | 163 | 57  | 27 | 10| 8  |

![perp_repetition](perp_repetition.png)

![acc_repetition](acc_repetition.png)

