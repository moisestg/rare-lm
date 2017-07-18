# EXPERIMENTS

## LAMBADA

### MULTILSTM

* `--emb_size 200 --vocab_size 93215 --num_layers 1 --hidden_size 512 --num_steps 35 --optimizer "grad_desc" --learning_rate 1.0 --keep_prob 1.0 --clip_norm 8.0 --batch_size 128 --num_epochs 1 --train_path "./lambada-dataset/lambada_train.txt" --dev_path "./lambada-dataset/lambada_development_plain_text.txt"`: valid perplexity:  / test perplexity: 8565.413, test accuracy: 0.1%

* BEST (./runs/1498636986/checkpoints/model-906870): `--emb_size 512 --vocab_size 93215 --num_layers 1 --hidden_size 512 --num_steps 35 --optimizer "grad_desc" --learning_rate 1.0 --learning_rate_decay 0.5 --keep_prob 1.0 --clip_norm 8.0 --batch_size 64 --num_epochs 10 --train_path "./lambada-dataset/lambada_train.txt" --dev_path "./lambada-dataset/lambada_development_plain_text.txt"`: valid perplexity: 4651.18, valid accuracy: 0.3% / test perplexity: 5195.83, test accuracy: 0.3%

### MULTILSTM + CONTINUOUS CACHE

* `--theta 0.3 --lambda 0.7` valid perplexity:  / test perplexity: 119.713, test accuracy: 14.6%



## PENN TREE BANK

* `./run.py --pretrained_emb "word2vec" --emb_size 200 --vocab_size 10000 --num_layers 2 --hidden_size 200 --num_steps 20 --optimizer "grad_desc" --learning_rate 1.0 --learning_rate_decay 0.5 --keep_prob 1.0 --clip_norm 5.0 --batch_size 20 --num_epochs 13`: valid perplexity: 109.3 / test perplexity: 105.2
