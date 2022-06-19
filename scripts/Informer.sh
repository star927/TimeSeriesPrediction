### ETT M
python -u Informer.py --model informer --data ETT --features M --seq_len 48 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 5 --factor 3
python -u Informer.py --model informer --data ETT --features M --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 5
python -u Informer.py --model informer --data ETT --features M --seq_len 96 --label_len 96 --pred_len 96 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 5
python -u Informer.py --model informer --data ETT --features M --seq_len 168 --label_len 168 --pred_len 168 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 5

### ETT S
python -u Informer.py --model informer --data ETT --features S --seq_len 96 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 5
python -u Informer.py --model informer --data ETT --features S --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 5

python -u Informer.py --model informer --data ETT --features S --seq_len 720 --label_len 168 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 5
python -u Informer.py --model informer --data ETT --features S --seq_len 720 --label_len 168 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 5
python -u Informer.py --model informer --data ETT --features S --seq_len 720 --label_len 168 --pred_len 96 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 5
python -u Informer.py --model informer --data ETT --features S --seq_len 720 --label_len 336 --pred_len 168 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 5

### Weather MS
### 参数敏感性实验 factor
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 128 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 3  --factor 7
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 96 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 3  --factor 7
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 64 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 3  --factor 7
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 32 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 3  --factor 7

python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 128 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 3  --factor 3
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 96 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 3  --factor 3
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 64 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 3  --factor 3
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 32 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 3  --factor 3

### 消融实验
### 参数敏感性实验 seq_len
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 128 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 3
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 96 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 3
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 64 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 3
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 32 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 3
### 参数敏感性实验 seq_len
### 参数敏感性实验 factor

python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 128 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn full --distil --exp_num 3
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 96 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn full --distil --exp_num 3
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 64 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn full --distil --exp_num 3
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 32 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn full --distil --exp_num 3

python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 128 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --exp_num 3
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 96 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --exp_num 3
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 64 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --exp_num 3
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 32 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --exp_num 3

python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 128 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --transformer_dec --exp_num 3
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 96 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --transformer_dec --exp_num 3
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 64 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --transformer_dec --exp_num 3
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 32 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --transformer_dec --exp_num 3
### 消融实验

### 参数敏感性实验 label_len
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 128 --label_len 16 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 3
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 128 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 3
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 128 --label_len 64 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 3
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 128 --label_len 96 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 3
### 参数敏感性实验 label_len

### 参数敏感性实验 encoder stack
python -u Informer.py --model informerstack --data Weather_WH --features MS --seq_len 128 --label_len 32 --pred_len 16 --s_layers "[3]" --inp_lens "[0]" --d_layers 1 --attn prob --distil --exp_num 3
python -u Informer.py --model informerstack --data Weather_WH --features MS --seq_len 96 --label_len 32 --pred_len 16 --s_layers "[3]" --inp_lens "[0]" --d_layers 1 --attn prob --distil --exp_num 3
python -u Informer.py --model informerstack --data Weather_WH --features MS --seq_len 64 --label_len 32 --pred_len 16 --s_layers "[3]" --inp_lens "[0]" --d_layers 1 --attn prob --distil --exp_num 3
python -u Informer.py --model informerstack --data Weather_WH --features MS --seq_len 32 --label_len 32 --pred_len 16 --s_layers "[3]" --inp_lens "[0]" --d_layers 1 --attn prob --distil --exp_num 3

python -u Informer.py --model informerstack --data Weather_WH --features MS --seq_len 128 --label_len 32 --pred_len 16 --s_layers "[3, 2]" --inp_lens "[0, 1]" --d_layers 1 --attn prob --distil --exp_num 3
python -u Informer.py --model informerstack --data Weather_WH --features MS --seq_len 96 --label_len 32 --pred_len 16 --s_layers "[3, 2]" --inp_lens "[0, 1]" --d_layers 1 --attn prob --distil --exp_num 3
python -u Informer.py --model informerstack --data Weather_WH --features MS --seq_len 64 --label_len 32 --pred_len 16 --s_layers "[3, 2]" --inp_lens "[0, 1]" --d_layers 1 --attn prob --distil --exp_num 3
python -u Informer.py --model informerstack --data Weather_WH --features MS --seq_len 32 --label_len 32 --pred_len 16 --s_layers "[3, 2]" --inp_lens "[0, 1]" --d_layers 1 --attn prob --distil --exp_num 3

python -u Informer.py --model informerstack --data Weather_WH --features MS --seq_len 128 --label_len 32 --pred_len 16 --s_layers "[3, 2, 1]" --inp_lens "[0, 1, 2]" --d_layers 1 --attn prob --distil --exp_num 3
python -u Informer.py --model informerstack --data Weather_WH --features MS --seq_len 96 --label_len 32 --pred_len 16 --s_layers "[3, 2, 1]" --inp_lens "[0, 1, 2]" --d_layers 1 --attn prob --distil --exp_num 3
python -u Informer.py --model informerstack --data Weather_WH --features MS --seq_len 64 --label_len 32 --pred_len 16 --s_layers "[3, 2, 1]" --inp_lens "[0, 1, 2]" --d_layers 1 --attn prob --distil --exp_num 3
python -u Informer.py --model informerstack --data Weather_WH --features MS --seq_len 32 --label_len 32 --pred_len 16 --s_layers "[3, 2, 1]" --inp_lens "[0, 1, 2]" --d_layers 1 --attn prob --distil --exp_num 3
### 参数敏感性实验


### 时间效率
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 512 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 3 --train_epochs 1
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 512 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn full --distil --exp_num 3 --train_epochs 1
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 256 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 3 --train_epochs 1
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 256 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn full --distil --exp_num 3 --train_epochs 1

python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 128 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 3 --train_epochs 1
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 128 --label_len 32 --pred_len 16 --e_layers 3 --d_layers 1 --attn prob --distil --exp_num 3 --train_epochs 1
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 128 --label_len 32 --pred_len 16 --e_layers 4 --d_layers 1 --attn prob --distil --exp_num 3 --train_epochs 1
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 128 --label_len 32 --pred_len 16 --e_layers 5 --d_layers 1 --attn prob --distil --exp_num 3 --train_epochs 1

python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 128 --label_len 32 --pred_len 16 --e_layers 2 --d_layers 1 --attn prob --exp_num 3 --train_epochs 1
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 128 --label_len 32 --pred_len 16 --e_layers 3 --d_layers 1 --attn prob --exp_num 3 --train_epochs 1
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 128 --label_len 32 --pred_len 16 --e_layers 4 --d_layers 1 --attn prob --exp_num 3 --train_epochs 1
python -u Informer.py --model informer --data Weather_WH --features MS --seq_len 128 --label_len 32 --pred_len 16 --e_layers 5 --d_layers 1 --attn prob --exp_num 3 --train_epochs 1
### 时间效率