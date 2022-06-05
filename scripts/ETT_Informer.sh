### M

python -u Informer.py --model informer --data ETT --features M --seq_len 48 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 5 --factor 3

python -u Informer.py --model informer --data ETT --features M --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 5

python -u Informer.py --model informer --data ETT --features M --seq_len 168 --label_len 168 --pred_len 168 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 5

python -u Informer.py --model informer --data ETT --features M --seq_len 168 --label_len 168 --pred_len 336 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 5

python -u Informer.py --model informer --data ETT --features M --seq_len 336 --label_len 336 --pred_len 720 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 5

### S

python -u Informer.py --model informer --data ETT --features S --seq_len 720 --label_len 168 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 5

python -u Informer.py --model informer --data ETT --features S --seq_len 720 --label_len 168 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 5

python -u Informer.py --model informer --data ETT --features S --seq_len 720 --label_len 336 --pred_len 168 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 5

python -u Informer.py --model informer --data ETT --features S --seq_len 720 --label_len 336 --pred_len 336 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 5

python -u Informer.py --model informer --data ETT --features S --seq_len 720 --label_len 336 --pred_len 720 --e_layers 2 --d_layers 1 --attn prob --distil --exp_num 5

### Transformer M

python -u Informer.py --model informer --data ETT --features M --seq_len 48 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn full --transformer_dec --exp_num 5 --factor 3

python -u Informer.py --model informer --data ETT --features M --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn full --transformer_dec --exp_num 5

python -u Informer.py --model informer --data ETT --features M --seq_len 168 --label_len 168 --pred_len 168 --e_layers 2 --d_layers 1 --attn full --transformer_dec --exp_num 5

python -u Informer.py --model informer --data ETT --features M --seq_len 168 --label_len 168 --pred_len 336 --e_layers 2 --d_layers 1 --attn full --transformer_dec --exp_num 5

python -u Informer.py --model informer --data ETT --features M --seq_len 336 --label_len 336 --pred_len 720 --e_layers 2 --d_layers 1 --attn full --transformer_dec --exp_num 5

### Transformer S

python -u Informer.py --model informer --data ETT --features S --seq_len 720 --label_len 168 --pred_len 24 --e_layers 2 --d_layers 1 --attn full --transformer_dec --exp_num 5

python -u Informer.py --model informer --data ETT --features S --seq_len 720 --label_len 168 --pred_len 48 --e_layers 2 --d_layers 1 --attn full --transformer_dec --exp_num 5

python -u Informer.py --model informer --data ETT --features S --seq_len 720 --label_len 336 --pred_len 168 --e_layers 2 --d_layers 1 --attn full --transformer_dec --exp_num 5

python -u Informer.py --model informer --data ETT --features S --seq_len 720 --label_len 336 --pred_len 336 --e_layers 2 --d_layers 1 --attn full --transformer_dec --exp_num 5

python -u Informer.py --model informer --data ETT --features S --seq_len 720 --label_len 336 --pred_len 720 --e_layers 2 --d_layers 1 --attn full --transformer_dec --exp_num 5