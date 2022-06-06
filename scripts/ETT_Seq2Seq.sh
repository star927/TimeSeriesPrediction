# GRU M
python -u Seq2Seq.py --cell GRU --data ETT --features M --seq_len 96 --pred_len 24 --num_hidden 512 --dropout 0.05 --learning_rate 0.0001 --train_epochs 6  --exp_num 5
python -u Seq2Seq.py --cell GRU --data ETT --features M --seq_len 96 --pred_len 48 --num_hidden 512 --dropout 0.05 --learning_rate 0.0001 --train_epochs 6  --exp_num 5
python -u Seq2Seq.py --cell GRU --data ETT --features M --seq_len 96 --pred_len 96 --num_hidden 512 --dropout 0.05 --learning_rate 0.0001 --train_epochs 6  --exp_num 5
python -u Seq2Seq.py --cell GRU --data ETT --features M --seq_len 168 --pred_len 168 --num_hidden 512 --dropout 0.05 --learning_rate 0.0001 --train_epochs 6  --exp_num 5

# LSTM M
python -u Seq2Seq.py --cell LSTM --data ETT --features M --seq_len 96 --pred_len 24 --num_hidden 512 --dropout 0.05 --learning_rate 0.0001 --train_epochs 6  --exp_num 5
python -u Seq2Seq.py --cell LSTM --data ETT --features M --seq_len 96 --pred_len 48 --num_hidden 512 --dropout 0.05 --learning_rate 0.0001 --train_epochs 6  --exp_num 5
python -u Seq2Seq.py --cell LSTM --data ETT --features M --seq_len 96 --pred_len 96 --num_hidden 512 --dropout 0.05 --learning_rate 0.0001 --train_epochs 6  --exp_num 5
python -u Seq2Seq.py --cell LSTM --data ETT --features M --seq_len 168 --pred_len 168 --num_hidden 512 --dropout 0.05 --learning_rate 0.0001 --train_epochs 6  --exp_num 5

# GRU S
python -u Seq2Seq.py --cell GRU --data ETT --features S --seq_len 96 --pred_len 24 --num_hidden 512 --dropout 0.05 --learning_rate 0.0001 --train_epochs 6  --exp_num 5
python -u Seq2Seq.py --cell GRU --data ETT --features S --seq_len 96 --pred_len 48 --num_hidden 512 --dropout 0.05 --learning_rate 0.0001 --train_epochs 6  --exp_num 5
python -u Seq2Seq.py --cell GRU --data ETT --features S --seq_len 96 --pred_len 96 --num_hidden 512 --dropout 0.05 --learning_rate 0.0001 --train_epochs 6  --exp_num 5
python -u Seq2Seq.py --cell GRU --data ETT --features S --seq_len 168 --pred_len 168 --num_hidden 512 --dropout 0.05 --learning_rate 0.0001 --train_epochs 6  --exp_num 5

# LSTM S
python -u Seq2Seq.py --cell LSTM --data ETT --features S --seq_len 96 --pred_len 24 --num_hidden 512 --dropout 0.05 --learning_rate 0.0001 --train_epochs 6  --exp_num 5
python -u Seq2Seq.py --cell LSTM --data ETT --features S --seq_len 96 --pred_len 48 --num_hidden 512 --dropout 0.05 --learning_rate 0.0001 --train_epochs 6  --exp_num 5
python -u Seq2Seq.py --cell LSTM --data ETT --features S --seq_len 96 --pred_len 96 --num_hidden 512 --dropout 0.05 --learning_rate 0.0001 --train_epochs 6  --exp_num 5
python -u Seq2Seq.py --cell LSTM --data ETT --features S --seq_len 168 --pred_len 168 --num_hidden 512 --dropout 0.05 --learning_rate 0.0001 --train_epochs 6  --exp_num 5