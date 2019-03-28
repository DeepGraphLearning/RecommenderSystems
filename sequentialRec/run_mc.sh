CUDA_VISIBLE_DEVICES=0 python -m markovChains.train \
 --data gowalla --model fossil \
 --worker 10 --optim adam --emsize 100 \
 --batch_size 256 --lr 0.01 --lr_decay 0.5 --l2_reg 0. \
 --log_interval 100 --eval_interval 500 \
