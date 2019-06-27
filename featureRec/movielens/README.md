### AutoInt code for movielens dataset(support multi-value field)

This is the tensorflow implementation of **AutoInt** on movielens dataset which has a multi-value field (genre).

Compared with the code for other 3 big datasets, we slightly modify some functions in `code/train.py` and `code/model.py`. We also provide the data processing code for movielens in `data/preprocess.py`

#### Preprocessing

```
cd data
python preprocess.py
```

#### Train and Test

```
CUDA_VISIBLE_DEVICES=1 python -m code.train \
                       --data_path data  \
                       --blocks 3 --heads 2  --block_shape "[64, 64, 64]" \
                       --is_save --has_residual \
                       --save_path ./models/movie/b3h2_64x64x64/ \
                       --field_size 7  --run_times 1 \
                       --dropout_keep_prob "[0.6, 0.9]" \
                       --epoch 50 --batch_size 1024 \

```

