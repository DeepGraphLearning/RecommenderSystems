CUDA_VISIBLE_DEVICES=1 python -m code.train \
                       --data_path data  \
                       --blocks 3 --heads 2  --block_shape "[64, 64, 64]" \
                       --is_save --has_residual \
                       --save_path ./models/movie/b3h2_64x64x64/ \
                       --field_size 7  --run_times 1 \
                       --dropout_keep_prob "[0.6, 0.9]" \
                       --epoch 50 --batch_size 1024 \
