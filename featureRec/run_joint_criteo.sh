CUDA_VISIBLE_DEVICES=0 python -m autoint.train \
                       --data_path data --data Criteo \
                       --blocks 3 --heads 2  --block_shape "[64, 64, 64]" \
                       --is_save --has_residual \
                       --save_path ./models/Criteo/b3h2_400x2_64x64x64/ \
                       --field_size 39  --run_times 1 \
                       --deep_layers "[400, 400]" --dropout_keep_prob "[1, 1, 1]" \
                       --epoch 3 --batch_size 1024 \

