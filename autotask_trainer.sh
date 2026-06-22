python train.py \
  --dataset "web_traffic" \
  --data_path "data" \
  --checkpoint "checkpoint" \
  --channel 1 \
  --backbone "inceptiontime"\
  --pooling "rap" \
  --d_model 128 \
  --apply_positional_encoding True \
  --batch_size 2048 \
  --epochs 1500 \
  --learning_rate 0.001 \
  --use_gpu True \
  --gpu_id 1

