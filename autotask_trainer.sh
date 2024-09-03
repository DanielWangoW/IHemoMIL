python train.py \
  --is_train True \
  --dataset "mp_ppg_VitalDB" \
  --data_path "data" \
  --checkpoint "checkpoint" \
  --channel 1 \
  --backbone "inceptiontime"\
  --pooling "gap" \
  --d_model 128 \
  --apply_positional_encoding True \
  --batch_size 2048 \
  --epochs 1500 \
  --learning_rate 0.001 \
  --use_gpu True \
  --gpu_id 1

python train.py \
  --is_train True \
  --dataset "mp_ppg_VitalDB" \
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

  python train.py \
  --is_train True \
  --dataset "mp_ppg_VitalDB" \
  --data_path "data" \
  --checkpoint "checkpoint" \
  --channel 1 \
  --backbone "inceptiontime"\
  --pooling "conj" \
  --d_model 128 \
  --apply_positional_encoding True \
  --batch_size 2048 \
  --epochs 1500 \
  --learning_rate 0.001 \
  --use_gpu True \
  --gpu_id 1

python train.py \
  --is_train True \
  --dataset "mp_ppg_VitalDB" \
  --data_path "data" \
  --checkpoint "checkpoint" \
  --channel 1 \
  --backbone "inceptiontime"\
  --pooling "rconj" \
  --d_model 128 \
  --apply_positional_encoding True \
  --batch_size 2048 \
  --epochs 1500 \
  --learning_rate 0.001 \
  --use_gpu True \
  --gpu_id 1

