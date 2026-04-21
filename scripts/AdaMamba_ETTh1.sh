export CUDA_VISIBLE_DEVICES=3 #2,3,4,5,6,7

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LTSF" ]; then
    mkdir ./logs/LTSF
fi

seq_len=96
root_path_name=./dataset/ETT-small
data_path_name=ETTm1.csv
model_id_name=ETTm1_LTSF
data_name=ETTm1
patience=3
random_seed=2021

patch_lens_list=(
  "96,18,9"
)

for model_name in AdaMamba
do
for pred_len in 96 
do
for e_layers in 2
do
for d_model in 32
do
for d_ff in 32
do
for batch_size in 24
do
for learning_rate in 0.0001
do
for patch_lens in "${patch_lens_list[@]}"  # 新增的 patch_lens 循环
do

    
    patch_lens_log_name=${patch_lens//,/_}

    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --devices 0 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --Phy_input_channels 127 \
      --Phy_output_channels 127 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --e_layers $e_layers \
      --n_heads 8 \
      --d_model $d_model \
      --dim_feq $d_model \
      --dim_pitch $d_model \
      --d_ff $d_ff \
      --patch_len 16 \
      --patch_lens "$patch_lens" \
      --stride 16 \
      --des 'Exp' \
      --train_epochs 100 \
      --patience $patience \
      --kernel_size 25 \
      --lradj type1 \
      --grid_size 5 \
      --spline_order 3 \
      --emb 96 \
      --period_list 24 48 72 \
      --itr 1 \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      > logs/LTSF/${model_name}_${model_id_name}_${seq_len}_${pred_len}_${e_layers}_${d_model}_${d_ff}_${learning_rate}_[${patch_lens_log_name}]_LTSF.log 

done
done
done
done
done
done
done
done