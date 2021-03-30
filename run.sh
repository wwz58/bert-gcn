# # bert_spc
# CUDA_VISIBLE_DEVICES=2 python train.py --model_name bert_spc --dataset restaurant  --valset_ratio 0.15 --pretrained_bert_name '/mnt/2/ckh/BERT系列预训练模型/bert-base-uncased/' --bert_dim 768
# CUDA_VISIBLE_DEVICES=3 python train.py --model_name bert_spc --dataset laptop  --valset_ratio 0.15 --pretrained_bert_name '/mnt/2/ckh/BERT系列预训练模型/bert-base-uncased/' --bert_dim 768

# for fn in ./datasets/acl-14-short-data/*.raw; doj
#     head -n90  $fn > "$fn.debug" 
#     echo "$fn.debug" 
# done

# for fn in ./datasets/semeval14/*; do
#     head -n90 $fn > "$fn.debug" 
#     echo "$fn.debug" 
# done

# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset restaurant \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset laptop \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     >/dev/null 2>&1 &

# wait
# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset twitter \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name bert_spc \
#     --dataset restaurant \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     >/dev/null 2>&1 &

# wait
# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name bert_spc \
#     --dataset twitter \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name bert_spc \
#     --dataset laptop \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     >/dev/null 2>&1 &

# wait
# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name aen_bert \
#     --dataset restaurant \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name aen_bert \
#     --dataset laptop \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     >/dev/null 2>&1 &

# wait
# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name aen_bert \
#     --dataset twitter \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     >/dev/null 2>&1 &

# # 使用pretrain的rest lap
# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset restaurant \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     --pretrained_bert_name /mnt/sda1/wwz/absa/pretrained_bert/ada/restaurants_10mio_ep3 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset laptop \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     --pretrained_bert_name /mnt/sda1/wwz/absa/pretrained_bert/ada/laptops_1mio_ep30 \
#     >/dev/null 2>&1 &

# # bert-ada 即 使用预训练语料的bert-spc
# wait
# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name bert_spc \
#     --dataset restaurant \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     --pretrained_bert_name /mnt/sda1/wwz/absa/pretrained_bert/ada/restaurants_10mio_ep3 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name bert_spc \
#     --dataset laptop \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     --pretrained_bert_name /mnt/sda1/wwz/absa/pretrained_bert/ada/laptops_1mio_ep30 \
#     >/dev/null 2>&1 &

# # bert-pt aka. 用DM和RRC pt预训练的rest 和lap 进行bert_spcwait
# wait
# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name bert_spc \
#     --dataset restaurant \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     --pretrained_bert_name /mnt/sda1/wwz/absa/pretrained_bert/pt/rest_pt \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name bert_spc \
#     --dataset laptop \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     --pretrained_bert_name /mnt/sda1/wwz/absa/pretrained_bert/pt/laptop_pt \
#     >/dev/null 2>&1 &

# 普通的bert-gcn调seed
# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset restaurant \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     --seed 42 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset laptop \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     --seed 42 \
#     >/dev/null 2>&1 &

# wait
# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset twitter \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     --seed 42 \
#     >/dev/null 2>&1 &



# # non-bert based methods
# # aoa
# CUDA_VISIBLE_DEVICES=0  python train.py \
#     --model_name aoa \
#     --learning_rate 1e-3 \
#     --num_epoch 10 \
#     --batch_size 32 \
#     --dataset restaurant \
#     --cross_val_fold=5 \
#     --seed 2020 \
#     >/dev/null 2>&1 

# CUDA_VISIBLE_DEVICES=0  python train.py \
#     --model_name aoa \
#     --dataset laptop \
#     --learning_rate 1e-3 \
#     --num_epoch 10 \
#     --batch_size 32 \
#     --cross_val_fold=5 \
#     --seed 2020 \
#     >/dev/null 2>&1 

# CUDA_VISIBLE_DEVICES=0  python train.py \
#     --model_name aoa \
#     --learning_rate 1e-3 \
#     --num_epoch 10 \
#     --batch_size 32 \
#     --dataset twitter \
#     --cross_val_fold=5 \
#     --seed 2020 \
#     >/dev/null 2>&1 

# # mgan
# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name mgan \
#     --learning_rate 1e-3 \
#     --num_epoch 15 \
#     --batch_size 32 \
#     --dataset restaurant \
#     --cross_val_fold=5 \
#     --seed 2020 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name mgan \
#     --learning_rate 1e-3 \
#     --num_epoch 15 \
#     --batch_size 32 \
#     --dataset laptop \
#     --cross_val_fold=5 \
#     --seed 2020 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name mgan \
#     --learning_rate 1e-3 \
#     --num_epoch 15 \
#     --batch_size 32 \
#     --dataset twitter \
#     --cross_val_fold=5 \
#     --seed 2020 \
#     >/dev/null 2>&1 &

# # td_lstm
# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name td_lstm \
#     --learning_rate 1e-3 \
#     --num_epoch 10 \
#     --batch_size 32 \
#     --dataset restaurant \
#     --cross_val_fold=5 \
#     --seed 2020 \
#     >/dev/null 2>&1 &

# wait
# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name td_lstm \
#     --learning_rate 1e-3 \
#     --num_epoch 10 \
#     --batch_size 32 \
#     --dataset laptop \
#     --cross_val_fold=5 \
#     --seed 2020 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name td_lstm \
#     --learning_rate 1e-3 \
#     --num_epoch 10 \
#     --batch_size 32 \
#     --dataset twitter \
#     --cross_val_fold=5 \
#     --seed 2020 \
#     >/dev/null 2>&1 &



# # gcn layers 非cross validation
# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset laptop \
#     --bert_dim 768 \
#     --valset_ration 0.2 \
#     --gnn_step 1 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset laptop \
#     --bert_dim 768 \
#     --valset_ration 0.2 \
#     --gnn_step 2 \
#     >/dev/null 2>&1 &
# wait
# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset laptop \
#     --bert_dim 768 \
#     --valset_ration 0.2 \
#     --gnn_step 3 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset laptop \
#     --bert_dim 768 \
#     --valset_ration 0.2 \
#     --gnn_step 4 \
#     >/dev/null 2>&1 &
# wait
# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset laptop \
#     --bert_dim 768 \
#     --valset_ration 0.2 \
#     --gnn_step 5 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset laptop \
#     --bert_dim 768 \
#     --valset_ration 0.2 \
#     --gnn_step 6 \
#     >/dev/null 2>&1 &



# wait
# # max_pool instead of MHA
# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset restaurant \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     --pool_tp max_pool \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset laptop \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     --pool_tp max_pool \
#     >/dev/null 2>&1 &

# wait
# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset twitter \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     --pool_tp max_pool \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset restaurant \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     --pool_tp max_pool \
#     --pretrained_bert_name /mnt/sda1/wwz/absa/pretrained_bert/ada/restaurants_10mio_ep3 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset laptop \
#     --cross_val_fold=5 \
#     --bert_dim 768 \
#     --pool_tp max_pool \
#     --pretrained_bert_name /mnt/sda1/wwz/absa/pretrained_bert/ada/laptops_1mio_ep30 \
#     >/dev/null 2>&1 &




# gcn layers rest 非cross validation
# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset restaurant \
#     --bert_dim 768 \
#     --valset_ration 0.2 \
#     --gnn_step 1 \
#     --pretrained_bert_name /mnt/sda1/wwz/absa/pretrained_bert/ada/restaurants_10mio_ep3 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset restaurant \
#     --bert_dim 768 \
#     --valset_ration 0.2 \
#     --gnn_step 2 \
#     --pretrained_bert_name /mnt/sda1/wwz/absa/pretrained_bert/ada/restaurants_10mio_ep3 \
#     >/dev/null 2>&1 &
# wait
# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset restaurant \
#     --bert_dim 768 \
#     --valset_ration 0.2 \
#     --gnn_step 3 \
#     --pretrained_bert_name /mnt/sda1/wwz/absa/pretrained_bert/ada/restaurants_10mio_ep3 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset restaurant \
#     --bert_dim 768 \
#     --valset_ration 0.2 \
#     --gnn_step 4 \
#     --pretrained_bert_name /mnt/sda1/wwz/absa/pretrained_bert/ada/restaurants_10mio_ep3 \
#     >/dev/null 2>&1 &
# wait
# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset restaurant \
#     --bert_dim 768 \
#     --valset_ration 0.2 \
#     --gnn_step 5 \
#     --pretrained_bert_name /mnt/sda1/wwz/absa/pretrained_bert/ada/restaurants_10mio_ep3 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset restaurant \
#     --bert_dim 768 \
#     --valset_ration 0.2 \
#     --gnn_step 6 \
#     --pretrained_bert_name /mnt/sda1/wwz/absa/pretrained_bert/ada/restaurants_10mio_ep3 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset restaurant \
#     --bert_dim 768 \
#     --valset_ration 0.2 \
#     --gnn_step 8 \
#     --pretrained_bert_name /mnt/sda1/wwz/absa/pretrained_bert/ada/restaurants_10mio_ep3 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset restaurant \
#     --bert_dim 768 \
#     --valset_ration 0.2 \
#     --gnn_step 10 \
#     --pretrained_bert_name /mnt/sda1/wwz/absa/pretrained_bert/ada/restaurants_10mio_ep3 \
#     >/dev/null 2>&1 &

# # freeze bert - bert_gcn
# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset restaurant \
#     --bert_dim 768 \
#     --n_repeat 5 \
#     --freeze_bert True \
#     --learning_rate 1e-3 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset laptop \
#     --bert_dim 768 \
#     --n_repeat 5 \
#     --freeze_bert True \
#     --learning_rate 1e-3 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset twitter \
#     --bert_dim 768 \
#     --n_repeat 5 \
#     --freeze_bert True \
#     --learning_rate 1e-3 \
#     >/dev/null 2>&1 &

# # freeze bert - aen_bert
# CUDA_VISIBLE_DEVICES=3 nohup python train.py \
#     --model_name aen_bert \
#     --dataset restaurant \
#     --bert_dim 768 \
#     --n_repeat 5 \
#     --freeze_bert True \
#     --learning_rate 1e-3 \
#     >/dev/null 2>&1 &

# wait
# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name aen_bert \
#     --dataset laptop \
#     --bert_dim 768 \
#     --n_repeat 5 \
#     --freeze_bert True \
#     --learning_rate 1e-3 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name aen_bert \
#     --dataset twitter \
#     --bert_dim 768 \
#     --n_repeat 5 \
#     --freeze_bert True \
#     --learning_rate 1e-3 \
#     >/dev/null 2>&1 &

# # freeze bert - ada-bert
# CUDA_VISIBLE_DEVICES=2 nohup python train.py \
#     --model_name bert_spc \
#     --dataset restaurant \
#     --bert_dim 768 \
#     --n_repeat 5 \
#     --freeze_bert True \
#     --learning_rate 1e-3 \
#     --pretrained_bert_name /home/wwz/absa/LCF-ABSA/data/restaurants_10mio_ep3 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python train.py \
#     --model_name bert_spc \
#     --dataset laptop \
#     --bert_dim 768 \
#     --n_repeat 5 \
#     --freeze_bert True \
#     --learning_rate 1e-3 \
#     --pretrained_bert_name /home/wwz/absa/LCF-ABSA/data/laptops_1mio_ep30 \
#     >/dev/null 2>&1 &

# # freeze bert - ada-bert_gcn
# wait
# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset restaurant \
#     --bert_dim 768 \
#     --n_repeat 5 \
#     --freeze_bert True \
#     --learning_rate 1e-3 \
#     --pretrained_bert_name /home/wwz/absa/LCF-ABSA/data/restaurants_10mio_ep3 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset laptop \
#     --bert_dim 768 \
#     --n_repeat 5 \
#     --freeze_bert True \
#     --learning_rate 1e-3 \
#     --pretrained_bert_name /home/wwz/absa/LCF-ABSA/data/laptops_1mio_ep30 \
#     >/dev/null 2>&1 &

# # freeze bert - bert_spc
# CUDA_VISIBLE_DEVICES=2 nohup python train.py \
#     --model_name bert_spc \
#     --dataset restaurant \
#     --bert_dim 768 \
#     --n_repeat 5 \
#     --freeze_bert True \
#     --learning_rate 1e-3 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python train.py \
#     --model_name bert_spc \
#     --dataset laptop \
#     --bert_dim 768 \
#     --n_repeat 5 \
#     --freeze_bert True \
#     --learning_rate 1e-3 \
#     >/dev/null 2>&1 &

# wait
# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#     --model_name bert_spc \
#     --dataset twitter \
#     --bert_dim 768 \
#     --n_repeat 5 \
#     --freeze_bert True \
#     --learning_rate 1e-3 \
#     >/dev/null 2>&1 &

# # random reproduce get test result bert_gcn
    
CUDA_VISIBLE_DEVICES=1 nohup python train.py \
    --model_name bert_gcn \
    --dataset restaurant \
    --bert_dim 768 \
    --freeze_bert False \
    --learning_rate 2e-5 \
    --from_xml \
    --step_eval \
    --log_step 20 \
    >/dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python train.py \
    --model_name bert_gcn \
    --dataset laptop \
    --bert_dim 768 \
    --freeze_bert True \
    --learning_rate 2e-5 \
    --from_xml \
    --step_eval \
    --log_step 20 \
    >/dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python train.py \
    --model_name bert_gcn \
    --dataset twitter \
    --bert_dim 768 \
    --freeze_bert True \
    --learning_rate 2e-5 \
    --step_eval \
    --log_step 20 \
    >/dev/null 2>&1 &

# wait 
# random reproduce get test result bert_spc

# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name bert_spc \
#     --dataset restaurant \
#     --bert_dim 768 \
#     --freeze_bert False \
#     --learning_rate 2e-5 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python train.py \
#     --model_name bert_spc \
#     --dataset laptop \
#     --bert_dim 768 \
#     --freeze_bert True \
#     --learning_rate 2e-5 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python train.py \
#     --model_name bert_spc \
#     --dataset twitter \
#     --bert_dim 768 \
#     --freeze_bert True \
#     --learning_rate 2e-5 \
#     >/dev/null 2>&1 &

# wait 
# # random reproduce get test result bert_gcn_ada

# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset restaurant \
#     --bert_dim 768 \
#     --pretrained_bert_name /home/wwz/absa/LCF-ABSA/data/restaurants_10mio_ep3 \
#     --freeze_bert False \
#     --learning_rate 2e-5 \
#     --step_eval \
#     --from_xml \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python train.py \
#     --model_name bert_gcn \
#     --dataset laptop \
#     --pretrained_bert_name /home/wwz/absa/LCF-ABSA/data/laptops_1mio_ep30 \
#     --bert_dim 768 \
#     --freeze_bert True \
#     --learning_rate 2e-5 \
#     --step_eval \
#     --from_xml \
#     >/dev/null 2>&1 &

# wait 
# random reproduce get test result bert_ada

# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#     --model_name bert_spc \
#     --dataset restaurant \
#     --bert_dim 768 \
#     --pretrained_bert_name /home/wwz/absa/LCF-ABSA/data/restaurants_10mio_ep3 \
#     --freeze_bert False \
#     --learning_rate 2e-5 \
#     >/dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python train.py \
#     --model_name bert_spc \
#     --dataset laptop \
#     --pretrained_bert_name /home/wwz/absa/LCF-ABSA/data/laptops_1mio_ep30 \
#     --bert_dim 768 \
#     --freeze_bert True \
#     --learning_rate 2e-5 \
#     >/dev/null 2>&1 &
