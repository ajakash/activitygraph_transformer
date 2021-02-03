#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gres=gpu:v100l:4   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24    # There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --mem=60G               # Request the full memory of the node
#SBATCH --time=0-23:59:00
#SBATCH --account=rrg-mori
#SBATCH --output=cedar_logs_tst/log_%A.out
hostname
nvidia-smi

DATASET=$1
FEATURES=$2
MODEL=$3
NCLASSES=$4
SR=$5
NSTEPS=$6
EPOCHS=$7
BATCHSIZE=$8
LR=$9
POSEMB=${10}
LRJOINER=${11}
SAVEFREQ=${12}
NQUERIES=${13}
ENCODER=${14}
DECODER=${15}
TASK=${16}
NENCLAYERS=${17}
NDECLAYERS=${18}
HDIM=${19}
NHEADS=${20}
NPOSEMB=${21}
DROPOUT=${22}
LRDROP=${23}
WDECAY=${24}
CLIPNORM=${25}
NF=${26}
RESUME_EPOCH=${27}
FOLDER_SUFFIX=${28}
EVALFREQ=${29}

OUT="./experiments/"${FOLDER_SUFFIX}"/"${DATASET}"/"${TASK}"/checkpoints_model_"${MODEL}"_numqueries"${NQUERIES}"_lr"$LR"_lrdrop"${LRDROP}"_dropout"${DROPOUT}"_clipmaxnorm"${CLIPNORM}"_weightdecay"${WDECAY}"_posemb"$POSEMB"_lrjoiner"${LRJOINER}"_encoder"${ENCODER}"_decoder"${DECODER}"_nheads"${NHEADS}"_nenclayers"${NENCLAYERS}"_ndeclayers"${NDECLAYERS}"_hdim"${HDIM}"_sr"${SR}"_batchsize"${BATCHSIZE}"_stepsize"${STEPSIZE}"_nposembdict"${NPOSEMB}"_numinputs"${NF}

LOGDIR="./experiments/"${FOLDER_SUFFIX}
LOG=${LOGDIR}"/"${DATASET}"/"${TASK}"/log_model_"${MODEL}"_numqueries"${NQUERIES}"_lr"$LR"_lrdrop"${LRDROP}"_dropout"${DROPOUT}"_clipmaxnorm"${CLIPNORM}"_weightdecay"${WDECAY}"_posemb"$POSEMB"_lrjoiner"${LRJOINER}"_encoder"${ENCODER}"_decoder"${DECODER}"_nheads"${NHEADS}"_nenclayers"${NENCLAYERS}"_ndeclayers"${NDECLAYERS}"_hdim"${HDIM}"_sr"${SR}"_batchsize"${BATCHSIZE}"_stepsize"${STEPSIZE}"_nposembdict"${NPOSEMB}"_numinputs"${NF}".log"

OUT="./checkpoints_"${FOLDER_SUFFIX}"/"${DATASET}"/"${TASK}"/checkpoints_model_"${MODEL}"_numqueries"${NQUERIES}"_lr"$LR"_lrdrop"${LRDROP}"_dropout"${DROPOUT}"_clipmaxnorm"${CLIPNORM}"_weightdecay"${WDECAY}"_posemb"$POSEMB"_lrjoiner"${LRJOINER}"_encoder"${ENCODER}"_decoder"${DECODER}"_nheads"${NHEADS}"_nenclayers"${NENCLAYERS}"_ndeclayers"${NDECLAYERS}"_hdim"${HDIM}"_sr"${SR}"_batchsize"${BATCHSIZE}"_stepsize"${STEPSIZE}"_nposembdict"${NPOSEMB}"_numinputs"${NF}

#LOGDIR="./logs_"${FOLDER_SUFFIX}
#LOG=${LOGDIR}"/"${DATASET}"/"${TASK}"/log_model_"${MODEL}"_numqueries"${NQUERIES}"_lr"$LR"_lrdrop"${LRDROP}"_dropout"${DROPOUT}"_clipmaxnorm"${CLIPNORM}"_weightdecay"${WDECAY}"_posemb"$POSEMB"_lrjoiner"${LRJOINER}"_encoder"${ENCODER}"_decoder"${DECODER}"_nheads"${NHEADS}"_nenclayers"${NENCLAYERS}"_ndeclayers"${NDECLAYERS}"_hdim"${HDIM}"_sr"${SR}"_batchsize"${BATCHSIZE}"_stepsize"${STEPSIZE}"_nposembdict"${NPOSEMB}"_numinputs"${NF}".log"

mkdir -p "./experiments/"${FOLDER_SUFFIX}"/"${DATASET}"/"${TASK}
mkdir -p ${LOGDIR}"/"${DATASET}"/"${TASK}
exec &> >(tee -a "${LOG}")
echo Logging output to "${LOG}"

DATA=$PWD"/data/"${DATASET}"/"${DATASET}"_sr"${SR}".tar.gz"

source ~/.bashrc
conda activate tst_env
export LD_LIBRARY_PATH="/home/mnawhal/miniconda3/envs/tst_env/lib":$LD_LIBRARY_PATH


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 src/main_detection.py --dataset ${DATASET} --data_root "/tmp/"${DATASET}"/"${DATASET}"_sr"${SR} --model ${MODEL} --task ${TASK} --features ${DATASET}"_i3d_feats_sr"${SR}  --batch_size ${BATCHSIZE} --enc_layers ${NENCLAYERS} --dec_layers ${NDECLAYERS} --num_queries ${NQUERIES} --nheads ${NHEADS} --dropout ${DROPOUT} --weight_decay ${WDECAY} --clip_max_norm ${CLIPNORM} --num_inputs ${NF} --num_pos_embed_dict ${NPOSEMB} --hidden_dim ${HDIM} --cuda --num_workers 0 --num_classes ${NCLASSES} --step_size ${NSTEPS} --sample_rate ${SR} --cluster --data_zip ${DATA} --cluster_tmp "/tmp/"${DATASET} --lr ${LR} --lr_drop ${LRDROP} --epochs ${EPOCHS} --output_dir ${OUT} --position_embedding ${POSEMB} --lr_joiner ${LRJOINER} --save_checkpoint_every ${SAVEFREQ} --evaluate_every ${EVALFREQ} --encoder ${ENCODER} --decoder ${DECODER} --dim_feedforward $(( 4 * ${HDIM} )) --set_cost_segment 5 --set_cost_siou 3 --segment_loss_coef 5 --siou_loss_coef 3 --sorted --decoder_attn ctx --resume ${OUT}"/checkpoint"${RESUME_EPOCH}".pth" #--resampled #

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 src/main_detection.py --dataset ${DATASET} --data_root "/tmp/"${DATASET}"/"${DATASET}"_sr"${SR} --model ${MODEL} --task ${TASK} --features ${DATASET}"_i3d_feats_sr"${SR}  --batch_size ${BATCHSIZE} --enc_layers ${NENCLAYERS} --dec_layers ${NDECLAYERS} --num_queries ${NQUERIES} --nheads ${NHEADS} --dropout ${DROPOUT} --weight_decay ${WDECAY} --clip_max_norm ${CLIPNORM} --num_inputs ${NF} --num_pos_embed_dict ${NPOSEMB} --hidden_dim ${HDIM} --cuda --num_workers 0 --num_classes ${NCLASSES} --step_size ${NSTEPS} --sample_rate ${SR} --cluster --data_zip ${DATA} --cluster_tmp "/tmp/"${DATASET} --lr ${LR} --lr_drop ${LRDROP} --epochs ${EPOCHS} --output_dir ${OUT} --position_embedding ${POSEMB} --lr_joiner ${LRJOINER} --save_checkpoint_every ${SAVEFREQ} --evaluate_every ${EVALFREQ} --encoder ${ENCODER} --decoder ${DECODER} --dim_feedforward $(( 4 * ${HDIM} )) --set_cost_segment 5 --set_cost_siou 3 --segment_loss_coef 5 --siou_loss_coef 3 --sorted --decoder_attn ctx  #--resume ${OUT}"/checkpoint"${RESUME_EPOCH}".pth" #--resampled #

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 src/main_detection.py --dataset ${DATASET} --data_root "/tmp/"${DATASET}"/"${DATASET}"_sr"${SR} --model ${MODEL} --task ${TASK} --features ${DATASET}"_i3d_feats_sr"${SR}  --batch_size ${BATCHSIZE} --enc_layers ${NENCLAYERS} --dec_layers ${NDECLAYERS} --num_queries ${NQUERIES} --nheads ${NHEADS} --dropout ${DROPOUT} --weight_decay ${WDECAY} --clip_max_norm ${CLIPNORM} --num_inputs ${NF} --num_pos_embed_dict ${NPOSEMB} --hidden_dim ${HDIM} --cuda --num_workers 0 --num_classes ${NCLASSES} --step_size ${NSTEPS} --sample_rate ${SR} --cluster --data_zip ${DATA} --cluster_tmp "/tmp/"${DATASET} --lr ${LR} --lr_drop ${LRDROP} --epochs ${EPOCHS} --output_dir ${OUT} --position_embedding ${POSEMB} --lr_joiner ${LRJOINER} --save_checkpoint_every ${SAVEFREQ} --evaluate_every ${EVALFREQ} --encoder ${ENCODER} --decoder ${DECODER} --dim_feedforward $(( 4 * ${HDIM} )) --set_cost_segment 5 --set_cost_siou 3 --segment_loss_coef 5 --siou_loss_coef 3 --resampled --resume ${OUT}"/checkpoint"${RESUME_EPOCH}".pth" #--resampled #

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 src/main_detection.py --dataset ${DATASET} --data_root "/tmp/"${DATASET}"/"${DATASET}"_sr"${SR} --model ${MODEL} --task ${TASK} --features ${DATASET}"_i3d_feats_sr"${SR}  --batch_size ${BATCHSIZE} --enc_layers ${NENCLAYERS} --dec_layers ${NDECLAYERS} --num_queries ${NQUERIES} --nheads ${NHEADS} --dropout ${DROPOUT} --weight_decay ${WDECAY} --clip_max_norm ${CLIPNORM} --num_inputs ${NF} --num_pos_embed_dict ${NPOSEMB} --hidden_dim ${HDIM} --cuda --num_workers 0 --num_classes ${NCLASSES} --step_size ${NSTEPS} --sample_rate ${SR} --cluster --data_zip ${DATA} --cluster_tmp "/tmp/"${DATASET} --lr ${LR} --lr_drop ${LRDROP} --epochs ${EPOCHS} --output_dir ${OUT} --position_embedding ${POSEMB} --lr_joiner ${LRJOINER} --save_checkpoint_every ${SAVEFREQ} --evaluate_every ${EVALFREQ} --encoder ${ENCODER} --decoder ${DECODER} --dim_feedforward $(( 4 * ${HDIM} )) --set_cost_segment 5 --set_cost_siou 0 --segment_loss_coef 5 --siou_loss_coef 0 --sorted #--resume ${OUT}"/checkpoint"${RESUME_EPOCH}".pth" #--resampled #

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 src/main_detection.py --dataset ${DATASET} --data_root "/tmp/"${DATASET}"/"${DATASET}"_sr"${SR} --model ${MODEL} --task ${TASK} --features ${DATASET}"_i3d_feats_sr"${SR}  --batch_size ${BATCHSIZE} --enc_layers ${NENCLAYERS} --dec_layers ${NDECLAYERS} --num_queries ${NQUERIES} --nheads ${NHEADS} --dropout ${DROPOUT} --weight_decay ${WDECAY} --clip_max_norm ${CLIPNORM} --num_inputs ${NF} --num_pos_embed_dict ${NPOSEMB} --hidden_dim ${HDIM} --cuda --num_workers 0 --num_classes ${NCLASSES} --step_size ${NSTEPS} --sample_rate ${SR} --cluster --data_zip ${DATA} --cluster_tmp "/tmp/"${DATASET} --lr ${LR} --lr_drop ${LRDROP} --epochs ${EPOCHS} --output_dir ${OUT} --position_embedding ${POSEMB} --lr_joiner ${LRJOINER} --save_checkpoint_every ${SAVEFREQ} --evaluate_every ${EVALFREQ} --encoder ${ENCODER} --decoder ${DECODER} --dim_feedforward $(( 4 * ${HDIM} )) --set_cost_segment 0 --set_cost_siou 3 --segment_loss_coef 0 --siou_loss_coef 3 --resampled #--resume ${OUT}"/checkpoint"${RESUME_EPOCH}".pth" #--resampled #

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 src/main_detection.py --dataset ${DATASET} --data_root "/tmp/"${DATASET}"/"${DATASET}"_sr"${SR} --model ${MODEL} --task ${TASK} --features ${DATASET}"_i3d_feats_sr"${SR}  --batch_size ${BATCHSIZE} --enc_layers ${NENCLAYERS} --dec_layers ${NDECLAYERS} --num_queries ${NQUERIES} --nheads ${NHEADS} --dropout ${DROPOUT} --weight_decay ${WDECAY} --clip_max_norm ${CLIPNORM} --num_inputs ${NF} --num_pos_embed_dict ${NPOSEMB} --hidden_dim ${HDIM} --cuda --num_workers 0 --num_classes ${NCLASSES} --step_size ${NSTEPS} --sample_rate ${SR} --cluster --data_zip ${DATA} --cluster_tmp "/tmp/"${DATASET} --lr ${LR} --lr_drop ${LRDROP} --epochs ${EPOCHS} --output_dir ${OUT} --position_embedding ${POSEMB} --lr_joiner ${LRJOINER} --save_checkpoint_every ${SAVEFREQ} --evaluate_every ${EVALFREQ} --encoder ${ENCODER} --decoder ${DECODER} --dim_feedforward $(( 4 * ${HDIM} )) --set_cost_segment 5 --set_cost_siou 3 --segment_loss_coef 5 --siou_loss_coef 3 --resampled --resume ${OUT}"/checkpoint"${RESUME_EPOCH}".pth" #--resampled #

rm -rf "/tmp/"${DATASET}   