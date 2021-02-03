#!/bin/bash
#
#DATASET=$1
#FEATURES=$2
#MODEL=$3
#NCLASSES=$4
#SR=$5
#NSTEPS=$6
#EPOCHS=$7
#BATCHSIZE=$8
#LR=$9
#POSEMB=${10}
#LRJOINER=${11}
#SAVEFREQ=${12}
#NQUERIES=${13}
#ENCODER=${14}
#DECODER=${15}
#TASK=${16}
#NENCLAYERS=${17}
#NDECLAYERS=${18}
#HDIM=${19}
#NHEADS=${20}
#NPOSEMB=${21}
#DROPOUT=${22}
#LRDROP=${23}
#WDECAY=${24}
#CLIPNORM=${25}
#NF=${26}

#FOLDER="fulldata_nov12_graphattn_encdec"
#FOLDER="fulldata_nov12_graphattn_encdec_self"
#FOLDER="fulldata_nov12_graphattn_encdec_sorted"
#FOLDER="fulldata_nov12_graphattn_encdec_noiou"
#FOLDER="fulldata_nov12_graphattn_encdec_nosegment"

#FOLDER="fulldata_nov16_thumos"
#FOLDER="fulldata_nov17_thumos"

#FOLDER="fulldata_nov19_nosiou"
#FOLDER="fulldata_nov19_noseg"


for SR in 1 
do
  for nf in 128 #64
  do

#### jan13 params

    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} detr 20 ${SR} 1 15000 8 1e-5 learned 1e-5 30 900 parallel parallel action 6 6 512 8 512 0 10000 0 10 512 07789 "fulldata_jan13_thumos" 4000

    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} graphdetr 20 ${SR} 1 15000 2 1e-5 learned 1e-5 30 500 graph graph action 6 6 256 4 512 0 10000 0 0 512 04499 "fulldata_jan13_thumos" 4000


####nov17 params
#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} detr 20 ${SR} 1 15000 8 1e-5 learned 1e-5 30 900 parallel parallel action 6 6 512 8 512 0 10000 0 10 512 10619 "fulldata_nov17_thumos" 4000
#
#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} graphdetr 20 ${SR} 1 15000 4 1e-5 learned 1e-5 30 500 graph graph action 4 4 256 4 512 0 10000 0 0 512 02129 "fulldata_nov17_thumos" 4000
#
#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} graphdetr 20 ${SR} 1 15000 4 1e-5 learned 1e-5 30 500 graph parallel action 4 6 256 4 512 0 10000 0 0 512 04979 "fulldata_nov17_thumos" 4000
#
#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} graphdetr 20 ${SR} 1 15000 4 1e-5 learned 1e-5 30 500 parallel graph action 6 4 256 4 512 0 10000 0 0 512 06869 "fulldata_nov17_thumos" 4000
##
##
#####nov16 params
#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} detr 20 ${SR} 1 15000 4 1e-5 learned 1e-5 30 900 parallel parallel action 6 6 512 8 1024 0 10000 0 0 1024 12329 "fulldata_nov16_thumos" 4000
#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} detr 20 ${SR} 1 15000 4 1e-5 learned 1e-5 30 900 parallel parallel action 3 3 512 8 1024 0 10000 0 0 1024 11099 "fulldata_nov16_thumos" 4000
#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} detr 20 ${SR} 1 15000 4 1e-5 learned 1e-5 30 300 parallel parallel action 6 6 512 8 1024 0 10000 0 0 1024 11219 "fulldata_nov16_thumos" 4000
#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} detr 20 ${SR} 1 15000 4 1e-5 learned 1e-5 30 300 parallel parallel action 3 3 512 8 1024 0 10000 0 0 1024 11819 "fulldata_nov16_thumos" 4000
##    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} graphdetr 20 ${SR} 1 15000 1 1e-5 learned 1e-5 30 500 graph graph action 4 4 256 4 1024 0 10000 0 0 1024 2069 "fulldata_nov16_thumos" 4000
#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} graphdetr 20 ${SR} 1 15000 2 1e-5 learned 1e-5 30 500 graph graph action 2 2 256 4 1024 0 10000 0 0 1024 05489 "fulldata_nov16_thumos" 4000
#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} graphdetr 20 ${SR} 1 15000 2 1e-5 learned 1e-5 30 250 graph graph action 4 4 256 4 1024 0 10000 0 0 1024 06509 "fulldata_nov16_thumos" 4000
##    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} graphdetr 20 ${SR} 1 15000 1 1e-5 learned 1e-5 30 250 graph graph action 2 2 256 4 1024 0 10000 0 0 1024 2069 "fulldata_nov16_thumos" 4000



###nov12 params

#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} detr 20 ${SR} 1 8000 8 1e-5 learned 1e-5 30 900 parallel parallel action 6 6 512 8 512 0 1500 0 10 512 6269 "fulldata_nov12_graphattn_encdec_sorted" 2000

#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} graphdetr 20 ${SR} 1 8000 4 1e-5 learned 1e-5 30 500 graph graph action 4 4 256 4 512 0 1500 0 0 512 5399 "fulldata_nov12_graphattn_encdec_sorted" 2000

#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} graphdetr 20 ${SR} 1 8000 4 1e-5 learned 1e-5 30 500 graph parallel action 4 6 256 4 512 0 1500 0 0 512 6059 "fulldata_nov12_graphattn_encdec_sorted" 2000

#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} graphdetr 20 ${SR} 1 8000 4 1e-5 learned 1e-5 30 500 parallel graph action 6 4 256 4 512 0 1500 0 0 512 4439 "fulldata_nov12_graphattn_encdec_sorted" 2000




#    bash job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} detr 20 ${SR} 1 1500 8 1e-5 learned 1e-5 30 500 parallel parallel action 6 6 512 8 256 0 500 0 10 256 0149 ${FOLDER}
#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} detr 20 ${SR} 1 3300 8 1e-5 learned 1e-5 30 1000 parallel parallel action 6 6 512 8 512 0 1500 0 0 512 0389 ${FOLDER}
#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} detr 20 ${SR} 1 3300 8 1e-4 learned 1e-4 30 1000 parallel parallel action 6 6 512 8 512 0 1500 0 0 512 0389 ${FOLDER}
#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} detr 20 ${SR} 1 3300 8 1e-5 learned 1e-5 30 500 parallel parallel action 6 6 512 8 512 0 1500 0 10 512 0389 ${FOLDER}
#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i4d_feats_sr"${SR} detr 20 ${SR} 1 3300 8 1e-5 learned 1e-5 30 300 parallel parallel action 6 6 512 8 1024 0 1500 0 0 1024 0149 ${FOLDER}
#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} detr 20 ${SR} 1 3300 8 1e-5 learned 1e-5 30 300 parallel parallel action 6 6 512 8 256 0 1500 1e-5 0 256 0089 ${FOLDER}
#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} graphdetr 20 ${SR} 1 3300 8 1e-4 learned 1e-5 30 300 parallel graph action 6 6 256 8 256 0 1500 0 1 256 0149 ${FOLDER}
#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} graphdetr 20 ${SR} 1 3300 8 1e-4 learned 1e-5 30 300 graph parallel action 6 6 256 8 256 0 1500 0 1 256 0149 ${FOLDER}
#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} graphdetr 20 ${SR} 1 3300 2 1e-5 learned 1e-5 30 500 graph graph action 6 6 512 8 256 0 1500 0 0 256 0389 ${FOLDER}
#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} graphdetr 20 ${SR} 1 3300 4 1e-4 learned 1e-4 30 500 graph graph action 4 4 256 4 512 0 1500 0 0 512 0389 ${FOLDER}
#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} graphdetr 20 ${SR} 1 3300 2 1e-5 learned 1e-5 30 500 graph graph action 9 9 256 8 256 0 1500 0 0 256 0389 ${FOLDER}
#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} graphdetr 20 ${SR} 1 3300 2 1e-3 learned 1e-3 30 500 graph graph action 6 6 512 8 256 0 1500 0 0 256 0389 ${FOLDER}
#    sbatch job_scripts/run_tst_trainer_detection.sh thumos "thumos_i3d_feats_sr"${SR} graphdetr 20 ${SR} 1 3300 2 1e-4 learned 1e-4 30 500 graph graph action 6 6 512 8 256 0 1500 0 0 256 0389 ${FOLDER}
  done
done