#!/bin/bash
DEFAULT_JOB_ID=offline-run-20240913_021321-prometheus_k8s_gsdecoder_exp_gsdecoder_dataset_gsdecoder_gsdecoder_exp_exp02_ampbugfixed_20240913_0206

if [ -z "$1" ]; then
  JOB_ID=$DEFAULT_JOB_ID
else
  JOB_ID=$1
fi

set -x

ossutil64 cp -r -f /input/yyb/PlatonicGen/outputs/wandb/${JOB_ID}/ oss://antsys-vilab/yyb/PlatonicGen/workdir/wandb_cluster/${JOB_ID}/