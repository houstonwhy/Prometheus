set -x

./scripts/scripts/setup4diffgau.sh && python train.py global_env=aistudio_notebook experiment=gsdecoder_exp dataset=gsdecoder_dataset_full algorithm=gsdecoder GPUS=[0]