set -x

ossutil64 cp -f -r /data1/yyb/PlatonicGen/pretrained/ oss://antsys-vilab/yyb/PlatonicGen/pretrained/ 
ossutil64 cp -f -r /data0/hf_weights/stabilityai/stable-diffusion-2-1/ oss://antsys-vilab/yyb/huggingface/stabilityai/stable-diffusion-2-1/
ossutil64 cp -f -r oss://antsys-vilab/yyb/huggingface/stabilityai/stable-diffusion-2-1/ /input/yyb/huggingface/stabilityai/stable-diffusion-2-1/ 
ossutil64 cp -f -r /data1/yyb/PlatonicGen/pretrained/clip_image/ oss://antsys-vilab/yyb/PlatonicGen/pretrained/clip_image/ 
ossutil64 cp -f -r oss://antsys-vilab/yyb/PlatonicGen/pretrained/clip_image/  /input/yyb/PlatonicGen/pretrained/clip_image/
ossutil64 cp -f -r /data1/yyb/PlatonicGen/third_party/ oss://antsys-vilab/yyb/PlatonicGen/third_party/ 
ossutil64 cp -f -r oss://antsys-vilab/yyb/PlatonicGen/third_party/ /input/yyb/PlatonicGen/third_party/
ossutil64 cp -f   oss://antsys-vilab/yyb/PlatonicGen/requirements.txt /input/yyb/PlatonicGen/requirements.txt

ossutil64 cp -f -r  /input/yyb/PlatonicGen/outputs/scene_traj/ oss://antsys-vilab/yyb/PlatonicGen//outputs/scene_traj/
ossutil64 cp -f -r  oss://antsys-vilab/yyb/PlatonicGen//outputs/scene_traj/  /data1/yyb/PlatonicGen/scene_traj/