CUDA_VISIBLE_DEVICES=0,1 python train_net_video.py \
  --config-file ./configs/youtubevis_2019/swin/video_maskformer2_swin_tiny_bs16_8ep.yaml --num-gpus 2 | tee train.log