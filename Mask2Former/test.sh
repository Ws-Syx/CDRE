rm -f test.log
for iter in {00..19..1}; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net_video.py \
    --config-file "./configs/youtubevis_2019/swin/video_maskformer2_swin_tiny_bs16_8ep.yaml" \
    --num-gpus 4 \
    --eval-only MODEL.WEIGHTS "./ckpt/train-50-5/model_0${iter}9999.pth" | tee -a test.log
done

# python train_net_video.py \
#   --config-file "./configs/youtubevis_2019/swin/video_maskformer2_swin_tiny_bs16_8ep.yaml" \
#   --num-gpus 2 \
#   --eval-only MODEL.WEIGHTS "./ckpt/train-26-1/model_0007999.pth"
