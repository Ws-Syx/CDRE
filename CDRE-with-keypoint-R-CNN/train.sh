CUDA_VISIBLE_DEVICES=2,3 python train_net.py \
    --num-gpus 2 \
    --config-file "./configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml" | tee train.log