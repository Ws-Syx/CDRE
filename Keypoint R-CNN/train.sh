CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py \
    --num-gpus 4 --resume \
    --config-file "./configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml" | tee train.log