CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py \
    --num-gpus 4 --config-file "./configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml" \
    --eval-only MODEL.WEIGHTS "./ckpt/train-2-1/model_final.pth" | tee test.log