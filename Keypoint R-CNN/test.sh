rm -f test.log
for iter in {04..59..5}; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py \
        --num-gpus 4 --config-file "./configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml" \
        --eval-only MODEL.WEIGHTS "./ckpt/train-3-3/model_00${iter}999.pth" | tee -a test.log
done