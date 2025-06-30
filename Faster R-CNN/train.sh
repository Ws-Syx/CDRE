python train_net.py \
    --num-gpus 2 --resume \
    --config-file "./configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml" | tee train.log