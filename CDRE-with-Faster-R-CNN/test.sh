rm -f test.log

for i in {05..29..2}; do
    python train_net.py \
        --num-gpus 4 --config-file "./configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml" \
        --eval-only MODEL.WEIGHTS "./ckpt/train-3-5/model_00${i}999.pth" | tee -a test.log
done