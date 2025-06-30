rm -f test.log

# for i in {05..09}; do
#     python train_net.py \
#         --num-gpus 2 --config-file "./configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml" \
#         --eval-only MODEL.WEIGHTS "./ckpt/train-6-1/model_0${i}9999.pth" | tee -a test.log
# done

python train_net.py \
    --num-gpus 2 --config-file "./configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml" \
    --eval-only MODEL.WEIGHTS "./ckpt/train-7-1/model_final.pth" | tee -a test.log