To run stage 4 trainning with checkpoint saving feature, and dual circle evaluation\
`python scripts/train.py --config configs/ham10000_voc_stage4.yaml data.root=/home/qiming/Documents/Datasets/VOCDatasets/HAM10000_VOC/VOCdevkit/VOC2012  output.dir=/experiments/runB-stage4-150epochs checkpoint.save_best=true checkpoint.save_latest=true checkpoint.keep_per_epoch=false eval.dual_circle.enable=true eval.dual_circle.image_dir=~/Documents/AllBruiseTrainingData/Unlabelled/uni_dataset/no_circle_data eval.dual_circle.circle_dir=~/Documents/AllBruiseTrainingData/Unlabelled/uni_dataset/circle_mask eval.dual_circle.output_dir=/experiments/runB-stage4-150epochs/eval/dual_circle_results.csv`

python scripts/eval_dual_circle.py \
    --config configs/ham10000_voc_stage4_sam2_dinov2.yaml \
    --checkpoint experiments/runC-stage4-150epochs/checkpoints/best.pt \
    --image_dir ~/Documents/AllBruiseTrainingData/Unlabelled/uni_dataset/CleanData \
    --circle_dir ~/Documents/AllBruiseTrainingData/Unlabelled/uni_dataset/clean_data_mask \             
    --ignore_rect_dir ~/Documents/AllBruiseTrainingData/Unlabelled/uni_dataset/blue_rect_masks \
    --output_csv experiments/runC-stage4-150epochs/eval_test_full_image.csv \
    --text "lesions on light skin" \
    model.dino.name=vit_base_patch16_224.dino

python scripts/train.py --config configs/ham10000_voc_stage4_swin.yaml \
    data.root=/home/qiming/Documents/Datasets/VOCDatasets/HAM10000_VOC/VOCdevkit/VOC2012 \
    output.dir=experiments/runB-stage4-150epochs-swin \
    checkpoint.save_best=true checkpoint.save_latest=true checkpoint.keep_per_epoch=false \
    eval.dual_circle.enable=true \
    eval.dual_circle.image_dir=~/Documents/AllBruiseTrainingData/Unlabelled/uni_dataset/
  no_circle_data \
    eval.dual_circle.circle_dir=~/Documents/AllBruiseTrainingData/Unlabelled/uni_dataset/
  circle_mask \
    eval.dual_circle.output_dir=experiments/runB-stage4-150epochs-swin/eval/dual_circle_results.csv