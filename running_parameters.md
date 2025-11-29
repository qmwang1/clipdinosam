To run stage 4 training, evaluate dual-circle F1 every epoch, and save the best checkpoint based on that metric:
`python scripts/train.py --config configs/ham10000_voc_stage4.yaml data.root=/home/qiming/Documents/Datasets/VOCDatasets/HAM10000_VOC/VOCdevkit/VOC2012 output.dir=/experiments/runB-stage4-150epochs checkpoint.save_best=true checkpoint.save_latest=true checkpoint.keep_per_epoch=false checkpoint.best_metric=dual_circle_f1 eval.dual_circle.enable=true eval.dual_circle.image_dir=~/Documents/AllBruiseTrainingData/Unlabelled/uni_dataset/no_circle_data eval.dual_circle.circle_dir=~/Documents/AllBruiseTrainingData/Unlabelled/uni_dataset/circle_mask eval.dual_circle.output_dir=/experiments/runB-stage4-150epochs/eval`

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
    eval.dual_circle.image_dir=~/Documents/AllBruiseTrainingData/Unlabelled/uni_dataset/no_circle_data \
    eval.dual_circle.circle_dir=~/Documents/AllBruiseTrainingData/Unlabelled/uni_dataset/circle_mask \
    eval.dual_circle.output_dir=experiments/runB-stage4-150epochs-swin/eval

---

Train on one HAM10000 class, evaluate on others

1) Create per-class splits (train/val/test) for the class you want to train on:
```
python - <<'PY'
import os, xml.etree.ElementTree as ET, random
root = "/home/qiming/Documents/Datasets/HAM10000/HAM10000_VOC"
cls = "Melanocytic Nevi"   # change to the training class name from XML
xml_dir = os.path.join(root, "Annotations")
ids = []
for fname in os.listdir(xml_dir):
    if not fname.endswith(".xml"): continue
    tree = ET.parse(os.path.join(xml_dir, fname))
    name = tree.find('.//name').text.strip()
    if name == cls:
        ids.append(os.path.splitext(fname)[0])
ids.sort()
random.seed(0)
random.shuffle(ids)
n = len(ids)
train, val, test = ids[:int(0.7*n)], ids[int(0.7*n):int(0.85*n)], ids[int(0.85*n):]
seg_dir = os.path.join(root, "ImageSets", "Segmentation")
os.makedirs(seg_dir, exist_ok=True)
for split_name, split_ids in [("train_onecls", train), ("val_onecls", val), ("test_onecls", test)]:
    with open(os.path.join(seg_dir, f"{split_name}.txt"), "w") as f:
        f.write("\n".join(split_ids))
print(f"Wrote splits for {cls}: train {len(train)}, val {len(val)}, test {len(test)}")
PY
```
Optionally make eval splits for each other class (change `cls` and filenames like `val_Melanoma.txt`).

2) Train (Stage 1 example):
```
python scripts/train.py \
  --config configs/ham10000_voc_stage1.yaml \
  data.root=/home/qiming/Documents/Datasets/HAM10000/HAM10000_VOC \
  data.split=/home/qiming/Documents/Datasets/HAM10000/HAM10000_VOC/ImageSets/Segmentation/train_onecls.txt \
  data.text="melanocytic nevi lesion" \
  output.dir=experiments/ham10000_onecls_stage1 \
  checkpoint.save_best=true \
  checkpoint.best_metric=train_loss \
  epochs=50
```

3) Evaluate segmentation on another class (Dice/IoU) with the new evaluator:
```
python scripts/eval_voc_seg.py \
  --config configs/ham10000_voc_stage1.yaml \
  --checkpoint experiments/ham10000_onecls_stage1/checkpoints/best.pt \
  --split /home/qiming/Documents/Datasets/HAM10000/HAM10000_VOC/ImageSets/Segmentation/val_Melanoma.txt \
  --data_root /home/qiming/Documents/Datasets/HAM10000/HAM10000_VOC \
  --text "melanoma lesion" \
  --output_csv experiments/ham10000_onecls_stage1/eval_val_melanoma.csv
```
Change `--split` and `--text` for each target class you want to test.

---

Optional t-SNE plot of token embeddings each epoch (colors = mask coverage)

Add this block to your config to enable:
```
visualize:
  tsne:
    enable: true
    max_samples: 256        # limit points
    max_batches: 4          # how many train batches to sample
    perplexity: 30
    learning_rate: 200
    n_iter: 750
    output: tsne_epoch_{epoch}.png  # saved under output.dir or checkpoint dir
```
Notes:
- Uses pooled CLIP token embeddings (`skip_sam=True`) for speed.
- Requires scikit-learn (added to requirements) and matplotlib.
- Writes to `output.dir` (or checkpoint dir) on rank 0 only.
