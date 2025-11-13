# Real Use Case

## Weapon Detection in CCTV Footage

1. Download the dataset as above.
2. Prepare your data by converting to COCO format using the CLI:
   ```bash
   python -m rf_detr_finetuning download kaggle-dataset --name simuletic/cctv-weapon-dataset --dest data
   python -m rf_detr_finetuning convert yolo-to-coco \
      --input_dir data/simuletic/cctv-weapon-dataset/Dataset \
      --output_dir data/cctv-weapon-dataset_coco \
      --class_names '{"0": "person", "1": "weapon"}'
   ```
3. Run training with a config tailored for weapon detection.
   ```bash
   python -m rf_detr_finetuning train --dataset data/cctv-weapon-dataset_coco --config_file config/weapon_detection.yaml
   ```
4. Evaluate the model on test footage.
   ```bash
   python -m rf_detr_finetuning predict --model_path output/checkpoint_best_total.pth --image_path data/simuletic/cctv-weapon-dataset/Dataset/images/Scene1_2.png
   ```

See the main README for module details and configuration options.
