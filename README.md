# RF-DETR Training Pipeline

A modular fine-tuning pipeline for RF-DETR (Real-time DEtection TRansformer) models.

## Introduction

The RF-DETR Training Pipeline simplifies the process of fine-tuning RF-DETR models for object detection tasks
It provides a user-friendly command-line interface (CLI) that handles everything from dataset preparation to model training and inference, enabling users to leverage state-of-the-art detection capabilities without deep expertise in machine learning pipelines.

For more details on RF-DETR, refer to the [official repository](https://github.com/LyuJingTechnology/RF-DETR) and the [training guide](https://rfdetr.roboflow.com/learn/train/).

## Motivation

Fine-tuning large vision models like RF-DETR can be daunting due to the need for data preprocessing, configuration management, and training orchestration.
This pipeline bridges that gap by offering a modular, CLI-driven approach that democratizes access to advanced object detection. Whether you're a researcher prototyping new ideas or a developer integrating detection into applications, the pipeline reduces boilerplate and accelerates iteration.

## Project Structure

```
rf-detr-training-pipeline/
├── config/                        # Training configuration files
├── data/                          # Downloaded and prepared datasets
├── src/
│   └── rf_detr_finetuning/       # Main package
├── examples/                      # Usage tutorials and demos
├── tests/                         # Test suite
└── README.md                      # This file
```

## Installation

To install the package with CLI and data extras:

```bash
pip install -e .[cli,data]
```

This includes dependencies for the command-line interface (jsonargparse) and data downloading (kaggle).

## Quick Start

### Dataset Download

Use the local CLI to download datasets. For example, to download the CCTV Weapon Dataset from Kaggle:

```bash
python -m rf_detr_finetuning download kaggle-dataset --name dataset/name --dest data
```

If authentication fails, you'll be prompted for your Kaggle username and API key.

### Dataset Conversion

To convert a YOLO dataset to COCO format with train/valid/test splits:

```bash
python -m rf_detr_finetuning convert yolo-to-coco --input_dir path/to/yolo/dataset --output_dir path/to/output --split_ratios 0.8 0.1 0.1
```

### Training

To train the model on the prepared dataset:

```bash
python -m rf_detr_finetuning train --config_file config/default.yaml --dataset path/to/output --model_size small
```

### Prediction

To run inference on an image using the trained model:

```bash
python -m rf_detr_finetuning predict --model_path path/to/checkpoint.pth --image_path path/to/image.jpg --confidence 0.5
```

## Real Use Case

See [examples/README.md](examples/README.md) for a complete walkthrough of fine-tuning RF-DETR on custom Datasets.

## Configuration

Configuration files are stored in the `config/` directory. See `config/README.md` for details.
