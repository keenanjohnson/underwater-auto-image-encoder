# Underwater Image Enhancement with ML

This is an automated machine learning pipeline that replaces manual image editing for underwater GoPro images captured during ROV surveys. 
Converts RAW GPR files to enhanced images matching manual Adobe Lightroom editing quality.

### For People That Just Want to Process Images (GUI Application)

**No programming required** - Desktop application available for Windows, macOS, and Linux.

👉 **[See GUI Documentation](gui/README.md)**

**Quick steps:**
1. Download the application for your platform
2. Download a trained model (.pth file)
3. Select a Folder of Images to Enhance and hit go!

### For People Wanting to Train Their Own Models or Use Command Line Inference

Train custom models or integrate into automated workflows.

👉 **[See Training Documentation](training/README.md)**

There are many options for customizing model architecture, training parameters, and datasets.

These are defined in the setup_and_train_config.yaml file.

The most important parameters are:

- **model** - Model architecture to use:
  - `unet` - Standard U-Net autoencoder (~31M params) - faster training, good baseline
  - `ushape_transformer` - U-shape Transformer with CMSFFT+SGFMT (~31M params) - better quality, slower training
- **repo_id** - Which hugging face dataset to download and train with
- **image_size** - What size of images to train on. Ideally this should be as large as your GPU memory allows.
- **batch_size** - How many images to process at once. Again, larger is better, but limited by GPU memory.
- **num_epochs** - How many passes through the dataset to train for.

**Quick start:**
```bash
python3.10 -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt

# Train a model (downloads dataset automatically and trains
python training/setup_and_train.py

# Run inference on images
python inference/inference.py input.jpg --checkpoint output/best_model.pth
```

### Run Inference (Command Line Image Processing)

See the scripts in the `inference/` folder for more details on args

```bash
python3.10 -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt

python inference/inference.py input.jpg --checkpoint checkpoints/best_model.pth
python inference/inference.py /path/to/images --checkpoint checkpoints/best_model.pth --output enhanced/
python inference/inference.py input.jpg --checkpoint checkpoints/best_model.pth --compare
```

### Preprocess GPR Files
```bash
python3.10 -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt

python preprocessing/preprocess_images.py /path/to/gpr/files --output-dir processed
```

## Pre-trained Models

Example trained models are available for download here:
https://huggingface.co/Seattle-Aquarium

## Datasets

The Seattle Aquarium CCR Underwater Image Enhancement Dataset is available at:
https://huggingface.co/datasets/Seattle-Aquarium/Seattle_Aquarium_benthic_imagery

## References

- [Project Discussion & Sample Data](https://github.com/Seattle-Aquarium/CCR_image_processing)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [U-shape Transformer for Underwater Image Enhancement](https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement) - Lintao Peng et al.

## Contributing

This project is developed to support the Seattle Aquarium's ROV survey enhancement pipeline. For questions or contributions, refer to the main [CCR development repository](https://github.com/Seattle-Aquarium/CCR_development).

You can also submit PRs or issues here and we will route them accordingly.

---

**Quick Links:**
- **GUI Users**: Start with [gui/README.md](gui/README.md)
- **Training Models**: Start with [training/README.md](training/README.md)
