# auto-image-encoder

This project aims to train and create a machine learning model to
replace a manual image editing process for enhncing underwater images.

## Dataset

There is a good manually edited input dataset that has been manually edited by the Seattle aquarium staff. This dataset includes both the unprocessed GPR files and the corresponding processed JPEG outputs.

## General Steps for Output Processing

The goal is to have a program that can:

1. Read a directory or single input .gpr file.
2. Convert that .gpr file to a standard raw file of size (4606 x 4030) cropped to image center using the gopro tools (https://github.com/gopro/gpr)
3. Input the cropped raw file into a trained machine learning auto encoder that will enhance the image to match the human steps
4. Save the output image in a specified directory.


## Development Steps 

1. Select the appropriate machine learning framework (e.g., TensorFlow, PyTorch) for building the autoencoder.
2. Design the architecture of the autoencoder, including the encoder and decoder components.
3. Preprocess the dataset by normalizing the images and splitting them into training and validation sets.
4. Train the autoencoder using the training set and evaluate its performance on the validation set.
5. Fine-tune the model by adjusting hyperparameters and retraining as necessary.
6. Save the trained model for later use in the output processing pipeline.