# DEMO PROGRAM SKRIPSI

This repository contains the demo program for the Skripsi project by Amirah Dzatul Himmah (2002871).

## Project Overview

This project demonstrates various experiments using different models and augmentation techniques for image classification. The models used include VanillaCNN, InceptionV3, and ResNet50 with different augmentation strategies.

## Models

### VanillaCNN
A simple Convolutional Neural Network (CNN) with multiple convolutional and pooling layers followed by dense layers.

### InceptionV3
A deep learning model from the Inception family, known for its efficiency and accuracy in image classification tasks.

### ResNet50
A deep residual network with 50 layers, known for its ability to train very deep networks without the vanishing gradient problem.

## Augmentation Techniques

### No Augmentation
No additional augmentation is applied to the images.

### Simple Augmentation
Includes basic augmentations such as random flipping, rotation, and brightness adjustments.

### RandAugment
A more advanced augmentation technique that applies a random combination of augmentations to each image.

## Tuning Parameters

### Trainable
Indicates whether the base model layers are trainable or frozen during training.

## Setup Instructions

### Prerequisites

- Python 3.x
- Jupyter Notebook or Google Colab

<!-- ### Installation

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd repo
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ``` -->

### Running the Notebook

1. Open the `Demo.ipynb` notebook in Jupyter Notebook or Google Colab.
2. Follow the instructions in the notebook to run the experiments.

### Dataset

The dataset should be uploaded to Google Drive and mounted in the notebook. The dataset should be in a zip file named `dataset_split.zip` and should be placed in the path `/content/drive/MyDrive/skripsi/`. (or you can change the initial path with your desired path directly on the notebook)

## Experiments

The experiments are configured in the `EXPERIMENT_CONFIGS` dictionary in the notebook. You can select an experiment by changing the `selected_experiment` variable.

### Experiment Outputs

For each experiment, the following files will be saved in the directory specified by `DRIVE_DIR/EXPERIMENT_NAME/`:
- `training_history.csv`: Contains the training history including accuracy and loss for each epoch.
- `best_model.keras`: The best model saved during training based on validation accuracy.
- `checkpoint.keras`: The model checkpoint saved at the end of each epoch.
- `plot.png`: A plot of the training and validation accuracy and loss over epochs.

## Training and Evaluation

The notebook includes code for training the models, resuming training from checkpoints, and evaluating the models on the test dataset. The training history is saved to a CSV file, and the best model is saved to a file.

## Results

The results of the experiments, including accuracy, precision, recall, F1 score, and confusion matrix, are printed in the notebook. The training and validation accuracy and loss are plotted and saved as an image.

<!-- ## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. -->

## Contact

For any questions or inquiries, please contact Amirah Dzatul Himmah at [amirahdzh@gmail.com].
