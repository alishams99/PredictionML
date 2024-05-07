# PredictionML
This project aims to train machine learning models to predict the "nature of injury" based on other columns within an Excel dataset. 


## Overview

This project aims to train machine learning models to predict the "nature of injury" based on other columns within an Excel dataset. The models implemented include Gaussian Naive Bayes, Random Forest, Standard Scaler, and a trained Logistic Regression model.

### Disclaimer
The dataset provided for demonstration purposes contains only 1000 rows of the main data. Consequently, the accuracy reported here may not reflect the performance of the models when trained on larger datasets.

## Installation

To run this project locally, follow these steps:

1. Clone this repository to your local machine using `git clone https://github.com/your-username/your-repository.git`.
2. Navigate to the project directory.
3. Install the required dependencies by running `pip install -r requirements.txt`.

## Usage

After installation, you can train and evaluate the machine learning models using the provided scripts:

1. `gaussian_nb` - Train the Gaussian Naive Bayes model.
2. `randomforest` - Train the Random Forest model.
3. `standardscaler` - Train the Standard Scaler.
4. `train_logistic_regression_model` - Train the Logistic Regression model.

Each script accepts parameters or configuration files for customization. Refer to the script documentation for more details.

## Results

Below are the performance metrics achieved by each model on the provided dataset:

- Gaussian Naive Bayes: [42%]
- Random Forest: [91%]
- Standard Scaler: [70%]
- Logistic Regression: [45%]
