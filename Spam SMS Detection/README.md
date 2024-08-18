# Spam Classification Project

## Overview

This project focuses on classifying SMS messages as either "spam" or "ham" (not spam) using machine learning techniques. The dataset used for this project contains SMS messages labeled as either spam or ham. The main goal is to build a text classification model that can accurately classify new SMS messages.

## Features

- **Dataset**: The dataset consists of SMS messages labeled as "spam" or "ham". Each entry includes the message content and its corresponding label.
- **Model**: The project uses a Naive Bayes classifier with TF-IDF vectorization to predict whether a message is spam or ham.
- **Evaluation Metrics**: The model's performance is evaluated using accuracy, confusion matrix, and classification report.

## Requirements

To run this project, you will need the following Python libraries:
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install the required libraries using pip.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The dataset used in this project is sourced from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).
- Special thanks to the contributors of the dataset and the scikit-learn library.
