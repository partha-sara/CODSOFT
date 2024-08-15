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

You can install the required libraries using pip:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## File Structure

- `spam_classification.py`: The main script that performs the following tasks:
  - Loads and preprocesses the dataset.
  - Splits the dataset into training and testing sets.
  - Vectorizes text data using TF-IDF.
  - Trains a Naive Bayes classifier.
  - Evaluates the model's performance.
  - Provides visualizations for model evaluation.

- `data/`: Directory containing the dataset file:
  - `spam.csv`: The dataset file with SMS messages and their labels.

## How to Run

1. **Download the Dataset**: Ensure the dataset file `spam.csv` is located in the `data/` directory.

2. **Run the Script**:
   ```bash
   python spam_classification.py
   ```

   This will execute the script, which will perform the following:
   - Load and preprocess the data.
   - Train a Naive Bayes classifier.
   - Print the model's accuracy, confusion matrix, and classification report.
   - Display visualizations for model evaluation.

## Example Output

The script will print the following information:
- **Accuracy**: The overall accuracy of the model.
- **Confusion Matrix**: A matrix showing the true vs. predicted classifications.
- **Classification Report**: Detailed metrics including precision, recall, and F1-score for each class.

Visualizations will include:
- **Confusion Matrix Heatmap**: A visual representation of the confusion matrix.
- **Distribution of Predictions**: A bar plot showing the distribution of predicted labels.

## Optional: Testing with Custom Messages

You can test the model with custom SMS messages by modifying the `sample_message` variable in the script. The script will output the predicted label for the given message.

```python
sample_message = ["Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/123456 to claim now."]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The dataset used in this project is sourced from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).
- Special thanks to the contributors of the dataset and the scikit-learn library.
