# MOVIE GENRE CLASSIFICATION

##Definition: Movie Genre Classification involves the automatic identification of the genre of a movie based on its description. This process utilizes machine learning and natural language processing (NLP) techniques to analyze textual data, transforming it into meaningful patterns that a model can use to accurately predict the genre.

### 1. **Data Loading and Handling:**
   - The dataset is loaded from a CSV file into a Pandas DataFrame, with precautions taken to skip any malformed lines to prevent errors. The file is read with UTF-8 encoding to ensure that special characters are properly handled.

### 2. **Data Inspection:**
   - After loading, the dataset's structure is checked to confirm that the data includes the expected columns (`Genre` and `Description`). This step verifies that the data is correctly prepared for further processing.

### 3. **Preprocessing:**
   - The `Genre` column, which contains the target labels, is converted into numerical codes, making it suitable for machine learning models. Only the relevant columns (`Genre` and `Description`) are selected for the subsequent steps.

### 4. **Data Splitting:**
   - The dataset is divided into training and testing sets, with 80% of the data allocated for training and 20% for testing. This split ensures that the model is evaluated on data it has not seen before, providing a true test of its predictive power.

### 5. **Feature Extraction Using TF-IDF:**
   - TF-IDF (Term Frequency-Inverse Document Frequency) is employed to convert textual movie descriptions into numerical features. This method emphasizes important words that are significant within the dataset while filtering out common words that may not contribute meaningfully to genre classification. Words appearing in more than 70% of the documents are ignored to reduce noise.

### 6. **Model Training:**
   - A Multinomial Naive Bayes classifier is used for training. This algorithm is well-suited for text classification tasks, as it assumes that features (words) are conditionally independent given the class (genre), simplifying the computation and improving efficiency.

### 7. **Model Prediction:**
   - The trained model is used to predict the genres of the movies in the test set. These predictions are then compared to the actual genres to evaluate the model’s performance.

### 8. **Model Evaluation:**
   - The model’s performance is assessed using several metrics:
     - **Accuracy** measures the proportion of correct predictions.
     - **Confusion Matrix** provides a detailed view of correct and incorrect predictions for each genre, highlighting areas where the model performs well or needs improvement.
     - **Classification Report** includes precision, recall, and F1-score for each genre, giving a comprehensive overview of the model's effectiveness across all classes.

### 9. **Custom Prediction:**
   - The project includes a demonstration of how the model can predict the genre of a new, unseen movie description, showcasing its practical application.

### 10. **Genre Mapping:**
   - The predicted numerical genre codes are mapped back to their original genre names, making the predictions more interpretable and user-friendly.

### Overall:
"MOVIE GENRE CLASSIFICATION" is a project that effectively combines natural language processing (NLP) and machine learning to classify movie genres based on descriptions. By leveraging TF-IDF for feature extraction and a Multinomial Naive Bayes classifier for prediction, the project demonstrates a powerful approach to automated genre classification, making it suitable for real-world applications.
