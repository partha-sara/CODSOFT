import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the Dataset
try:
    data = pd.read_csv(
        r"C:\Users\praji\Downloads\archive (2)\Genre Classification Dataset\train_data.csv", 
        delimiter=',', 
        on_bad_lines='skip',  # Skip bad lines
        encoding='utf-8'
    )
except pd.errors.ParserError as e:
    print(f"Error reading the CSV file: {e}")
    data = None

if data is not None:
    # Check the first few rows to ensure the data loaded correctly
    print("Data loaded successfully!")
    print(data.head())
    
    # Inspect the columns to make sure they match expected names
    print("Columns in the dataset:", data.columns)

    # Assuming the dataset has columns ['Genre', 'Description']
    if 'Genre' in data.columns and 'Description' in data.columns:
        data = data[['Genre', 'Description']]

        # Step 2: Preprocessing
        # Convert genres to categorical codes (if not already done)
        data['Genre'] = data['Genre'].astype('category').cat.codes

        # Step 3: Split the Dataset
        X_train, X_test, y_train, y_test = train_test_split(data['Description'], data['Genre'], test_size=0.2, random_state=42)

        # Step 4: Feature Extraction using TF-IDF
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        # Step 5: Train the Naive Bayes Classifier
        model = MultinomialNB()
        model.fit(X_train_tfidf, y_train)

        # Step 6: Make Predictions
        y_pred = model.predict(X_test_tfidf)

        # Step 7: Evaluate the Model
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)

        # Optionally, you can test with a custom movie description
        sample_description = ["A young man, who's been searching for his father all his life, finally discovers where he is."]
        sample_description_tfidf = tfidf_vectorizer.transform(sample_description)
        prediction = model.predict(sample_description_tfidf)
        print("Predicted Genre Code for sample description:", prediction[0])
        
        # Optionally, map the genre code back to the original genre if needed
        genre_map = {0: 'Action', 1: 'Comedy', 2: 'Drama'}  # Adjust this map as per your dataset
        print("Predicted Genre for sample description:", genre_map.get(prediction[0], "Unknown"))
    else:
        print("Expected columns 'Genre' and 'Description' are missing in the dataset.")

else:
    print("Failed to load data.")
