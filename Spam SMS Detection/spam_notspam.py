# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Step 1: Load the CSV Data with specified encoding
df = pd.read_csv(r"spam.csv")

# Inspect the column names
print("Column names in the dataset:")
print(df.columns)

# Rename the columns for easier access (if necessary)
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

# Preview the first few rows of the dataset
print("First few rows of the dataset after renaming:")
print(df.head())

# Step 2: Data Preprocessing
# Convert labels to numeric (e.g., 'ham' to 0 and 'spam' to 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Visualize the distribution of spam vs. ham messages
sns.countplot(x='label', data=df)
plt.title('Distribution of Spam vs Ham Messages')
plt.xlabel('Label (0: Ham, 1: Spam)')
plt.ylabel('Count')
plt.show()

# Step 3: Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Step 4: Vectorization using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Model Training with Logistic Regression
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 6: Prediction and Evaluation
y_pred = model.predict(X_test_tfidf)

# Print accuracy and classification report
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix visualization
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Step 7: Save the Model (Optional)
# Uncomment the following line to save the model
# joblib.dump(model, 'spam_sms_detector.pkl')

# Save the vectorizer for future use (Optional)
# Uncomment the following line to save the vectorizer
# joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
