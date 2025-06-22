import pandas as pd

# Load the dataset
df = pd.read_csv("spam_ham_dataset.csv")

# Show basic structure
print(df.head())
print(df.columns)
# Keep only the required columns
df = df[['label', 'text']]

# Encode label: spam = 1, ham = 0
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Features and target
X = df['text']
y = df['label_num']
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))
def predict_email(message):
    msg_vec = vectorizer.transform([message])
    prediction = model.predict(msg_vec)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Test examples
print(predict_email("Win a free iPhone now!"))
print(predict_email("Meeting rescheduled to 2 PM."))
