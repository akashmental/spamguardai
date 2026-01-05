
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
data = pd.read_csv("Dataset/spam.csv")
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

# Input/output split
msg = data['Message']
cat = data['Category']

# Train/test split
msg_train, msg_test, cat_train, cat_test = train_test_split(msg, cat, test_size=0.2, random_state=42)

# Vectorizer
cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(msg_train)

# Model
model = MultinomialNB()
model.fit(features, cat_train)

# -------- Function for prediction --------
def predict(message):
    input_message = cv.transform([message])
    return model.predict(input_message)[0]

# -------- Evaluation Section --------
# Transform test messages
test_features = cv.transform(msg_test)

# Predict on test set
predictions = model.predict(test_features)

# Calculate metrics
accuracy = accuracy_score(cat_test, predictions) * 100
precision = precision_score(cat_test, predictions, pos_label="Spam") * 100
recall = recall_score(cat_test, predictions, pos_label="Spam") * 100
f1 = f1_score(cat_test, predictions, pos_label="Spam") * 100

# Print summary
print("===== Spam Email Classifier Summary =====")
print(f"Total Messages in Dataset: {len(data)}")
print("----- Naive Bayes Performance -----")
print(f"Accuracy : {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall   : {recall:.2f}%")
print(f"F1-Score : {f1:.2f}%")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
data = pd.read_csv("Dataset/spam.csv")
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

# Input/output split
msg = data['Message']
cat = data['Category']

# Train/test split
msg_train, msg_test, cat_train, cat_test = train_test_split(msg, cat, test_size=0.2, random_state=42)

# Vectorizer
cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(msg_train)

# Model
model = MultinomialNB()
model.fit(features, cat_train)

# -------- Function for prediction --------
def predict(message):
    input_message = cv.transform([message])
    return model.predict(input_message)[0]

# -------- Evaluation Section --------
# Transform test messages
test_features = cv.transform(msg_test)

# Predict on test set
predictions = model.predict(test_features)

# Calculate metrics
accuracy = accuracy_score(cat_test, predictions) * 100
precision = precision_score(cat_test, predictions, pos_label="Spam") * 100
recall = recall_score(cat_test, predictions, pos_label="Spam") * 100
f1 = f1_score(cat_test, predictions, pos_label="Spam") * 100

# Print summary
print("===== Spam Email Classifier Summary =====")
print(f"Total Messages in Dataset: {len(data)}")
print("----- Naive Bayes Performance -----")
print(f"Accuracy : {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall   : {recall:.2f}%")
print(f"F1-Score : {f1:.2f}%")
print("========================================")