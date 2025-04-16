import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string

# Functions
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Load and Prepare Data
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['length'] = df['message'].apply(len)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
df['clean_message'] = df['message'].apply(clean_text)

# Tokenize and Vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
x = tfidf.fit_transform(df['clean_message'])
y = df['label_num']

# Train-Test Split
from sklearn.model_selection import train_test_split
X = df['clean_message']
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_vect, y, test_size=0.2, random_state=42
)

# Train classifier
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate classifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Print results
#print (df.head())
#print (df.isnull().sum())
#print (df['label'].value_counts())
#print (f"CONFUSION MATRIX")
#print (confusion_matrix(y_test, y_pred))
#print (f"CLASSIFICATION REPORT")
#print (classification_report(y_test, y_pred))
print (f"ACCURACY: {accuracy_score(y_test, y_pred)}")
print (f"PRECISION: {precision_score(y_test, y_pred)}")
print (f"RECALL: {recall_score(y_test, y_pred)}")
print (f"F1 SCORE: {f1_score(y_test, y_pred)}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Visualize Data
# sns.countplot(x='label', data=df)
# sns.histplot(data=df, x='length', hue='label', bins=50, kde=True)
# plt.title("Spam vs Ham Distribution")
# plt.show()