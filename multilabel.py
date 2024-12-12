# Importieren der notwendigen Bibliotheken
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report


# loading the dataset
csv_path = "C:/Users/chiar/Downloads/dataset.csv"
df = pd.read_csv(csv_path)

# defining the columns for the emotion, we're ignoring the id column since it's not relevant 
emotion_columns = ['anger', 'fear', 'suprise', 'sadness', 'joy', 'disgust', 'envy', 'jealousy', 'other']
text_column = 'description'
X_text = df[text_column]
y = df[emotion_columns]


# using TF-IDF with a limited amount of 5000 most relevant words (to prevent Overfitting)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X_text)

# using 70% of Data to train and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# defining the baseline systems and wrapping them
base_model_1 = RandomForestClassifier(n_estimators=100, random_state=42)
base_model_2 = SVC(kernel='linear', probability=True, random_state=42)
base_model_3 = MultinomialNB()

multi_rf = MultiOutputClassifier(base_model_1)
multi_svc = MultiOutputClassifier(base_model_2)
multi_nb = MultiOutputClassifier(base_model_3)

# creating a soft voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', multi_rf),
        ('svc', multi_svc),
        ('nb', multi_nb)
    ],
    voting='soft'
)

# creating a stacking classifier
estimators = [
    ('rf', multi_rf),
    ('svc', multi_svc)
]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000)
)

# train the models
print("Training Voting Classifier...")
voting_clf.fit(X_train, y_train)

print("\nTraining Stacking Classifier...")
stacking_clf.fit(X_train, y_train)
