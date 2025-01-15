# Importiere die benötigten Bibliotheken
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

# Laden der Tweets und Extrahieren der Labels
tweets = []
labels = []

with open("/Users/chanti/Desktop/5. Semester/Softwareprojekt/Code/Emotion-Detection/intermediate_results/preprocessed_results_2.txt", "r", encoding="utf-8") as file:
    for line in file:
        try:
            tweet, label = line.rsplit(",", 1)  # Splitting the preprocessed tweets because the labels are after the last comma
            tweets.append(tweet.strip())
            labels.append(1 if label.strip().lower() == "yes" else 0)  # Changing the labels from yes/no to 1/0 -> yes = 1, no = 0
        except ValueError:
            print("Fehlerhafte Zeile:", line)

# Splitting the data into train (80%) and test (20%) data
X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.2)

# Generating a BoW representation as our feature
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(tweets)  # Vectorizing all tweets
Y = labels

# Since we have 5637 tweets with emotions and 3460 tweets with no emotions, we use SMOTE to balance our dataset
smote = SMOTE(random_state=42)  # random_state = 42 -> same output in every run

# Resampling the data
X_resampled, y_resampled = smote.fit_resample(X, Y)

# Training a multinomial naive bayes as our baseline system
nb_classifier = MultinomialNB()

# Trainiere zusätzlich einen CatBoostClassifier für binäre Klassifikation
catboost_classifier = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=False)  # <-- CatBoost-Modell für binäre Klassifikation

# Defining scoring for cross-validation
scoring = {
    'precision_macro': make_scorer(precision_score, average='macro'),
    'recall_macro': make_scorer(recall_score, average='macro'),
    'f1_macro': make_scorer(f1_score, average='macro')
}  # We want to evaluate our model using macro precision, macro recall, and macro f1

# Cross-Validation für Naive Bayes
cv_results_nb = cross_validate(nb_classifier, X_resampled, y_resampled, cv=5, scoring=scoring)
y_pred_nb = cross_val_predict(nb_classifier, X_resampled, y_resampled, cv=5)

# Cross-Validation für CatBoost
cv_results_catboost = cross_validate(catboost_classifier, X_resampled, y_resampled, cv=5, scoring=scoring)  # <-- Cross-Validation für CatBoost
y_pred_catboost = cross_val_predict(catboost_classifier, X_resampled, y_resampled, cv=5)  # <-- Vorhersagen für CatBoost

# Ergebnisse Naive Bayes
print("Naive Bayes Ergebnisse:")
print(f"Precision: {cv_results_nb['test_precision_macro'].mean():.4f}")
print(f"Recall: {cv_results_nb['test_recall_macro'].mean():.4f}")
print(f"F1 Score: {cv_results_nb['test_f1_macro'].mean():.4f}")

# Ergebnisse CatBoost
print("CatBoost Ergebnisse:")  # <-- Ausgabe der Ergebnisse von CatBoost
print(f"Precision: {cv_results_catboost['test_precision_macro'].mean():.4f}")
print(f"Recall: {cv_results_catboost['test_recall_macro'].mean():.4f}")
print(f"F1 Score: {cv_results_catboost['test_f1_macro'].mean():.4f}")
