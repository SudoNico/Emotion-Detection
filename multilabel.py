import pandas as pd
import numpy as np
import gensim
from imblearn.over_sampling import RandomOverSampler
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate

#https://github.com/Hamxea/Multi-label-Classification and ChatGPT for corrections

# loading the dataset
txt_path = "/Users/chanti/Desktop/5. Semester/Softwareprojekt/Code/Emotion-Detection/intermediate_results/preprocessed_results_multi.txt"
data = pd.read_csv(txt_path, sep=',') 
pd.set_option('future.no_silent_downcasting', True)

# giving the text column names for future processing so we can access the columns 
columns = ['Tweet'] + [f'Label{i+1}' for i in range(9)]
data.columns = columns

# Überprüfe auf ungültige Tweets (NaN oder leere Strings)
invalid_rows = data[data['Tweet'].isna() | (data['Tweet'] == '')]

# Entferne ungültige Zeilen
data = data.dropna(subset=['Tweet'])
data = data[data['Tweet'] != '']

# tokenising the tweets 
tokenized_tweets = [word_tokenize(tweet.lower()) for tweet in data['Tweet']]

# training a word2vec model -> 
word2vec_model = Word2Vec(sentences=tokenized_tweets, vector_size=100, window=5, min_count=1, workers=4)


# generating word2vec for each tweet, not each word 
def tweet_vector(tweet, model):
    words = word_tokenize(tweet.lower())
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# generating the feature-matrix 
X_word2vec = np.array([tweet_vector(tweet, word2vec_model) for tweet in data['Tweet']])

# Schritt 1: Leerzeichen entfernen und Strings bereinigen
data.iloc[:, 1:] = data.iloc[:, 1:].apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

data.iloc[:, 1:] = data.iloc[:, 1:].astype(str).apply(lambda col: col.str.lower().str.strip())

# Schritt 2: 'true' zu 1 und 'false' zu 0 ersetzen
data.iloc[:, 1:] = data.iloc[:, 1:].replace({'true': 1, 'false': 0}).astype(int)

# Labels extrahieren
Y = data.iloc[:, 1:].values


# using TF-IDF with a limited amount of 5000 most relevant words (to prevent Overfitting)
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(data['Tweet']).toarray()

# combining word2vec and TF-IDF 
X_combined = np.hstack((X_word2vec, X_tfidf))

scaler = StandardScaler()
X_combined = scaler.fit_transform(X_combined)


# Aufteilen der Daten in Training und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X_combined, Y, test_size=0.2, random_state=42)
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Erstellen der Basismodelle
base_model_1 = RandomForestClassifier(n_estimators=100, random_state=42)
base_model_2 = SVC(kernel='linear', probability=True, random_state=42)
base_model_3 = GaussianNB()

# Definiere die Anzahl der Folds
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Erstelle das Voting Classifier Modell
voting_clf = MultiOutputClassifier(VotingClassifier(
    estimators=[
        ('rf', base_model_1),
        ('svc', base_model_2),
        ('gnb', base_model_3)
    ],
    voting='soft'
))

# Speichere Metriken
precision_scores = []
recall_scores = []
f1_scores = []

# Cross-Validation-Schleife
print(f"Starte {k}-Fold Cross-Validation...")
for train_index, test_index in kf.split(X_combined):
    X_train, X_test = X_combined[train_index], X_combined[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    # Trainiere das Modell
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    
    # Berechne die Metriken (macro precision, recall, f1)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Speichere die Ergebnisse
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    print(f"Fold abgeschlossen - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# Durchschnittliche Metriken berechnen
print("\nCross-Validation Ergebnisse:")
print(f"Durchschnittliche Precision (macro): {np.mean(precision_scores):.4f}")
print(f"Durchschnittliche Recall (macro): {np.mean(recall_scores):.4f}")
print(f"Durchschnittliche F1-Score (macro): {np.mean(f1_scores):.4f}")

# Erstellen des Stacking Classifiers
estimators = [
    ('rf', base_model_1),
    ('svc', base_model_2)
]

stacking_clf = MultiOutputClassifier(StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000)
))

# Speichere Metriken
precision_scores_stacking = []
recall_scores_stacking = []
f1_scores_stacking = []

# Cross-Validation-Schleife
print(f"\nStarte {k}-Fold Cross-Validation für den Stacking Classifier...")
for train_index, test_index in kf.split(X_combined):
    X_train, X_test = X_combined[train_index], X_combined[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    # Trainiere den StackingClassifier
    stacking_clf.fit(X_train, y_train)
    y_pred = stacking_clf.predict(X_test)
    
    # Berechne die Metriken (macro precision, recall, f1)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Speichere die Ergebnisse
    precision_scores_stacking.append(precision)
    recall_scores_stacking.append(recall)
    f1_scores_stacking.append(f1)

    print(f"Fold abgeschlossen - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")


# Zählen, wie viele 1en es für jedes Label gibt
label_counts = np.sum(Y, axis=0)

# Erstelle ein DataFrame, um die Labelverteilung übersichtlicher darzustellen
label_names = [f'Label{i+1}' for i in range(Y.shape[1])]
label_distribution = pd.DataFrame({
    'Label': label_names,
    'Count': label_counts,
    'Percentage': (label_counts / Y.shape[0]) * 100  # Prozentsatz der Beispiele für jedes Label
})

# Ausgabe der Verteilung
print(label_distribution)

# Visualisierung der Verteilung der Labels
plt.figure(figsize=(10, 6))
sns.barplot(x='Label', y='Percentage', data=label_distribution, palette='viridis')
plt.title('Verteilung der Labels')
plt.ylabel('Prozentuale Häufigkeit')
plt.xlabel('Label')
plt.xticks(rotation=45)
plt.show()