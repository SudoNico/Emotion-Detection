import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# NLTK vorbereiten
nltk.download('punkt')

# 1. Daten einlesen und vorbereiten
data = pd.read_csv('/Users/chanti/Desktop/5. Semester/Softwareprojekt/Code/Emotion-Detection/intermediate_results/preprocessed_results_ems.txt', sep=',', header=None)
data.columns = ['Tweet'] + [f'Label{i+1}' for i in range(9)]  # Spalten benennen

additional_data = pd.read_csv('/Users/chanti/Desktop/5. Semester/Softwareprojekt/Code/Emotion-Detection/intermediate_results/preprocessed_results_GoldStandardGerman.txt', sep=',', header=None)
additional_data.columns = ['Tweet'] + [f'Label{i+1}' for i in range(9)]  # Gleiche Spaltennamen

# Tweets und Labels extrahieren
tweets = data['Tweet']
labels = data.iloc[:, 1:]

additional_tweets = additional_data['Tweet']
additional_labels = additional_data.iloc[:, 1:]

# Bereinigung: Entferne Leerzeichen und konvertiere zu Integer
labels = labels.applymap(lambda x: int(str(x).strip().lower() in ['1', 'true']))
# Bereinigung: NaN-Werte in Textdaten entfernen
tweets = tweets.fillna('')  # NaN-Werte durch leere Strings ersetzen
tweets = tweets.astype(str)  # Sicherstellen, dass alle Werte Strings sind

# Bereinigung: Entferne Leerzeichen und konvertiere zu Integer
additional_labels = additional_labels.applymap(lambda x: int(str(x).strip().lower() in ['1', 'true']))
# Bereinigung: NaN-Werte in Textdaten entfernen
additional_tweets = additional_tweets.fillna('')  # NaN-Werte durch leere Strings ersetzen
additional_tweets = additional_tweets.astype(str)  # Sicherstellen, dass alle Werte Strings sind

# 2. TF-IDF berechnen
tfidf_vectorizer = TfidfVectorizer(analyzer=lambda x: x, token_pattern=None)  # Wir nutzen die Tokenisierung direkt
tfidf_matrix = tfidf_vectorizer.fit_transform(tweets)
tfidf_vocab = tfidf_vectorizer.vocabulary_

# 3. Word2Vec trainieren
word2vec_model = Word2Vec(sentences=tweets, vector_size=100, window=5, min_count=1, workers=4)

# Funktion, um TF-IDF-gewichteten Word2Vec-Vektor zu berechnen
def get_tfidf_weighted_word2vec(tokens, model, tfidf_vocab, tfidf_matrix, index, vector_size=100):
    vector = np.zeros(vector_size)
    tfidf_values = tfidf_matrix[index]
    
    valid_words = 0
    for word in tokens:
        if word in model.wv and word in tfidf_vocab:
            tfidf_weight = tfidf_values[0, tfidf_vocab[word]]  # TF-IDF-Wert des Wortes
            vector += model.wv[word] * tfidf_weight
            valid_words += tfidf_weight
    
    if valid_words > 0:
        vector /= valid_words  # Durchschnitt der gewichteten Vektoren
    return vector

# TF-IDF-gewichtete Word2Vec-Features berechnen
X_combined = np.array([
    get_tfidf_weighted_word2vec(tokens, word2vec_model, tfidf_vocab, tfidf_matrix, i)
    for i, tokens in enumerate(tweets)
])

# 2. TF-IDF berechnen
additional_tfidf_matrix = tfidf_vectorizer.fit_transform(additional_tweets)
tfidf_vocab = tfidf_vectorizer.vocabulary_

# 3. Word2Vec trainieren
additional_word2vec_model = Word2Vec(sentences=additional_tweets, vector_size=100, window=5, min_count=1, workers=4)

additional_X_combined = np.array([
    get_tfidf_weighted_word2vec(tokens, additional_word2vec_model, tfidf_vocab, additional_tfidf_matrix, i)
    for i, tokens in enumerate(additional_tweets)
])
# 4. KFold initialisieren
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Ergebnisse speichern
macro_f1_scores_list = []
micro_f1_scores_list = []
weighted_f1_scores_list = []
macro_recall_list = []
macro_precision_list = []

# Liste der Emotionen in der gleichen Reihenfolge wie die Spalten "Label1", "Label2", ..., "Label9"
emotions = ['anger', 'fear', 'surprise', 'sadness', 'joy', 'disgust', 'envy', 'jealousy', 'other']

# Cross-Validation Schleife
for fold, (train_index, test_index) in enumerate(kf.split(X_combined)):
    print(f"Fold {fold + 1}")
    
    # Train- und Testdaten aufteilen
    X_train, X_test = X_combined[train_index], X_combined[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
    
    X_train = np.vstack([X_train, additional_X_combined])
    y_train = pd.concat([y_train, additional_labels], ignore_index=True)
    
    y_pred_combined = np.zeros_like(y_test)
    
    for i, label in enumerate(labels.columns):
        emotion_name = emotions[i]  # Hole den Namen der Emotion aus der Liste
        print(f"Training für Label: {emotion_name}")
        
        # Klassen-Gewichte für CatBoost berechnen
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=[0, 1],
            y=y_train[label]
        )
        class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        # VotingClassifier initialisieren
        voting_clf = VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
            ('xgb', XGBClassifier(n_estimators=200, random_state=42, scale_pos_weight=class_weights[1])),
            ('catboost', CatBoostClassifier(iterations=200, verbose=0, class_weights=class_weights))
        ], voting='soft')
        
        # Modell trainieren
        voting_clf.fit(X_train, y_train[label])
        y_pred = voting_clf.predict(X_test)
        y_pred_combined[:, labels.columns.get_loc(label)] = y_pred
    
    # F1-Scores und Metriken berechnen
    macro_f1_scores = []
    micro_f1_scores = []
    weighted_f1_scores = []
    macro_recall = []
    macro_precision = []
    
    for i in range(y_test.shape[1]):
        f1_macro = f1_score(y_test.iloc[:, i], y_pred_combined[:, i], average='macro', zero_division=0)
        f1_micro = f1_score(y_test.iloc[:, i], y_pred_combined[:, i], average='micro', zero_division=0)
        f1_weighted = f1_score(y_test.iloc[:, i], y_pred_combined[:, i], average='weighted', zero_division=0)
        precision_macro = precision_score(y_test.iloc[:, i], y_pred_combined[:, i], average='macro', zero_division=0)
        recall_macro = recall_score(y_test.iloc[:, i], y_pred_combined[:, i], average='macro', zero_division=0)
        
        macro_f1_scores.append(f1_macro)
        micro_f1_scores.append(f1_micro)
        weighted_f1_scores.append(f1_weighted)
        macro_recall.append(recall_macro)
        macro_precision.append(precision_macro)
    
    # Durchschnittliche Metriken speichern
    macro_f1_scores_list.append(np.mean(macro_f1_scores))
    micro_f1_scores_list.append(np.mean(micro_f1_scores))
    weighted_f1_scores_list.append(np.mean(weighted_f1_scores))
    macro_recall_list.append(np.mean(macro_recall))
    macro_precision_list.append(np.mean(macro_precision))

# Durchschnittliche Ergebnisse über alle Folds
print("\nDurchschnittliche Ergebnisse über alle Folds:")
print(f"Macro F1: {np.mean(macro_f1_scores_list):.2f} ± {np.std(macro_f1_scores_list):.2f}")
print(f"Micro F1: {np.mean(micro_f1_scores_list):.2f} ± {np.std(micro_f1_scores_list):.2f}")
print(f"Weighted F1: {np.mean(weighted_f1_scores_list):.2f} ± {np.std(weighted_f1_scores_list):.2f}")
print(f"Macro Recall: {np.mean(macro_recall_list):.2f} ± {np.std(macro_recall_list):.2f}")
print(f"Macro Precision: {np.mean(macro_precision_list):.2f} ± {np.std(macro_precision_list):.2f}")