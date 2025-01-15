import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import vstack

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

# Bereinigung: NaN-Werte in Textdaten entfernen
tweets = tweets.fillna('')  # NaN-Werte durch leere Strings ersetzen
tweets = tweets.astype(str)  # Sicherstellen, dass alle Werte Strings sind

# Textdaten vektorisieren
vectorizer = TfidfVectorizer(max_features=1000)  # Begrenze die Anzahl der Features
X = vectorizer.fit_transform(tweets)  # Numerische Repräsentation der Texte
X_additional = vectorizer.transform(additional_tweets)

# KFold initialisieren
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold Cross Validation

# Ergebnis-Listen für F1-Scores
macro_f1_scores = []
micro_f1_scores = []
weighted_f1_scores = []

# 2. Cross-Validation-Schleife
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    
    # Train- und Testdaten aufteilen
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
    
    # Zusätzliche Daten zum Trainingsset hinzufügen
    X_train = vstack([X[train_index], X_additional])
    y_train = pd.concat([y_train, additional_labels], ignore_index=True)
    
    # Daten skalieren
    scaler = MaxAbsScaler() # Sparse-Matrix
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Modelle für jedes Label trainieren
    models = {}
    y_pred_combined = np.zeros_like(y_test)  # Array für kombinierte Vorhersagen
    
    # Vorhersagen kombinieren
    for label in labels.columns:
        print(f"  Training für Label: {label}")
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train[label])
        models[label] = model

        # Vorhersage
        y_pred = model.predict(X_test)
        y_pred_combined[:, labels.columns.get_loc(label)] = y_pred

    # NEUER CODE: Evaluation der einzelnen Labels und kombinierte F1-Scores
    label_f1_scores_macro = []
    label_f1_scores_micro = []
    label_f1_scores_weighted = []

    for label_idx, label in enumerate(labels.columns):
        f1_macro = f1_score(y_test.iloc[:, label_idx], y_pred_combined[:, label_idx], average='macro')
        f1_micro = f1_score(y_test.iloc[:, label_idx], y_pred_combined[:, label_idx], average='micro')
        f1_weighted = f1_score(y_test.iloc[:, label_idx], y_pred_combined[:, label_idx], average='weighted')
    
        label_f1_scores_macro.append(f1_macro)
        label_f1_scores_micro.append(f1_micro)
        label_f1_scores_weighted.append(f1_weighted)

    # Kombinierte F1-Scores berechnen
    combined_f1_macro = np.mean(label_f1_scores_macro)  # Durchschnitt über alle Labels
    combined_f1_micro = np.mean(label_f1_scores_micro)
    combined_f1_weighted = np.mean(label_f1_scores_weighted)

    print(f"  Fold {fold + 1}: Macro F1 = {combined_f1_macro:.2f}, Micro F1 = {combined_f1_micro:.2f}, Weighted F1 = {combined_f1_weighted:.2f}")

# 4. Durchschnittliche Ergebnisse über alle Folds
print("\nDurchschnittliche Ergebnisse über alle Folds:")
print(f"Macro F1: {np.mean(macro_f1_scores):.2f} ± {np.std(macro_f1_scores):.2f}")
print(f"Micro F1: {np.mean(micro_f1_scores):.2f} ± {np.std(micro_f1_scores):.2f}")
print(f"Weighted F1: {np.mean(weighted_f1_scores):.2f} ± {np.std(weighted_f1_scores):.2f}")
