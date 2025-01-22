import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import vstack
from sklearn.ensemble import RandomForestClassifier

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
kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold Cross Validation

# Ergebnis-Listen für Metriken
label_metrics = {label: {'precision': [], 'recall': [], 'f1_macro': [], 'f1_micro': [], 'f1_weighted': []} for label in labels.columns}
combined_metrics = {'precision': [], 'recall': [], 'f1_macro': [], 'f1_micro': [], 'f1_weighted': []}

emotions = ['anger', 'fear', 'surprise', 'sadness', 'joy', 'disgust', 'envy', 'jealousy', 'other']

# 2. Cross-Validation-Schleife
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    
    # Train- und Testdaten aufteilen
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
    
    # Zusätzliche Daten zum Trainingsset hinzufügen
    X_train = vstack([X[train_index], X_additional])
    y_train = pd.concat([y_train, additional_labels], ignore_index=True)
    
    X_train_dense = X_train.toarray()
    
    # Daten skalieren
    scaler = MaxAbsScaler() # Sparse-Matrix
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Modell für jedes Label trainieren
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    y_pred_combined = np.zeros_like(y_test)  # Array für kombinierte Vorhersagen

    # Vorhersagen für jedes Label
    for i, label in enumerate(labels.columns):
        emotion_name = emotions[i]  # Hole den Namen der Emotion aus der Liste
        print(f"Training für Label: {emotion_name}")
        model.fit(X_train, y_train[label])

        # Vorhersage
        y_pred = model.predict(X_test)
        y_pred_combined[:, labels.columns.get_loc(label)] = y_pred

        # Metriken für jedes Label berechnen
        precision_macro = precision_score(y_test.iloc[:, labels.columns.get_loc(label)], y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test.iloc[:, labels.columns.get_loc(label)], y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test.iloc[:, labels.columns.get_loc(label)], y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_test.iloc[:, labels.columns.get_loc(label)], y_pred, average='micro', zero_division=0)
        f1_weighted = f1_score(y_test.iloc[:, labels.columns.get_loc(label)], y_pred, average='weighted', zero_division=0)

        # Speichern der Metriken für das Label
        label_metrics[label]['precision'].append(precision_macro)
        label_metrics[label]['recall'].append(recall_macro)
        label_metrics[label]['f1_macro'].append(f1_macro)
        label_metrics[label]['f1_micro'].append(f1_micro)
        label_metrics[label]['f1_weighted'].append(f1_weighted)

    # NEUER CODE: Kombinierte Metriken berechnen
    combined_precision = np.mean([precision_score(y_test.iloc[:, label_idx], y_pred_combined[:, label_idx], average='macro', zero_division=0) for label_idx in range(y_test.shape[1])])
    combined_recall = np.mean([recall_score(y_test.iloc[:, label_idx], y_pred_combined[:, label_idx], average='macro', zero_division=0) for label_idx in range(y_test.shape[1])])
    combined_f1_macro = np.mean([f1_score(y_test.iloc[:, label_idx], y_pred_combined[:, label_idx], average='macro', zero_division=0) for label_idx in range(y_test.shape[1])])
    combined_f1_micro = np.mean([f1_score(y_test.iloc[:, label_idx], y_pred_combined[:, label_idx], average='micro', zero_division=0) for label_idx in range(y_test.shape[1])])
    combined_f1_weighted = np.mean([f1_score(y_test.iloc[:, label_idx], y_pred_combined[:, label_idx], average='weighted', zero_division=0) for label_idx in range(y_test.shape[1])])

    # Speichern der kombinierten Metriken
    combined_metrics['precision'].append(combined_precision)
    combined_metrics['recall'].append(combined_recall)
    combined_metrics['f1_macro'].append(combined_f1_macro)
    combined_metrics['f1_micro'].append(combined_f1_micro)
    combined_metrics['f1_weighted'].append(combined_f1_weighted)

# 3. Zusammenfassung der Ergebnisse nach allen Folds
print("\nZusammenfassung der Metriken über alle Folds:")

# Ergebnisse für jedes Label
for label in labels.columns:
    print(f"\nMetriken für Label {label}:")
    print(f"  Precision (Macro): {np.mean(label_metrics[label]['precision']):.4f} ± {np.std(label_metrics[label]['precision']):.4f}")
    print(f"  Recall (Macro): {np.mean(label_metrics[label]['recall']):.4f} ± {np.std(label_metrics[label]['recall']):.4f}")
    print(f"  F1-Score (Macro): {np.mean(label_metrics[label]['f1_macro']):.4f} ± {np.std(label_metrics[label]['f1_macro']):.4f}")
    print(f"  F1-Score (Micro): {np.mean(label_metrics[label]['f1_micro']):.4f} ± {np.std(label_metrics[label]['f1_micro']):.4f}")
    print(f"  F1-Score (Weighted): {np.mean(label_metrics[label]['f1_weighted']):.4f} ± {np.std(label_metrics[label]['f1_weighted']):.4f}")

# Zusammenfassende Ergebnisse für das Gesamtsystem
print("\nGesamte Modellmetriken über alle Folds:")
print(f"  Macro Precision: {np.mean(combined_metrics['precision']):.4f} ± {np.std(combined_metrics['precision']):.4f}")
print(f"  Macro Recall: {np.mean(combined_metrics['recall']):.4f} ± {np.std(combined_metrics['recall']):.4f}")
print(f"  Macro F1-Score: {np.mean(combined_metrics['f1_macro']):.4f} ± {np.std(combined_metrics['f1_macro']):.4f}")
print(f"  Micro F1-Score: {np.mean(combined_metrics['f1_micro']):.4f} ± {np.std(combined_metrics['f1_micro']):.4f}")
print(f"  Weighted F1-Score: {np.mean(combined_metrics['f1_weighted']):.4f} ± {np.std(combined_metrics['f1_weighted']):.4f}")