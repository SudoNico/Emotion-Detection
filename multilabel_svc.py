# This script trains a Multi-Label Classfication System, namely a SVM model
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import vstack
from sklearn.svm import SVC

# read the results from the preprocessing of the annotated tweets of the data ems.csv
data = pd.read_csv('preprocessed_results_ems.txt', sep=',', header=None)
data.columns = ['Tweet'] + [f'Label{i+1}' for i in range(9)]   # define column names

# extract the tweets and the names of the labels
tweets = data['Tweet']
labels = data.iloc[:, 1:]

# remove all NaN-Values
tweets = tweets.fillna('')  # replace NaN-values with empty strings
tweets = tweets.astype(str)  # turn every tweet into a string

# vectorize the text data
vectorizer = TfidfVectorizer(max_features=1000)  # limit number of features to 1000
X = vectorizer.fit_transform(tweets)  # numeric representation of the tweet texts

# KFold initiation
kf = KFold(n_splits=10, shuffle=True, random_state=42) # 10 cross fold validation; shuffling the data before splitting it 

# lists for results for the metrics: precision, recall, macro f1, micro f1, weighted f1
label_metrics = {label: {'precision': [], 'recall': [], 'f1_macro': [], 'f1_micro': [], 'f1_weighted': []} for label in labels.columns}
combined_metrics = {'precision': [], 'recall': [], 'f1_macro': [], 'f1_micro': [], 'f1_weighted': []}

# list with the names of the emotions matching the order of the columns in the annotated data
emotions = ['anger', 'fear', 'surprise', 'sadness', 'joy', 'disgust', 'envy', 'jealousy', 'other']

# 10 times loop (for every fold of the Cross Validation)
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    
    # split the vectorized data of the modified preprocessed ems.csv in training and testing data for the model
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
    
    X_train_dense = X_train.toarray()
    
     # scaling data into a value between [-1,1]
    scaler = MaxAbsScaler() # Sparse matrix (which was created by using TF-IDF) remains
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # create a model for c-support vector classification
    model = SVC(kernel='linear', probability=True) 
    y_pred_combined = np.zeros_like(y_test) # array for combined predictions
    
    # prediction for every label
    for i, label in enumerate(labels.columns):
        emotion_name = emotions[i]  # get name of label from the list
        print(f"Training für Label: {emotion_name}")
        model.fit(X_train, y_train[label])

        # prediction of the labels of the test data
        y_pred = model.predict(X_test)
        y_pred_combined[:, labels.columns.get_loc(label)] = y_pred

        # calculate the metrics precision, recall, macro f1, micro f1 and weighted f1 for every label        
        precision_macro = precision_score(y_test.iloc[:, labels.columns.get_loc(label)], y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test.iloc[:, labels.columns.get_loc(label)], y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test.iloc[:, labels.columns.get_loc(label)], y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_test.iloc[:, labels.columns.get_loc(label)], y_pred, average='micro', zero_division=0)
        f1_weighted = f1_score(y_test.iloc[:, labels.columns.get_loc(label)], y_pred, average='weighted', zero_division=0)

        # save metrics for every label in arrays
        label_metrics[label]['precision'].append(precision_macro)
        label_metrics[label]['recall'].append(recall_macro)
        label_metrics[label]['f1_macro'].append(f1_macro)
        label_metrics[label]['f1_micro'].append(f1_micro)
        label_metrics[label]['f1_weighted'].append(f1_weighted)

    # get the mean value of the metrics of every label
    combined_precision = np.mean([precision_score(y_test.iloc[:, label_idx], y_pred_combined[:, label_idx], average='macro', zero_division=0) for label_idx in range(y_test.shape[1])])
    combined_recall = np.mean([recall_score(y_test.iloc[:, label_idx], y_pred_combined[:, label_idx], average='macro', zero_division=0) for label_idx in range(y_test.shape[1])])
    combined_f1_macro = np.mean([f1_score(y_test.iloc[:, label_idx], y_pred_combined[:, label_idx], average='macro', zero_division=0) for label_idx in range(y_test.shape[1])])
    combined_f1_micro = np.mean([f1_score(y_test.iloc[:, label_idx], y_pred_combined[:, label_idx], average='micro', zero_division=0) for label_idx in range(y_test.shape[1])])
    combined_f1_weighted = np.mean([f1_score(y_test.iloc[:, label_idx], y_pred_combined[:, label_idx], average='weighted', zero_division=0) for label_idx in range(y_test.shape[1])])

    # get the mean value of the metrics in arrays
    combined_metrics['precision'].append(combined_precision)
    combined_metrics['recall'].append(combined_recall)
    combined_metrics['f1_macro'].append(combined_f1_macro)
    combined_metrics['f1_micro'].append(combined_f1_micro)
    combined_metrics['f1_weighted'].append(combined_f1_weighted)

# print out the metric results
print("\nZusammenfassung der Metriken über alle Folds:")

# results for every label
for label in labels.columns:
    print(f"\nMetriken für Label {label}:")
    print(f"  Precision (Macro): {np.mean(label_metrics[label]['precision']):.4f} ± {np.std(label_metrics[label]['precision']):.4f}")
    print(f"  Recall (Macro): {np.mean(label_metrics[label]['recall']):.4f} ± {np.std(label_metrics[label]['recall']):.4f}")
    print(f"  F1-Score (Macro): {np.mean(label_metrics[label]['f1_macro']):.4f} ± {np.std(label_metrics[label]['f1_macro']):.4f}")
    print(f"  F1-Score (Micro): {np.mean(label_metrics[label]['f1_micro']):.4f} ± {np.std(label_metrics[label]['f1_micro']):.4f}")
    print(f"  F1-Score (Weighted): {np.mean(label_metrics[label]['f1_weighted']):.4f} ± {np.std(label_metrics[label]['f1_weighted']):.4f}")

# summarised results for the entire system
print("\nGesamte Modellmetriken über alle Folds:")
print(f"  Macro Precision: {np.mean(combined_metrics['precision']):.4f} ± {np.std(combined_metrics['precision']):.4f}")
print(f"  Macro Recall: {np.mean(combined_metrics['recall']):.4f} ± {np.std(combined_metrics['recall']):.4f}")
print(f"  Macro F1-Score: {np.mean(combined_metrics['f1_macro']):.4f} ± {np.std(combined_metrics['f1_macro']):.4f}")
print(f"  Micro F1-Score: {np.mean(combined_metrics['f1_micro']):.4f} ± {np.std(combined_metrics['f1_micro']):.4f}")
print(f"  Weighted F1-Score: {np.mean(combined_metrics['f1_weighted']):.4f} ± {np.std(combined_metrics['f1_weighted']):.4f}")

