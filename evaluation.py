from sklearn.metrics import precision_score, recall_score, f1_score

# exemplary predictions (y_pred) and actual labels (y_true)
y_true = [0, 0, 1, 2, 0, 2, 1, 1, 1, 2]
y_pred = [0, 2, 2, 2, 0, 0, 1, 0, 1, 2]

# calculating macro-precision and macro-recall  
macro_precision = precision_score(y_true, y_pred, average='macro')
macro_recall = recall_score(y_true, y_pred, average='macro')
macro_f1 = f1_score(y_true, y_pred, average='macro')

print(f"Makro-Precision: {macro_precision:.4f}")
print(f"Makro-Recall: {macro_recall:.4f}")
print(f"Makro-F1-Score: {macro_f1:.4f}")
# source: ChatGPT

