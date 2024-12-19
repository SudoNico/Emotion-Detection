import pandas as pd
import numpy as np
import gensim
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from nlpaug.augmenter.word import SynonymAug

def strict_ml_smote(X, Y, target_samples, k_neighbors=5):
    X_resampled = list(X)
    Y_resampled = list(Y)

    for label_idx in range(Y.shape[1]):
        current_count = np.sum(Y[:, label_idx])
        if current_count >= target_samples:
            continue

        minority_class_indices = np.where(Y[:, label_idx] == 1)[0]
        minority_class = X[minority_class_indices]

        if len(minority_class) < k_neighbors:
            continue

        nn = NearestNeighbors(n_neighbors=k_neighbors).fit(minority_class)

        num_samples_to_generate = target_samples - current_count
        for _ in range(num_samples_to_generate):
            idx = np.random.choice(len(minority_class))
            neighbors = nn.kneighbors([minority_class[idx]], return_distance=False)[0]
            neighbor = minority_class[np.random.choice(neighbors[1:])]
            synthetic_sample = minority_class[idx] + np.random.rand() * (neighbor - minority_class[idx])

            X_resampled.append(synthetic_sample)
            new_label = np.zeros(Y.shape[1])
            new_label[label_idx] = 1
            Y_resampled.append(new_label)

    return np.array(X_resampled), np.array(Y_resampled)

# Lade die Daten
txt_path = "/Users/chanti/Desktop/5. Semester/Softwareprojekt/Code/Emotion-Detection/intermediate_results/preprocessed_results_multi.txt"
data = pd.read_csv(txt_path, sep=',')
pd.set_option('future.no_silent_downcasting', True)

columns = ['Tweet'] + [f'Label{i+1}' for i in range(9)]
data.columns = columns
data = data.dropna(subset=['Tweet'])
data = data[data['Tweet'] != '']

# Tokenisierung und Word2Vec
word2vec_model = Word2Vec(sentences=[word_tokenize(tweet.lower()) for tweet in data['Tweet']], vector_size=100, window=5, min_count=1, workers=4)

def tweet_vector(tweet, model):
    words = word_tokenize(tweet.lower())
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

X_word2vec = np.array([tweet_vector(tweet, word2vec_model) for tweet in data['Tweet']])
data.iloc[:, 1:] = data.iloc[:, 1:].replace({'true': 1, 'false': 0}).astype(int)
Y = data.iloc[:, 1:].values
Y = np.delete(Y, [6, 7], axis=1)

# TF-IDF und Kombination
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(data['Tweet']).toarray()
X_combined = np.hstack((X_word2vec, X_tfidf))
scaler = StandardScaler()
X_combined = scaler.fit_transform(X_combined)

X_train, X_test, y_train, y_test = train_test_split(X_combined, Y, test_size=0.2, random_state=42)

# =============================== Lösung 1: ADASYN ===============================
adasyn_oversampler = ADASYN(sampling_strategy='minority', random_state=42)
X_resampled_adasyn, y_resampled_adasyn = adasyn_oversampler.fit_resample(X_train, y_train)

# =============================== Lösung 2: SMOTE-ENN ============================
smoteenn_oversampler = SMOTEENN(random_state=42)
X_resampled_smoteenn, y_resampled_smoteenn = smoteenn_oversampler.fit_resample(X_train, y_train)

# ========================== Lösung 3: Balanced Random Forest ====================
balanced_rf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
balanced_rf.fit(X_resampled_adasyn, y_resampled_adasyn)

# ========================== Lösung 4: EasyEnsembleClassifier ====================
easy_ensemble_clf = EasyEnsembleClassifier(n_estimators=10, random_state=42)
easy_ensemble_clf.fit(X_resampled_smoteenn, y_resampled_smoteenn)

# =========================== Lösung 5: Data Augmentation ========================
aug = SynonymAug(aug_src='wordnet')
augmented_tweets = [aug.augment(tweet) for tweet in data['Tweet']]
augmented_data = data.copy()
augmented_data['Tweet'] = augmented_tweets
combined_data = pd.concat([data, augmented_data])

# ====================== Lösung 6: Voting Classifier mit Weights ==================
base_model_1 = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
base_model_2 = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
base_model_3 = GaussianNB()

voting_clf = MultiOutputClassifier(VotingClassifier(
    estimators=[('rf', base_model_1), ('svc', base_model_2), ('gnb', base_model_3)],
    voting='soft'
))

voting_clf.fit(X_resampled_adasyn, y_resampled_adasyn)

# ======================== Cross-Validation und Metriken =========================
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
precision_scores, recall_scores, f1_scores = [], [], []

for train_index, test_index in kf.split(X_combined):
    X_train, X_test = X_combined[train_index], X_combined[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    X_resampled, y_resampled = adasyn_oversampler.fit_resample(X_train, y_train)

    voting_clf.fit(X_resampled, y_resampled)
    y_pred = voting_clf.predict(X_test)

    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

print("\nCross-Validation Ergebnisse:")
print(f"Durchschnittliche Precision: {np.mean(precision_scores):.4f}")
print(f"Durchschnittliche Recall: {np.mean(recall_scores):.4f}")
print(f"Durchschnittliche F1-Score: {np.mean(f1_scores):.4f}")
