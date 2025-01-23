import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# prepares the NLTK download of ressources
nltk.download('punkt')

# loading the data and initialises that the first column is the text, the others are labels 
data = pd.read_csv('/Users/chanti/Desktop/5. Semester/Softwareprojekt/Code/Emotion-Detection/intermediate_results/preprocessed_results_ems.txt', sep=',', header=None)
data.columns = ['Tweet'] + [f'Label{i+1}' for i in range(9)] 

#doing the same for our own additional annotated wordlist ("GoldstandardGerman)"
additional_data = pd.read_csv('/Users/chanti/Desktop/5. Semester/Softwareprojekt/Code/Emotion-Detection/intermediate_results/preprocessed_results_GoldStandardGerman.txt', sep=',', header=None)
additional_data.columns = ['Tweet'] + [f'Label{i+1}' for i in range(9)]  

# extracting tweets and labels for the additional wordlist
tweets = data['Tweet']
labels = data.iloc[:, 1:]

# extracting tweets and labels for the additional wordlist
additional_tweets = additional_data['Tweet']
additional_labels = additional_data.iloc[:, 1:]

# removal of spaces and converting them into integer-values (1 is ture, 0 false)
labels = labels.map(lambda x: int(str(x).strip().lower() in ['1', 'true']))
# removal of non-number-values and substituting them with empty strings
tweets = tweets.fillna('') 
# resolving all values left into strings 
tweets = tweets.astype(str)  

# removal of spaces and converting them into integer-values (1 is ture, 0 false) in the additional wordlist
additional_labels = additional_labels.map(lambda x: int(str(x).strip().lower() in ['1', 'true']))
# # removal of non-number-values and substituting them with empty strings
additional_tweets = additional_tweets.fillna('') 
# resolving all values left into strings 
additional_tweets = additional_tweets.astype(str)  

# calculating TF-IDF by initialising a TF-IDF Vector that accepts gets tokenized directly
tfidf_vectorizer = TfidfVectorizer(analyzer=lambda x: x, token_pattern=None)
# calculating the TF-IDF matrix and retrieving the vocabulary
tfidf_matrix = tfidf_vectorizer.fit_transform(tweets)
tfidf_vocab = tfidf_vectorizer.vocabulary_

# training and initialising Word2Vec modell
word2vec_model = Word2Vec(sentences=tweets, vector_size=100, window=5, min_count=1, workers=4)

# calculating the TF-IDF weighted Word2Vec-Vectors
def get_tfidf_weighted_word2vec(tokens, model, tfidf_vocab, tfidf_matrix, index, vector_size=100):
    #initialising zero-vector and matching the TF-IDF-values to the current tweet
    vector = np.zeros(vector_size)
    tfidf_values = tfidf_matrix[index]
    
    # counting valid tokens
    valid_words = 0
    for word in tokens:
        # checking if the word is in both the Word2Vec modell and TF-IDF vocabulary
        if word in model.wv and word in tfidf_vocab:
            tfidf_weight = tfidf_values[0, tfidf_vocab[word]]
           # adding the weighted vector
            vector += model.wv[word] * tfidf_weight
            # adding the TF-IDF-value
            valid_words += tfidf_weight
    
    if valid_words > 0:
        #normalising the vector to the sum of all weights
        vector /= valid_words 
    return vector

#  calculating the TF-IDF-weighted Word2Vec-Features for all tweets
X_combined = np.array([
    get_tfidf_weighted_word2vec(tokens, word2vec_model, tfidf_vocab, tfidf_matrix, i)
    for i, tokens in enumerate(tweets)
])

# calculating TF-IDF for the additional wordlist
additional_tfidf_matrix = tfidf_vectorizer.fit_transform(additional_tweets)
tfidf_vocab = tfidf_vectorizer.vocabulary_
# training the Word2Vec-Modell with the additional wordlist
additional_word2vec_model = Word2Vec(sentences=additional_tweets, vector_size=100, window=5, min_count=1, workers=4)
additional_X_combined = np.array([
    get_tfidf_weighted_word2vec(tokens, additional_word2vec_model, tfidf_vocab, additional_tfidf_matrix, i)
    for i, tokens in enumerate(additional_tweets)
])

# Initialising KFold with a 10x K-Fold-Cross-Validation with a random permutation and a fixed seed
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# creating dictionaries to save the results for each label
macro_f1_scores_dict = {label: [] for label in labels.columns}
micro_f1_scores_dict = {label: [] for label in labels.columns}
weighted_f1_scores_dict = {label: [] for label in labels.columns}
macro_recall_dict = {label: [] for label in labels.columns}
macro_precision_dict = {label: [] for label in labels.columns}

# matching the order of the emotion-sequences to the columns (Label1, Label2, ..., Label9)
emotions = ['anger', 'fear', 'surprise', 'sadness', 'joy', 'disgust', 'envy', 'jealousy', 'other']

# Cross-Validation-Loop
for fold, (train_index, test_index) in enumerate(kf.split(X_combined)):
    # printing out the current fold-number
    print(f"Fold {fold + 1}")
    
    # splitting data into train- and testdata
    X_train, X_test = X_combined[train_index], X_combined[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
    X_train = np.vstack([X_train, additional_X_combined])
    y_train = pd.concat([y_train, additional_labels], ignore_index=True)
    
    # initialising an array for predicting
    y_pred_combined = np.zeros_like(y_test)
    
    # Loop over all emotions (labels)
    for i, label in enumerate(labels.columns):
        # taking one specific emotion
        emotion_name = emotions[i] 
        print(f"Training für Label: {emotion_name}")
        
        
        # initialising a stacking classifier containing a RandomForrest-, SVC(linear kernel)- and naive bayes-component
        stacking_clf = StackingClassifier(estimators=[
                ('rf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
                ('svc', SVC(kernel='linear', probability=True)),
                ('nb', GaussianNB()),
            ], 
            final_estimator=RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
            passthrough=True,
            # Cross-Validation here with 5 folds
            cv=5)
        
        # training the modell
        stacking_clf.fit(X_train, y_train[label])
        # making predictions based on the testdata
        y_pred = stacking_clf.predict(X_test)
        # saving the predictions in a combined prediction-matrix
        y_pred_combined[:, labels.columns.get_loc(label)] = y_pred
    
    # calculating f1-score, recall and precision for each label
    for i in range(y_test.shape[1]):
        f1_macro = f1_score(y_test.iloc[:, i], y_pred_combined[:, i], average='macro', zero_division=0)
        f1_micro = f1_score(y_test.iloc[:, i], y_pred_combined[:, i], average='micro', zero_division=0)
        f1_weighted = f1_score(y_test.iloc[:, i], y_pred_combined[:, i], average='weighted', zero_division=0)
        precision_macro = precision_score(y_test.iloc[:, i], y_pred_combined[:, i], average='macro', zero_division=0)
        recall_macro = recall_score(y_test.iloc[:, i], y_pred_combined[:, i], average='macro', zero_division=0)
        
        # saving the results in the corresponding dictionaries
        label = labels.columns[i]
        
        macro_f1_scores_dict[label].append(f1_macro)
        micro_f1_scores_dict[label].append(f1_micro)
        weighted_f1_scores_dict[label].append(f1_weighted)
        macro_recall_dict[label].append(recall_macro)
        macro_precision_dict[label].append(precision_macro)

# summarizing the results of all folds
print("\nGesamte Ergebnisse über alle Folds (durchschnittlich):")
macro_f1_system = np.mean([np.mean(scores) for scores in macro_f1_scores_dict.values()])
micro_f1_system = np.mean([np.mean(scores) for scores in micro_f1_scores_dict.values()])
weighted_f1_system = np.mean([np.mean(scores) for scores in weighted_f1_scores_dict.values()])
macro_recall_system = np.mean([np.mean(scores) for scores in macro_recall_dict.values()])
macro_precision_system = np.mean([np.mean(scores) for scores in macro_precision_dict.values()])

# printing all calculated metrics 
print(f"Macro F1 (System): {macro_f1_system:.2f} ± {np.std(list(macro_f1_scores_dict.values())):.2f}")
print(f"Micro F1 (System): {micro_f1_system:.2f} ± {np.std(list(micro_f1_scores_dict.values())):.2f}")
print(f"Weighted F1 (System): {weighted_f1_system:.2f} ± {np.std(list(weighted_f1_scores_dict.values())):.2f}")
print(f"Macro Recall (System): {macro_recall_system:.2f} ± {np.std(list(macro_recall_dict.values())):.2f}")
print(f"Macro Precision (System): {macro_precision_system:.2f} ± {np.std(list(macro_precision_dict.values())):.2f}")

# printing the average results for each label
print("\nDurchschnittliche Ergebnisse pro Label über alle Folds:")
for label in labels.columns:
    print(f"\nLabel: {label}")
    print(f"Macro F1: {np.mean(macro_f1_scores_dict[label]):.2f} ± {np.std(macro_f1_scores_dict[label]):.2f}")
    print(f"Micro F1: {np.mean(micro_f1_scores_dict[label]):.2f} ± {np.std(micro_f1_scores_dict[label]):.2f}")
    print(f"Weighted F1: {np.mean(weighted_f1_scores_dict[label]):.2f} ± {np.std(weighted_f1_scores_dict[label]):.2f}")
    print(f"Macro Recall: {np.mean(macro_recall_dict[label]):.2f} ± {np.std(macro_recall_dict[label]):.2f}")
    print(f"Macro Precision: {np.mean(macro_precision_dict[label]):.2f} ± {np.std(macro_precision_dict[label]):.2f}")
