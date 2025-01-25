# This script trains multiple machine learning models to classify tweets as containing emotion or not. 
# It includes preprocessing, feature extraction (BoW and TF-IDF), SMOTE for class balancing, and evaluation via cross-validation.
# source: ChatGPT and online documentation

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from collections import Counter
from sklearn.model_selection import cross_val_predict
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

# defining two variables in which we will load our data -> one for the tweets and one for the labels (= emotion yes/no) 
tweets = []
labels = []

# load the preprocessed tweets and their corresponding labels (emotion yes/no) from the file
# file should have one tweet per line in the format: "tweet_text,label"
with open("Path to preprocessed_results_emo", "r", encoding="utf-8") as file:
    for line in file:
        try:
            tweet, label = line.rsplit(",", 1)  # Splitting the preprocessed tweets because the labels are after the last comma
            tweets.append(tweet.strip())
            labels.append(1 if label.strip().lower() == "yes" else 0)  # Changing the labels from yes/no to 1/0 -> yes = 1, no = 0
        except ValueError: # catch ValueError in case a line does not follow the expected format 
            print("Fehlerhafte Zeile:", line) 

# generating a BoW representation which represents the frequency of each word in the dataset as our feature for the first two models using CountVectorizer 
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(tweets)  # Vectorizing all tweets
Y = labels 

# Since we have 5637 tweets with emotions and 3460 tweets with no emotions, we use SMOTE to balance the dataset by generating synthetic samples -> prevents the model from being biased toward the majority class
smote = SMOTE(random_state=42)  # random_state = 42 -> same output in every run

# Resampling the data
X_resampled, y_resampled = smote.fit_resample(X, Y)

# using a multinomial naive bayes as our baseline system
nb_classifier = MultinomialNB()

# using CatBoost as our first improved model 
catboost_classifier = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=False)  
# explanation: iterations=100 -> number of iterations for training, we chose this number because we don't have a big dataset
# depth=6 -> max depth of a Decision Tree, we chose this number to prevent Overfitting but still recognize patterns in our dataset
# learning_rate=0.1 -> weight for new trees, we chose this number as a compromise between fast learning/training and preventing overfitting 

# base models for VotingClassifier 
base_model_1 = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42) # Random Forest as our first base model -> 100 trees, balancing the classes (in case there is still some imbalance), defining a random state so we get the same result every time (any number works)
base_model_2 = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42) # SVM as our second base model -> using a linear kernel which is usefull for textual data (esp. after a BoW representation), activating the calculation of probabilities which we need as input for the VotingClassifier, balancing the classes, defining a random state (any number works) 
base_model_3 = MultinomialNB()

# using VotingClassifier with the base models as our final improved system 
voting_clf = VotingClassifier(
    estimators=[('rf', base_model_1), ('svc', base_model_2), ('mnb', base_model_3)], # our base models are the input 
    voting='soft' # soft voting combines the probabilites of our base models and chooses the class with the highest weighted probabilities 
)

# Defining scoring for cross-validation
scoring = {
    'precision_macro': make_scorer(precision_score, average='macro'),
    'recall_macro': make_scorer(recall_score, average='macro'),
    'f1_macro': make_scorer(f1_score, average='macro')
}  # We want to evaluate our model using macro precision, macro recall, and macro f1

# Cross-Validation for Naive Bayes with 10 Folds and our scoring system 
cv_results_nb = cross_validate(nb_classifier, X_resampled, y_resampled, cv=10, scoring=scoring)
y_pred_nb = cross_val_predict(nb_classifier, X_resampled, y_resampled, cv=10)

# Cross-Validation for CatBoost with 10 Folds and our scoring system 
cv_results_catboost = cross_validate(catboost_classifier, X_resampled, y_resampled, cv=10, scoring=scoring)  
y_pred_catboost = cross_val_predict(catboost_classifier, X_resampled, y_resampled, cv=10)  

# generating a TF-IDF representation as our feature for the improved system using a VotingClassifier 
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tweets)  # vectorizing all tweets
Y = labels # saving our labels (=1/0) as Y 

# since we have 5637 tweets with emotions and 3460 tweets with no emotions, we use SMOTE to balance our dataset
smote = SMOTE(random_state=42)  # random_state = 42 -> same output in every run

# resampling the data using SMOTE
X_resampled, y_resampled = smote.fit_resample(X, Y)

# Cross-Validation for VotingClassifier with 10 Folds and our scoring system 
cv_results_vc = cross_validate(voting_clf, X_resampled, y_resampled, cv=10, scoring=scoring)
y_pred_vc = cross_val_predict(voting_clf, X_resampled, y_resampled, cv=10)

# printing the evaluation results for each model to compare their performance and determine which model works best for this classification task
# results for Naive Bayes 
print("Naive Bayes Ergebnisse:")
print(f"Precision: {cv_results_nb['test_precision_macro'].mean():.4f}") 
print(f"Recall: {cv_results_nb['test_recall_macro'].mean():.4f}")
print(f"F1 Score: {cv_results_nb['test_f1_macro'].mean():.4f}")

# results for CatBoost
print("CatBoost Ergebnisse:")
print(f"Precision: {cv_results_catboost['test_precision_macro'].mean():.4f}")
print(f"Recall: {cv_results_catboost['test_recall_macro'].mean():.4f}")
print(f"F1 Score: {cv_results_catboost['test_f1_macro'].mean():.4f}")

# results for VotingClassifier
print("VotingClassifier Ergebnisse:")
print(f"Precision: {cv_results_vc['test_precision_macro'].mean():.4f}")
print(f"Recall: {cv_results_vc['test_recall_macro'].mean():.4f}")
print(f"F1 Score: {cv_results_vc['test_f1_macro'].mean():.4f}")
