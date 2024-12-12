# code for deciding wheter or not a tweet contains emotion 
# naive bayes classifier using bow as a feature 

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


# loading the tweets and extracting the labels and data 
tweets = []
labels = []

with open("/Users/chanti/Desktop/5. Semester/Softwareprojekt/Code/Emotion-Detection/intermediate_results/preprocessed_results_2.txt", "r", encoding="utf-8") as file:
    for line in file: 
        try:
            tweet, label = line.rsplit(",", 1)   # splitting the preprocessed tweets because the labels are after the last comma 
            tweets.append(tweet.strip())
            labels.append(1 if label.strip().lower() == "yes" else 0)  # changing the labels from yes/no to 1/0 -> yes = 1, no = 0
        except ValueError:
            print("Fehlerhafte Zeile:", line)

# splitting the data in train (80%) and test (20%) data 
X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.2)

# generating a bow-representation as our feature 
vectorizer = CountVectorizer()  
X = vectorizer.fit_transform(tweets) # vectorizing all tweets 
Y = labels

# since we have 5637 tweets with emotions and 3460 tweets with no emotions we use SMOTE to balance our data set 
smote = SMOTE(random_state=42) # random_state = 42 -> same output in every run 

# resampling the data 
X_resampled, y_resampled = smote.fit_resample(X, Y)

# training a multinominal naive bayes as our baseline system 
nb_classifier = MultinomialNB()

scoring = {
    'precision_macro': make_scorer(precision_score, average='macro'),
    'recall_macro': make_scorer(recall_score, average='macro'),
    'f1_macro': make_scorer(f1_score, average='macro')
} # we want to evaluate our model using macro precision, macro recall and macro f1

cv_results = cross_validate(nb_classifier, X_resampled, y_resampled, cv=5, scoring=scoring) 
y_pred = cross_val_predict(nb_classifier, X_resampled, y_resampled, cv=5)
# we split our train data into 5 folds 
# in each iteration we train the model on 4 out of 5 folds (k-1) and use the one fold left for evaluation  

macro_precision = cv_results['test_precision_macro'].mean() # average macro precision for this model computed from the 5 precisions from each fold 
macro_recall = cv_results['test_recall_macro'].mean() # average macro recall for this model computed from the 5 recall from each fold 
macro_f1 = cv_results['test_f1_macro'].mean() # average macro f1 for this model computed from the 5 f1 from each fold 

print(macro_precision)
print(macro_recall)
print(macro_f1)
print(Counter(y_resampled))
print(confusion_matrix(y_resampled, y_pred))
