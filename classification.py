from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
import pandas as pd
import joblib


# defining the baseline systems 
basemodels = [
    ('nb', Pipeline([
        #('bow', BagOfWords()),
        ('tfidf', TfidfVectorizer()), 
        ('nb', MultinomialNB())
    ])),
    ('knn', Pipeline([
        ('tfidf', TfidfVectorizer()), 
        ('knn',RandomForestClassifier(n_estimators=100))
    ])),
    ('svc', Pipeline([
        ('tfidf', TfidfVectorizer()), 
        ('svc', SVC(probability=True))
    ])),
]

# StackingClassifier using RandomForest as a Meta-Model
stacked_clf = StackingClassifier(
    estimators=basemodels,
    final_estimator=RandomForestClassifier(n_estimators=100),
    cv=5
)

# StackingClassifier using LogisticRegression as a Meta-Model
stacked_clf_2 = StackingClassifier(
    estimators=basemodels,
    final_estimator= LogisticRegression()
)

# VotingClassifier 
voting_clf = VotingClassifier(
    estimators=basemodels,
    voting="soft"
)

# defining the MultiOutputClassifiers
multi_out_sclf1 = MultiOutputClassifier(stacked_clf)
multi_out_sclf2 = MultiOutputClassifier(stacked_clf_2)
multi_out_vclf = MultiOutputClassifier(voting_clf)

# saving the models 
joblib.dump(multi_out_sclf1, 'stacking_classifier_1.pkl')
joblib.dump(multi_out_sclf2, 'stacking_classifier_2.pkl')
joblib.dump(multi_out_vclf, 'voting_classifier.pkl')


