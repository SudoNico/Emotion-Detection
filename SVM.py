import numpy as npy
import pandas as panda
import json
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


#muss sehr sicher noch angepasst werden, hier ist nur super basic svm
#unter https://gist.github.com/pb111/ca4680d8960c46aeb1b824a93a079fa7 ist eine 
#super Anleitung die ist aber viel zu komplex als das, was wir machen
#da spielt aber auch ROC, k-fold Cross Validation, Konfusionsmatrix und alle moeglichen Kernel-Arten eine Rolle


def datenladen(jsondata):
    #open in read-only-mode
    with open (jsondata, 'r') as file:
        data =json.load(file)
        #extracting features and labeled emotions and return them in two individual lists
        feature = [entry["Kommentar"]for entry in data]
        label = [entry["Klasse"]for entry in data]
        return feature, label

    
jsondata = "PFAD/ZUR/JSON.json"
X, y = datenladen(jsondata)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, testsize= 0.3, random_state=22)

svmModel= SVC(kernel= 'rbf', C=1.0, gamma='scale')
svmModel.fit(Xtrain, ytrain)

predict= svmModel.predict(Xtest)

print("Klassifikationsgenauigkeit:", accuracy_score(ytest, predict))
print("\nKlassifikationsbericht:", classification_report(ytest, predict))


