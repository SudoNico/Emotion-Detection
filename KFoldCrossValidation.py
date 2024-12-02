import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

#source: https://github.com/vaasha/Machine-leaning-in-examples/blob/master/sklearn/cross-validation/Cross%20Validation.ipynb and ChatGPT 

# df is Dateframe used as imput
# targetcolumn is meant for the column with the labeled emotion
# nsplits is for the ammount of folds
# model which is supposed to be validated (easily exchangeable ex. Random Forrest, Logistic Regression, SVM, ...)
def kfoldcross (df, targetcolumn, nsplits=5, model= None):
    #if no Model is specified use a Random Forrest
    if model is None:
        model = RandomForestClassifier()

    #All Data from df is splittet into features (X) and Targetvariables (y), here we're using all columns exept the targetcolumn as Fetures
    X= df.drop(columns= [targetcolumn])
    #defining the targetcolumn as y
    y= df[targetcolumn]


    #defining the K-Fold Objekt with n Folds, a random shuffle (mix) and a specific state to reproduce the numbers
    kf = KFold(n_splits=nsplits, shuffle=True, random_state=58)
    #using a dictionary to save the different metrics for each fold
    prfmetrics = {'precision': [], 'recall':[], 'f1':[]}

    #now iterating through folds and splitting the data in trainings and test pieces
    for fold, (training, testing) in enumerate (kf.split(X)):
        Xtrain, Xtest = X.iloc[training], X.iloc[testing]
        ytrain, ytest = y.iloc[training], y.iloc[testing]

        #trains the chosen Model (or Random Forrest as default) with the testdata and makes predictions
        model.fit(Xtrain, ytrain)
        yprediction= model.predict(Xtest)#

        #calculating the different metrics for the fold
        precision= precision_score(ytest, yprediction, average='macro' )
        recall = recall_score(ytest, yprediction, average='macro')
        f1= f1_score(ytest, yprediction, average='macro')

        #defining the metrics to be safed in the metrics dictionary
        prfmetrics['precision'].append(precision)
        prfmetrics['recall'].append(recall)
        prfmetrics['f1'].append(f1)

        #since first fold is at index 0 we start at fold+1, calculate the precision, recall and F1-Score for the current fold as a float number with 4 decimals (.4f)
        print (f"Fold {fold+1}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    #now instead of looking at one fold in particular we view the average over all folds using a sum-function, again as a float number with 4 decimals (.4f)
    print("averages on all folds:")
    print(f"Macro-Precision: {sum(prfmetrics['precision']) / nsplits:.4f}")
    print(f"Macro-Recall: {sum(prfmetrics['recall']) / nsplits:.4f}")
    print(f"Macro-F1-Score: {sum(prfmetrics['f1']) / nsplits:.4f}")
    
    return prfmetrics