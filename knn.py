from sklearn.neighbors import KNeighborsClassifier

#seperate knn for every Label

bayes = [0.1]
#bayes --> values of the Tweet to be classified
values =  [[0], [1], [2], [3]]
#values --> values of the trainingsset
label = [0, 0, 1, 1]
#label --> does the value of the trainingsset has the Label

#create Classifier with 3 neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(values,label)

#prediction
print(knn.predict(bayes))
