import pandas as panda
import json
from sklearn.feature_extraction.text import CountVectorizer


#opens the file in the read-only-mode with UTF 8
# jsdata is the variable name reffering to that file
with open('C:\\Users\\chiar\\Downloads\\Beispiel_Daten.json' , 'r', encoding='utf-8') as jsdata:
    data= json.load(jsdata)


#converting the data into a Dataframe to make further processing easier
emotions = panda.DataFrame(data, columns=['Kommentar-ID', 'Kommentar', 'Klasse'])
#to make sure every Tweet is available as a string
emotions['Kommentar']= emotions['Kommentar'].fillna('').apply(str)
#now creating a BOW without the automatic removal of stop words
bowVector = CountVectorizer(stop_words=None)
bowMatrix = bowVector.fit_transform(emotions['Kommentar'])


#also turning the BOW into a Dataframe for easier further processing
bow_df = panda.DataFrame(bowMatrix.toarray(), columns=bowVector.get_feature_names_out())
print (bow_df)