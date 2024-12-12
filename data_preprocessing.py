import pandas as pd 
import nltk as nl 
from nltk.tokenize import word_tokenize, wordpunct_tokenize 
from nltk.corpus import stopwords
import spacy 
import language_tool_python
import re 
import emoji
import csv


# spelling correction
tool = language_tool_python.LanguageTool('de-DE')
def spellingCorrection(text, tool): 
    # finding errors in the text
    matches = tool.check(text)
    # correcting the errors 
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

# closing LanguageTool at the end
def closeLanguageTool(tool):
    tool.close()

# removing punctuation from the text 
def removeText(text):
    without = re.sub(r"[^\w\s]", "", text) 
    return without

# removing emojicons from the text 
def removeEmoji(text):
    return emoji.replace_emoji(text, replace= ' HierEmojiEntfernt ')

# url to "Link"
def urlToLink(text):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'   # reg expression for URL
    # replacing the URL with "Link"
    result = re.sub(url_pattern, " Link ", text)
    return result

# removing stop words from the tokenized text
def removeStopwords():
    nltk_stopwords = set(stopwords.words('german'))
    filtered_text = []
 
    for w in tokens:
        if w not in nltk_stopwords:
            filtered_text.append(w)
    return filtered_text

# lemmatizing the text, source: ChatGPT 
lemmatizer = spacy.load("de_core_news_sm")
def lemmatizingText():
    lemmatized_text = []

    for w in filtered_text:
        doc = lemmatizer(w)
        lemmatized_text.append(doc[0].lemma_)
    return lemmatized_text

def writingFile(text, label):
    with open('/Users/chanti/Desktop/5. Semester/Softwareprojekt/Code/Emotion-Detection/intermediate_results/preprocessed_results_2.txt', 'a', encoding='utf-8') as f:
        f.write(' '.join(text) + ', ' + label + '\n')  # Join the words with spaces and write to file

def writingFile2(text, label1, label2, label3, label4, label5, label6, label7, label8, label9):
    with open('/Users/chanti/Desktop/5. Semester/Softwareprojekt/Code/Emotion-Detection/intermediate_results/preprocessed_results.txt', 'a', encoding='utf-8') as f:
        f.write(' '.join(text) + ', ' + label1 + ', ' + label2 + ', ' + label3 + ', ' + label4 + ', ' + label5 + ', ' + label6 + ', ' + label7 + ', ' + label8 + ', ' + label9 + '\n')  # Join the words with spaces and write to file

# 1) preprocessing for binary classification

# loading the csv file
#data = pd.read_csv('/Users/chanti/Desktop/5. Semester/Softwareprojekt/Code/Emotion-Detection/Annotierte Tweets/emo.csv', delimiter=';' ,encoding='utf-8')
    
# for index, row in data.iterrows(): 
#     s = row['description']
#     l = row['EMO']
#     corrected_text = spellingCorrection(s, tool)
    
#     no_url = urlToLink(corrected_text)
    
#     no_punctuation = removeText(no_url)
    
#     no_emojis = removeEmoji(no_punctuation)

#     # tokenizing the data into seperate words 
#     tokens = word_tokenize(no_emojis)
#     wordpunct_tokenize(no_emojis)

#     filtered_text = removeStopwords()

#     lemmatized_text = lemmatizingText()
    
#     # all lower characters 
#     lower_char = [w.lower() for w in lemmatized_text]
    
#     writingFile(lower_char, l)
 
# 2) preprocessing for multi label classification

data2 = pd.read_csv('/Users/chanti/Desktop/5. Semester/Softwareprojekt/Code/Emotion-Detection/Annotierte Tweets/ems.csv', delimiter=';' ,encoding='utf-8')

try:
       for index, row in data2.iterrows(): 
        s = str(row['description'])
        a = str(row['anger'])
        f = str(row['fear'])
        su = str(row['surprise'])
        sa = str(row['sadness'])
        j = str(row['joy'])
        d = str(row['disgust'])
        e = str(row['envy'])
        je = str(row['jealousy'])
        o = str(row['other'])
        
        corrected_text = spellingCorrection(s, tool)
    
        no_url = urlToLink(corrected_text)
    
        no_punctuation = removeText(no_url)
    
        no_emojis = removeEmoji(no_punctuation)

     
        tokens = word_tokenize(no_emojis)
        wordpunct_tokenize(no_emojis)

        filtered_text = removeStopwords()

        lemmatized_text = lemmatizingText()
    
       
        lower_char = [w.lower() for w in lemmatized_text]
    
        writingFile2(lower_char, a, f, su, sa, j, d, e, je, o)

finally: 
    closeLanguageTool(tool)