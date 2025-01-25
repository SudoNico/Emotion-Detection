# this code preprocesses our data for future use in classification 
# we want to -> correct the spelling, remove punctuation, normalize all links, remove stopwords, tokenize our text and lemmatize it 

import pandas as pd 
import nltk as nl 
from nltk.tokenize import word_tokenize, wordpunct_tokenize 
from nltk.corpus import stopwords
import spacy 
import language_tool_python
import re 
import emoji
import csv


# Spelling correction using LanguageTool (needs to be installed beforehand) 
tool = language_tool_python.LanguageTool('de-DE')
def spellingCorrection(text, tool): 
    # checking the text for errors
    matches = tool.check(text)
    # correcting found errors 
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

# closes LanguageTool at the end
def closeLanguageTool(tool):
    tool.close()

# removes punctuation from the text
def removeText(text):
    without = re.sub(r"[^\w\s]", "", text) 
    return without

# removes emojicons and replaces them with the string "HierEmojiEntfernt" instead
def removeEmoji(text):
    return emoji.replace_emoji(text, replace= ' HierEmojiEntfernt ')

# Removal and replacement of URLs
def urlToLink(text):
    # using the reg expression for URLs to identify them
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # replacing found URLs with the string "Link"
    result = re.sub(url_pattern, " Link ", text)
    return result

# Removal of stop words from the tokenized text
def removeStopwords():
    # comparing the text to a list of german stop words
    nltk_stopwords = set(stopwords.words('german'))
    filtered_text = []
    # if the word isn't a stop word, it gets added to a non-stop-word-list
    for w in tokens:
        if w not in nltk_stopwords:
            filtered_text.append(w)
    return filtered_text

# Lemmatizing the text
# source: ChatGPT (only used for the following lemmatizing-part)
# initialises a spacy lemmatizing model and uses "de_core_news_sm", a pretrained model, which is specifically optimized for processing german language
lemmatizer = spacy.load("de_core_news_sm")
def lemmatizingText():
    lemmatized_text = []

    #processes each word using "de_core_news_sm"
    for w in filtered_text:
        doc = lemmatizer(w)
        # retrieves the lemma of the word and adds it to "lemmatized_text"
        lemmatized_text.append(doc[0].lemma_)
    return lemmatized_text

# writes the data to the file and joins the preprocessed words with a label for the binary classification
def writingFile(text, label):
    with open('preprocessed_results_2.txt', 'a', encoding='utf-8') as f:
        f.write(' '.join(text) + ', ' + label + '\n') 

# writes the data to the file and joins the preprocessed words with a multiple labels for the multilabel classification
def writingFile2(text, label1, label2, label3, label4, label5, label6, label7, label8, label9):
    with open('preprocessed_results.txt', 'a', encoding='utf-8') as f:
        f.write(' '.join(text) + ', ' + label1 + ', ' + label2 + ', ' + label3 + ', ' + label4 + ', ' + label5 + ', ' + label6 + ', ' + label7 + ', ' + label8 + ', ' + label9 + '\n')  # Join the words with spaces and write to file

# 1) preprocessing for binary classification, the preprocessing for multilabel classification is below
# since we used the preprocessing last for the multilabel system the code for the binary classification is commented out and therefore disabled
# if preprocessing is desired for a binary classification please enable this section and disable the section below

# loading the csv data file
#data = pd.read_csv('emo.csv', delimiter=';' ,encoding='utf-8')

# iterates through the file and preprocesses the text according to the methods above
# for index, row in data.iterrows(): 
#     s = row['description']
#     l = row['EMO']

# corrects misspellings
#     corrected_text = spellingCorrection(s, tool)

# removes URLs and substitutes them with the string "Link"
#     no_url = urlToLink(corrected_text)

# removes punctuation
#     no_punctuation = removeText(no_url)

# removes Emojis and substitutes them with string "HierEmojiEntfernt"
#     no_emojis = removeEmoji(no_punctuation)

# tokenization of the words
#     tokens = word_tokenize(no_emojis)
#     wordpunct_tokenize(no_emojis)

# removes stop words
#     filtered_text = removeStopwords()

# lemmatizes the text
#     lemmatized_text = lemmatizingText()
    
# transformes the text into all lowercase letters
#     lower_char = [w.lower() for w in lemmatized_text]

# writes preprocesses text into the file
#     writingFile(lower_char, l)
 

 
# 2) preprocessing for multi label classification

# reads the given file in a panda dataframe setting; file should either ems.csv or GoldStandardGerman_annotated.csv 
data2 = pd.read_csv('ems.csv', delimiter=';' ,encoding='utf-8')

# iterates through each line and preprocesses those according to the methods above
try:
       for index, row in data2.iterrows(): # our dataset has the following structure <description>, <anger>, <fear>, <surprise> ....
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
        
        # corrects misspellings
        corrected_text = spellingCorrection(s, tool)
    
        # removes URLs and substitutes them with the string "Link"
        no_url = urlToLink(corrected_text)
    
        # removes punctuation
        no_punctuation = removeText(no_url)
    
        # removes Emojis and substitutes them with string "HierEmojiEntfernt"
        no_emojis = removeEmoji(no_punctuation)

        # tokenization of the words
        tokens = word_tokenize(no_emojis)
        wordpunct_tokenize(no_emojis)

        # removes stop words
        filtered_text = removeStopwords()

        # lemmatizes the text
        lemmatized_text = lemmatizingText()
    
        # transformes the text into all lowercase letters
        lower_char = [w.lower() for w in lemmatized_text]

        # writes preprocessed data into a file using the template <text, emotion> with emotions being an element of [False,True]
        writingFile2(lower_char, a, f, su, sa, j, d, e, je, o)

finally: 
    closeLanguageTool(tool)
