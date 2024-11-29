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
    without = re.sub(r"[^\w\s]", "", text) # ? und ! auch entfernt, da ansonsten "--" eingefügt wird 
    return without

# removing emojicons from the text 
def removeEmoji(text):
    return emoji.replace_emoji(text, replace= 'HierEmojiEntfernt')

# url to "Link"
def urlToLink(text):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'   # reg expression for URL
    # replacing the URL with "Link"
    result = re.sub(url_pattern, "Link", text)
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

def writingFile(text):
    with open('processed_results_2.txt', 'a', encoding='utf-8') as f:
        f.write(' '.join(text) + '\n')  # Join the words with spaces and write to file

# loading the csv file
data = pd.read_csv('/Users/chanti/Desktop/5. Semester/Softwareprojekt/Code/Emotion-Detection/Annotierte Tweets/emo2.csv', delimiter=';' ,encoding='utf-8')
try: 
    for index, row in data.iterrows(): 
        s = row['description']

        corrected_text = spellingCorrection(s, tool)
    
        no_url = urlToLink(corrected_text)
    
        no_punctuation = removeText(no_url)
    
        no_emojis = removeEmoji(no_punctuation)

        # tokenizing the data into seperate words 
        tokens = word_tokenize(no_emojis)
        wordpunct_tokenize(no_emojis)

        filtered_text = removeStopwords()

        lemmatized_text = lemmatizingText()
    
        # all lower characters 
        lower_char = [w.lower() for w in lemmatized_text]
    
        writingFile(lower_char)
        
finally: 
    closeLanguageTool(tool)
 


