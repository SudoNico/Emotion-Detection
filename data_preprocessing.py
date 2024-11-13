import pandas as pd 
import nltk as nl 
from nltk.tokenize import word_tokenize, wordpunct_tokenize 
from nltk.corpus import stopwords
import spacy 
import language_tool_python
import re 
import json


# loading the JSON file
with open('file.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

results = []

# the text is in a field called 'Text' in the JSON
for entry in data: 
    s = entry['Text'] 

    # spelling correction, source: ChatGPT
    tool = language_tool_python.LanguageTool('de-DE')
    # finding errors in the text
    matches = tool.check(s)
    # correcting the errors 
    corrected_text = language_tool_python.utils.correct(s, matches)


    # url to [link], source: ChatGPT
    def url_to_link(text):
     # reg expression for URL
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
     # replacing the URL with "[Link]"
        result = re.sub(url_pattern, 'Link', text)
        return result

    no_url = url_to_link(corrected_text)

    # tokenizing the data into seperate words 
    tokens = word_tokenize(no_url)
    wordpunct_tokenize(no_url)

    # removing stop words from the tokenized text
    nltk_stopwords = set(stopwords.words('german'))
    filtered_text = []
 
    for w in tokens:
        if w not in nltk_stopwords:
            filtered_text.append(w)
        

    # lemmatizing the text, source: ChatGPT 
    lemmatizer = spacy.load("de_core_news_sm")
    lemmatized_text = []

    for w in filtered_text:
        doc = lemmatizer(w)
        lemmatized_text.append(doc[0].lemma_)
    
    # all lower characters 
    lower_char = [w.lower() for w in lemmatized_text]
    
    results.append(lower_char)
        
 
# saving the results in a new file    
with open('processed_results.txt', 'w', encoding='utf-8') as f:
    for result in results:
        f.write(' '.join(result) + '\n')  # Join the words with spaces and write to file



