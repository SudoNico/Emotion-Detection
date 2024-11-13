import pandas as pd 
import nltk as nl 
from nltk.tokenize import word_tokenize, wordpunct_tokenize 
from nltk.corpus import stopwords
import spacy 
import language_tool_python
import re 


# sample text
s = 'Er beschlos, den Rechtssteit mit dem Kloster einzstellen und seine Ansprüche aufzugeben. Holzfäller- und Fischereirechte auf einmal. Er war dazu umso bereitwilliger, weil die Rechte war viel weniger wertvoll geworden, und er hatte tatsächlich eine vage Vorstellung davon, wo der Wald und der Fluss in Frage kamen. Schau dir diese Webseite an: https://www.example.com und diese auch: http://test.com'

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
    result = re.sub(url_pattern, '[Link]', text)
    return result

no_url = url_to_link(corrected_text)

# tokenizing the data into seperate words 
tokens = word_tokenize(no_url)
wordpunct_tokenize(no_url)

# removing stop words from the tokenized text
stopwords = set(stopwords.words('german'))
filtered_text = []
 
for w in tokens:
    if w not in stopwords:
        filtered_text.append(w)
        

# lemmatizing the text, source: ChatGPT 
lemmatizer = spacy.load("de_core_news_sm")
lemmatized_text = []

for w in filtered_text:
    doc = lemmatizer(w)
    lemmatized_text.append(doc[0].lemma_)
    
# all lower characters 
lower_char = []

for w in lemmatized_text:
    lower_char.append(w.lower())


#print(s)
#print()
#print(corrected_text)
#print()
#print(no_url)
#print()
#print(tokens)
#print()
#print(filtered_text)
#print()
#print(lemmatized_text)
#print()
#print(lower_char)
