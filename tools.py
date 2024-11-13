# for text processing we can use nltk 

import nltk # you first have to install nltk by running "pip install nltk" in your terminal 
nltk.download('all') # identifier = all in your terminal 

# for working with data sets; has functions for analyzing, cleaning, exploring, and manipulating data
import pandas 
# so first we process the data/tweets using nltk and then we use panda to create a data frame and work on that 

# we can use scikit-learn form classification, clustering and feauture extraction 
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

# naive bayes implementation
from bayesian import Bayes
 


#Collection of lightweight text tools for tokenization, word normalization, word segmentation (for splitting hashtags) and spell correction
#https://github.com/cbaziotis/ekphrasis
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.classes.segmenter import Segmenter
from ekphrasis.classes.spellcorrect import SpellCorrector

#spaCy comes with pretrained pipelines and currently supports tokenization and training for 70+ languages
import spacy
#It features state-of-the-art speed and neural network models for tagging, parsing, named entity recognition, text classification and more


