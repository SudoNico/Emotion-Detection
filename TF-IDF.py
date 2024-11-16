import pandas as panda
import numpy as npy
import nltk
from collections import Counter
import scipy.sparse as scispa
from numpy.linalg import norm

#source(not used 1:1):
#https://github.com/Wittline/tf-idf


class TFIDF(object):


    #konstructor that uses the corpus as input
    def konstructor(self, corpus):
        #original corpus
        self.corpus = corpus
        #saving a space for the normalized corpus
        self.norm_corpus = None


    #normalizing the document here
    def normalizecorpus (self,dok):
        #converting to all lowercase and removes empty space at the beginning and end
        dok= dok.lower().strip()
        #splitting the text into singular tokens (words)
        token = nltk.word_tokenize(dok)
        #return tokens as a string with a space in between
        return ' '.join(token)
    

    def adjusttext(self):
        #use function to normalize the corpus on all elements of the Array
        everywhere= npy.vectorize(self.normalizecorpus)
        #saving the normalized Version under the variable self.norm_corpus
        self.norm_corpus = everywhere(self.corpus)


    #calculates the term frequency for every individual word in every document using self.norm_corpus as 'input'
    def tf(self):
        #to split every document into its tokens
        wortsammlung= [doc.split() for doc in self.norm_corpus]
        #determines vocabulary
        worte= list(set([word for words in wortsammlung for word in words]))
        dictionary = {w: 0 for w in worte}
        #counts all words and returns them as a pandas Dataframe 
        tf = [] 
        for doc in self.norm_corpus:
            count= Counter(doc.split())
            countall= Counter(dictionary)
            count.update(countall)
            tf.append(count)
        return panda.DataFrame(tf)
    
    #uses Pandas Dataframe as 'input' to calculate document frequency
    def df(self, tf):
        wordcolumnname = list(tf.columns)
        #converts the term frequency into a sparse matrix
        df= npy.diff(scispa.csc_matrix(tf, copy=True).indptr)
        #smoothing do avoid mathematical issues in the future
        df = 1 + df
        return df
    

    
    def idf (self, df):
        #calculating the total number of documents and adds 1 to avoid issues that occure with division by 0
        numberdocs = 1+ len(self.norm_corpus)
        #calculates idf with following formular: idf = 1+ log (number of documents/document frequency) 
        idf= (1.0 + npy.log(float(numberdocs)/df ))
        #saves idf in a diagonal sparse matrix
        idfdiag= scispa.spdiags(idf, diags=0, m=len(df), n= len(df)).todense()
        return idf, idfdiag


    def tfidf(self, tf, idf):
        #transforms term frequency into a float64 Array 
        tf = npy.array(tf, dtype='float64')
        #calculates the tfidf
        tfidf = tf*idf
        #normalizes all tfidf-Vectors so that all are the lenth 1
        lzweinorm = norm(tfidf, axis=1)
        return (tfidf/lzweinorm[:, None])