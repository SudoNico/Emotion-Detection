# implementing word2vec as a feature 
from gensim.models import Word2Vec

# loading the preprocessed data 
def load_tokenized_txt_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        # creating list from the sentences 
        sentences = [line.strip().split() for line in file if line.strip()]
    return sentences

tokenized_tweets = load_tokenized_txt_file('/Users/chanti/Desktop/5. Semester/Softwareprojekt/Code/Emotion-Detection/processed_results.txt')

model = Word2Vec(
    sentences=tokenized_tweets, # our dataset
    vector_size=150,            # dimension of the vectors
    window=3,                   # how many words next to the word are considered 
    min_count=1,                # minimal wordcount
    sg=1,                       # 1=skip-gram, 0=cbow
    epochs=20,                  # number of runs in the training
    workers=4,                  # number of threads
    sample=1e-4                 # subsampling common words
)

# training the modell 
model.train(tokenized_tweets, total_examples=len(tokenized_tweets), epochs=model.epochs)

# saving the model for future use 
model.save("word2vec_tweets.model")
