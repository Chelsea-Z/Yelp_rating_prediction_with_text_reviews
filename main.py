
import json 
import pandas as pd 
import numpy as np 
import re 
import string
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt 
from sklearn.svm import libsvm 

def read_train(url):
    data = []
    with open(url) as f:
        for line in f:
            data.append(json.loads(line))
    
    # extract star rating and text review from data
    stars_data=[]
    text_data=[]
    for review in data:
        #put them in the star&text list accordingly
        stars_data.append(review['stars'])
        text_data.append(review['text'])

    return stars_data, text_data

def read_dev(url):
    data = []
    with open(url) as f:
        for line in f:
            data.append(json.loads(line))
    
    # extract text review from data
    text_data=[]
    for review in data:
        text_data.append(review['text'])

    return text_data

def tokenize(docs):
    token_counts = Counter()
    df_counts = Counter()
    text_train=[]

    for doc in docs:
        for c in string.punctuation:
            doc = doc.replace(c, "")  # remove punctuation

        doc = doc.lower().split()           # lowercase all tokens       
        
        li=[]
        for token in doc:
            if not bool(re.search(r'\d', token)): #remove tokens with number
                if not token in stop: #remove stopwords
                    li.append(token)
                    token_counts[token] +=1 

        #find document frequency            
        df_doc = set(li)
        for token in df_doc:
            df_counts[token] += 1

        #store processed text
        text = " ".join(li)        
        text_train.append(text)
    return token_counts,df_counts,text_train

def feature_vectorize(docs, dictionary):
    vectorizer = CountVectorizer(vocabulary=dictionary)
    feature_vecs = vectorizer.fit_transform(docs)
    return feature_vecs

def rating_vectorize(star):
    rating_vector = []
    for s in star:
        rate = list(np.zeros(5))
        #print(s)
        rate[int(s)-1]=1
        #print(rate)
        rating_vector.append(rate)
    # print(len(rating_vector))
    # print(len(rating_vector[0]))

    rating_vector = csr_matrix(rating_vector)
    return rating_vector

class LogR_solver:
    def __init__(self, num_feature, num_category):
        self.W = np.random.rand(num_feature,num_category)

    def fit(self, X_train, y_train, epoch, batch_size, lr, lbd):
        batch_num = X_train.shape[0]//batch_size

        for ep in range(epoch):
            #shuffle data for each epoch
            X_train, y_train = shuffle(X_train, y_train, random_state=0)
            y_eval = np.argmax(y_train.toarray(), axis=1)+1

            for b in range(batch_num):
                #slice data for each batch 
                X_batch = X_train[b*batch_size:(b+1)*batch_size]
                y_batch = y_train[b*batch_size:(b+1)*batch_size]

                #compute delta
                wx = np.exp(X_batch.dot(self.W))
                nrow = X_batch.shape[0]
                row_sum = wx.sum(axis=1).reshape(nrow,1)
                wx_norm = np.divide(wx,row_sum)

                delta = X_batch.T.dot(y_batch-wx_norm)-lbd*self.W

                #update weights
                self.W = self.W + lr*delta

    def predict(self, X):
        prob = np.exp(X.dot(self.W))
        nrow = X.shape[0]
        prob_norm = np.divide(prob,prob.sum(axis=1).reshape(nrow,1))
        
        #find hard prediction
        hard = np.argmax(prob, axis=1).reshape(nrow,1)+1

        #find soft prediction
        rate = np.arange(1,6)
        soft = prob_norm.dot(rate).reshape(nrow,1)

        return hard, soft 

    def eval(self, y_true, hard, soft):
        nrow = y_true.shape[0]
        y_true = y_true.reshape(nrow,1)
        acc = accuracy_score(y_true, hard)
        rmse = np.sqrt(mean_squared_error(y_true, soft))
        return acc, rmse 

def write_predictions(name, pred1, pred2):
    lines = [' '.join((str(int(p1)),str(float(p2)))) for p1, p2 in zip(pred1, pred2)] 
    with open(name, 'w') as f:
        for pred in lines:
            f.write("%s\n" % pred)

def main():
	## Task 1 Data Preprocessing
	#read in data
	stars_train, text_train = read_train('/Users/chizhang/Desktop/TextMining/HW/HW5/resources/yelp_reviews_train.json')
	text_dev = read_dev('/Users/chizhang/Desktop/TextMining/HW/HW5/resources/yelp_reviews_dev.json')
	#read in stopwords
	with open('/Users/chizhang/Desktop/TextMining/HW/HW5/resources/stopword.list') as f:
	    stop = f.read().splitlines()

	#tokenize training data
	token_counts, df_counts, text = tokenize(text_train)

	## Task 2 Feature Engineering
	#construct CTF, DF dictionary
	ctf2000=dict(token_counts.most_common(2000))
	ctf_dict=list(ctf2000.keys())

	df2000 = dict(df_counts.most_common(2000))
	df_dict=list(df2000)    

	#vectorize features
	ctf_feature_vecs = feature_vectorize(text, ctf_dict)  
	df_feature_vecs = feature_vectorize(text, df_dict)

	#vectorize ratings
	rate_vecs = rating_vectorize(stars_train) 

	## Task 3 Modeling

	# get training data
	X_ctf = ctf_feature_vecs
	X_df = df_feature_vecs

	y = rate_vecs

	# train ctf model
	logR = LogR_solver(2000,5)
	logR.fit(X_ctf, y, 200, 100, 0.001, 0.1)

	hard, soft = logR.predict(X_ctf)
	y_true = np.array(stars_train)
	acc_ctf, rmse_ctf = logR.eval(y_true, hard, soft)

	# train df model
	logR_df = LogR_solver(2000,5)
	logR_df.fit(X_df, y, 200, 100, 0.001, 0.1)
	hard_df, soft_df = logR_df.predict(X_df)
	y_true = np.array(stars_train)
	acc_df, rmse_df = logR_df.eval(y_true, hard_df, soft_df)

	# use trained model to predict dev set
	text_dev = read_dev('/Users/chizhang/Desktop/TextMining/HW/HW5/resources/yelp_reviews_dev.json')
	_ , _ , text_dev = tokenize(text_dev)

	ctf_feature_vecs_dev = feature_vectorize(text_dev, ctf_dict)  
	df_feature_vecs_dev = feature_vectorize(text_dev, df_dict)

	hard, soft = logR.predict(ctf_feature_vecs_dev)
	hard_dev_df, soft_dev_df = logR_df.predict(df_feature_vecs_dev)



	## Task 4 Feature Engineering - continued

	#construct feature vectors with tf-idf
	tfidf_vector = TfidfVectorizer(max_features=2000)
	tfidf_feature_vecs = tfidf_vector.fit_transform(text)

	X_tfidf = tfidf_feature_vecs

	#tfidf
	logR_tfidf = LogR_solver(2000,5)
	logR_tfidf.fit(X_tfidf, y, 200, 100, 0.001, 0.1)
	hard_tfidf, soft_tfidf = logR_tfidf.predict(X_tfidf)
	acc_tfidf, rmse_tfidf = logR_tfidf.eval(y_true, hard_tfidf, soft_tfidf)

if __name__ == '__main__':
	main()


