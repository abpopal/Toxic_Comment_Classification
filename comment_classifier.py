# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 01:13:31 2021

@author: Abdul Baqi Popal
"""


# importing libraries
import matplotlib.pyplot as plt
import numpy as np
import re 
import pandas as pd
import string
import seaborn as sns
import wordcloud
from nltk.corpus import stopwords  # Remove useless words
from nltk.stem import SnowballStemmer  # Convert words to base form; aggressive

# Import packages that help us to create document-term matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# Import packages for pre-processing
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel

# Import tools to split data and evaluate model performance
from sklearn.model_selection import train_test_split
from sklearn.metrics import  precision_score, confusion_matrix


# Import ML algos

from sklearn.ensemble import RandomForestClassifier

#---------------data preprocessing_------------------
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    x = text.split()
    new_text = []
    for word in x:
        if word not in stop_words:
            new_text.append(stemmer.stem(word))
            
    text = ' '.join(new_text)
    return text

dataset = pd.read_csv('dataset/train.csv')

data_count=dataset.iloc[:,2:].sum()
# ------some data visualization before further process 

plt.figure(figsize=(8,4))

# Plot a bar chart using the index (category values) and the count of each category.
ax = sns.barplot(data_count.index, data_count.values)

plt.title("Comment Classifier")
plt.ylabel('Occurrences', fontsize=12)
plt.xlabel('Comment Category', fontsize=12)

#adding the text labels for each bar
rects = ax.patches
labels = data_count.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()

#------------end of visualization 
#since there are no null values present in the dataset we will no worry about it
#one hot encoding is already done on the data since all features have seperate columns

# remove all numbers with letters attached to them
alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)

# '[%s]' % re.escape(string.punctuation),' ' - replace punctuation with white space
# .lower() - convert all strings to lowercase 
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())

# Remove all '\n' in the string and replace it with a space
remove_n = lambda x: re.sub("\n", " ", x)

# Remove all non-ascii characters 
remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]',r' ', x)
preprocess_v = lambda x: preprocess(x)
# Apply all the lambda functions wrote previously through .map on the comments column
dataset['comment_text'] = dataset['comment_text'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)


# now we will create different data frames for each category
dataset_toxic = dataset.reindex(columns =['id', 'comment_text', 'toxic'])
dataset_severe_toxic = dataset.reindex(columns =['id', 'comment_text', 'severe_toxic'])
dataset_obscene = dataset.reindex(columns =['id', 'comment_text', 'obscene'])
dataset_threat = dataset.reindex(columns =['id', 'comment_text', 'threat'])
dataset_insult = dataset.reindex(columns =['id', 'comment_text', 'insult'])
dataset_identity_hate = dataset.reindex(columns =['id', 'comment_text', 'identity_hate'])

#we need to organize our datasets for training in such a way that
#for each category 50% will be 1 and 50% will 0. in real dataset each feature
#make less than 10%



toxic_1 = dataset_toxic[dataset_toxic['toxic']==1].iloc[0:5000,:]
toxic_0 = dataset_toxic[dataset_toxic['toxic']==0].iloc[0:5000,:]
toxic_data = pd.concat([toxic_1,toxic_0], axis=0)

severe_toxic_1 = dataset_severe_toxic[dataset_severe_toxic['severe_toxic']==1].iloc[0:1595,:]
severe_toxic_0 = dataset_severe_toxic[dataset_severe_toxic['severe_toxic']==0].iloc[0:1595,:]
severe_toxic_data = pd.concat([severe_toxic_1,severe_toxic_0], axis=0)

obscene_1 = dataset_obscene[dataset_obscene['obscene']==1].iloc[0:5000,:]
obscene_0 = dataset_obscene[dataset_obscene['obscene']==0].iloc[0:5000,:]
obscene_data = pd.concat([obscene_1,obscene_0], axis=0)

insult_1 = dataset_insult[dataset_insult['insult']==1].iloc[0:5000,:]
insult_0 = dataset_insult[dataset_insult['insult']==0].iloc[0:5000,:]
insult_data = pd.concat([insult_1,insult_0], axis=0)

threat_1 = dataset_threat[dataset_threat['threat']==1].iloc[0:478,:]
threat_0 = dataset_threat[dataset_threat['threat']==0].iloc[0:478,:]
threat_data = pd.concat([threat_1,threat_0], axis=0)

identity_hate_1 = dataset_identity_hate[dataset_identity_hate['identity_hate']==1].iloc[0:1405,:]
identity_hate_0 = dataset_identity_hate[dataset_identity_hate['identity_hate']==0].iloc[0:5620,:]
identity_hate_data = pd.concat([identity_hate_1,identity_hate_0], axis=0)

#------------------visualize wordcloud for each category
def wordcloud_vis(df, label):
    
    subset=df[df[label]==1]
    text=subset.comment_text.values
    wc= wordcloud.WordCloud(background_color="white",max_words=4000)

    wc.generate(" ".join(text))

    plt.figure(figsize=(50,70))
    plt.subplot(221)
    plt.axis("off")
    plt.title("{} words".format(label), fontsize=70)
    plt.imshow(wc.recolor(colormap= 'gist_earth' , random_state=244), alpha=0.98)

wordcloud_vis(identity_hate_data, 'identity_hate')

#-----random forest model training and prediction for toxic 

x = toxic_data.comment_text
y = toxic_data['toxic']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)

#intialize vectorizer
tfv = TfidfVectorizer(ngram_range=(1,1), stop_words='english')
x_train_fit = tfv.fit_transform(x_train)
x_test_fit = tfv.transform(x_test)

randomforest = RandomForestClassifier(n_estimators=100, random_state=42)
randomforest.fit(x_train_fit, y_train)

pred = randomforest.predict(x_test_fit)

#lets display the results for this one confusion matrix 

cm = confusion_matrix(y_test, pred)

print("confusion matrix for toxic ", cm)
toxic_precision = precision_score(y_test, pred)
print("precision for toxic is: ", toxic_precision)

#-----random forest model training and prediction for identity hate 

x_identity_hate = identity_hate_data.comment_text
y_identity_hate = identity_hate_data['identity_hate']

identity_hate_x_train, identity_hate_x_test, identity_hate_y_train, identity_hate_y_test = train_test_split(x_identity_hate,y_identity_hate, test_size=0.3, random_state=42)

#intialize vectorizer
identity_hate_tfv = TfidfVectorizer(ngram_range=(1,1), stop_words='english')
identity_hate_x_train_fit = identity_hate_tfv.fit_transform(identity_hate_x_train)
identity_hate_x_test_fit = identity_hate_tfv.transform(identity_hate_x_test)

identity_hate_randomforest = RandomForestClassifier(n_estimators=100, random_state=42)
identity_hate_randomforest.fit(identity_hate_x_train_fit, identity_hate_y_train)

identity_hate_x_test_fit_pred = identity_hate_randomforest.predict(identity_hate_x_test_fit)

#lets display the results for this one confusion matrix 

identity_hate_cm = confusion_matrix(identity_hate_y_test , identity_hate_x_test_fit_pred)
print("confusion matrix for identity hate",identity_hate_cm)
identity_hate_precision = precision_score(identity_hate_y_test, identity_hate_x_test_fit_pred)
print("precision for Identity hate is: ", identity_hate_precision)

#-----random forest model training and prediction for obscene

x_obscene = obscene_data.comment_text
y_obscene = obscene_data['obscene']

obscene_x_train, obscene_x_test, obscene_y_train, obscene_y_test = train_test_split(x_obscene,y_obscene, test_size=0.3, random_state=42)

#intialize vectorizer
obscene_tfv = TfidfVectorizer(ngram_range=(1,1), stop_words='english')
obscene_x_train_fit = obscene_tfv.fit_transform(obscene_x_train)
obscene_x_test_fit = obscene_tfv.transform(obscene_x_test)

obscene_randomforest = RandomForestClassifier(n_estimators=100, random_state=42)
obscene_randomforest.fit(obscene_x_train_fit, obscene_y_train)

obscene_x_test_fit_pred = obscene_randomforest.predict(obscene_x_test_fit)

#lets display the results for this one confusion matrix 

obscene_cm = confusion_matrix(obscene_y_test , obscene_x_test_fit_pred)
print("confusion matrix for obscene",obscene_cm)
obscene_precision = precision_score(obscene_y_test, obscene_x_test_fit_pred)
print("precision for obscene is: ", obscene_precision)
#-----random forest model training and prediction for threat

x_threat = threat_data.comment_text
y_threat= threat_data['threat']

threat_x_train, threat_x_test, threat_y_train, threat_y_test = train_test_split(x_threat,y_threat, test_size=0.3, random_state=42)

#intialize vectorizer
threat_tfv = TfidfVectorizer(ngram_range=(1,1), stop_words='english')
threat_x_train_fit = threat_tfv.fit_transform(threat_x_train)
threat_x_test_fit = threat_tfv.transform(threat_x_test)

threat_randomforest = RandomForestClassifier(n_estimators=100, random_state=42)
threat_randomforest.fit(threat_x_train_fit, threat_y_train)

threat_x_test_fit_pred = threat_randomforest.predict(threat_x_test_fit)

#lets display the results for this one confusion matrix 

threat_cm = confusion_matrix(threat_y_test , threat_x_test_fit_pred)
print("confusion matrix for threat",threat_cm)
threat_precision = precision_score(threat_y_test, threat_x_test_fit_pred)
print("precision for threat is: ", threat_precision)
#-----random forest model training and prediction for insult

x_insult = insult_data.comment_text
y_insult= insult_data['insult']

insult_x_train, insult_x_test, insult_y_train, insult_y_test = train_test_split(x_insult,y_insult, test_size=0.3, random_state=42)

#intialize vectorizer
insult_tfv = TfidfVectorizer(ngram_range=(1,1), stop_words='english')
insult_x_train_fit = insult_tfv.fit_transform(insult_x_train)
insult_x_test_fit = insult_tfv.transform(insult_x_test)

insult_randomforest = RandomForestClassifier(n_estimators=100, random_state=42)
insult_randomforest.fit(insult_x_train_fit, insult_y_train)

insult_x_test_fit_pred = insult_randomforest.predict(insult_x_test_fit)

#lets display the results for this one confusion matrix 

insult_cm = confusion_matrix(insult_y_test , insult_x_test_fit_pred)
print("confusion matrix for insult",insult_cm)
insult_precision = precision_score(insult_y_test, insult_x_test_fit_pred)
print("precision for insult is: ", threat_precision)


'''
cmmnt_toxic = tfv.transform(['you are very nice person'])
cmmnt_identity_hate = identity_hate_tfv.transform(cmmnt)

print("randome forest toxic " ,randomforest.predict_proba(cmmnt_toxic)[:,1])
print("randome forest identity " ,identity_hate_randomforest.predict_proba(cmmnt_identity_hate)[:,1])

'''

cmmnt = ["stupid stop deleting my stuff go die and fall in a hole go to hell "]
nice_cmmnt = ['you are such a great person']


percent_arr =[]

def classify(cmmnt):

    percent_arr.append(('toxic',randomforest.predict_proba(tfv.transform(cmmnt))[:,1] ))
    percent_arr.append(('threat', threat_randomforest.predict_proba(threat_tfv.transform(cmmnt))[:,1]))
    percent_arr.append(('identity_hate',identity_hate_randomforest.predict_proba(identity_hate_tfv.transform(cmmnt))[:,1] ))
    percent_arr.append(('insult',insult_randomforest.predict_proba(insult_tfv.transform(cmmnt))[:,1] ))


#-------------calling classify function from here 
    
classify(cmmnt)

#------------------------------

percent_arr = sorted(percent_arr,key=lambda l:l[1], reverse=True)
mean_value = 0
t_sum = 0
comment_classify = ""
for x in range(4):
    t_sum =t_sum + percent_arr[x][1]

mean_value = t_sum/4


if mean_value <0.5:
    comment_classify = 'Normal'
else:
    comment_classify = percent_arr[0][0]
print("--------------your comment was:   ", comment_classify)