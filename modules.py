# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:54:23 2022

@author: umium
"""

import re
import json
import os
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, Bidirectional
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


class ExploratoryDataAnalysis():
    
    def __init(self):
        pass
    
    def remove_tags(self,data):
        for index, comment in enumerate(data):
            data[index] = re.sub('<.*?>', '', comment)
        return data
    
    def category_tokenizer(self,data,token_save_path,
                            num_words=10000,
                            oov_token = '<OOV>',
                            prt=False):
    
        tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        tokenizer.fit_on_texts(data)

        #to save tokenizer for deployment purpose
        token_json = tokenizer.to_json()
        
        with open(token_save_path, 'w') as json_file:
            json.dump(token_json, json_file)
        
        
        #to observe the number of words
        word_index = tokenizer.word_index
        print(word_index)
        print(dict(list(word_index.items())[0:10]))
        
        
        #to vectorize the sequences of text
        data = tokenizer.texts_to_sequences(data)
        
        return data
    
    def category_pad_sequences(self,data):
        return pad_sequences(data, 
                             maxlen=350,
                             padding='post',
                             truncating='post') 

class ModelCreation():
    
    def lstm_layer(self,num_words, nb_category,
                   embedding_output=64, nodes=32, dropout=0.2):
        
        model = Sequential()
        model.add(Embedding(num_words, embedding_output))
        model.add(Bidirectional(LSTM(nodes, return_sequences=True)))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nodes)))
        model.add(Dropout(dropout))
        model.add(Dense(nb_category, activation='softmax'))
        model.summary()
        return model
    
class ModelEvaluation():
    def report_metrics(self,y_true,y_pred):
        print(classification_report(y_true,y_pred))
        print(confusion_matrix(y_true,y_pred))
        print(accuracy_score(y_true,y_pred))


#%%
if __name__ == '__main__':
    
    PATH_LOGS = os.path.join(os.getcwd(), 'logs')
    MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.SA')
    TOKENIZER_JSON_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')
    
    URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
    df = pd.read_csv(URL)
    text = df['text']
    category = df['category']
    
    #%%
    eda = ExploratoryDataAnalysis()
    test = eda.remove_tags(text)
    
    test = eda.category_tokenizer(test, token_save_path=TOKENIZER_JSON_PATH)
    
    test = eda.category_pad_sequences(test)
    
    #%%
    nb_category = len(df['category'].unique())
    mc = ModelCreation()
    model = mc.lstm_layer(10000, nb_category)
    
    
