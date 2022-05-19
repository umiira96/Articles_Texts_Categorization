# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:33:01 2022

@author: umium
"""

import os
import json
import numpy as np
from modules import ExploratoryDataAnalysis
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json


TOKEN_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')
MODEL_PATH = os.path.join(os.getcwd(),'model.h5')

#%% model loading
category_classifier = load_model(MODEL_PATH)
category_classifier.summary()

#%% tokenizer loading
with open(TOKEN_PATH, 'r') as json_file:
    loaded_tokenizer = json.load(json_file)
    
#%%EDA
#Step1) loading data
#insert input manully
new_text = [input('Articles/Texts:\n')]


#Step2) data cleaning
##% data cleaning
eda = ExploratoryDataAnalysis()
removed_tags = eda.remove_tags(new_text) #to remove tags


#Step3) features selection
#Step4) data preprocessing
#to vectorize new review
#to feed the tokens into keras
loaded_tokenizer = tokenizer_from_json(loaded_tokenizer)

#to vectorize the review into integers
new_text = loaded_tokenizer.texts_to_sequences(removed_tags)
new_text = eda.category_pad_sequences(new_text)

#%%
#model prediction
outcome = category_classifier.predict(np.expand_dims(new_text,axis=-1))


#%%
#tech          = [0,0,0,0,1]
#sport         = [0,0,0,1,0]
#politics      = [0,0,1,0,0]
#entertainment = [0,1,0,0,0]
#business      = [1,0,0,0,0]

text_dict = {4:'Tech', 3:'Sport', 2:'Politics', 1:'Entertainment', 
                  0:'Business'}
print('The category of the article/text is: ' + text_dict[np.argmax(outcome)])




