# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:28:34 2022

@author: umium
"""

import os
import pandas as pd
import numpy as np
import datetime
from modules import ExploratoryDataAnalysis, ModelCreation, ModelEvaluation
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard


URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
TOKENIZER_JSON_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')
PATH_LOGS = os.path.join(os.getcwd(), 'logs')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')


#%%EDA
#Step 1) Import data
df = pd.read_csv(URL)
text = df['text']
category = df['category']


#%%
#Step 2) Data cleaning
#remove tags
eda = ExploratoryDataAnalysis()
text = eda.remove_tags(text)

#%%
#Step 3) Features selection
#Step 4) Data vectorization
text = eda.category_tokenizer(text, TOKENIZER_JSON_PATH)

text = eda.category_pad_sequences(text)

#%%
#Step 5) Preprocessing
one_hot_encoder = OneHotEncoder(sparse=False)
category = one_hot_encoder.fit_transform(np.expand_dims(category,axis=-1))

#to calculate the number of total category
nb_category = len(np.unique(df['category']))

#train test split
X_train,X_test,y_train,y_test = train_test_split(text,
                                                 category,
                                                 test_size=0.3,
                                                 random_state=123)
X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)

#%%
#Step 6) Model creation class
mc = ModelCreation()

num_words = 10000

model = mc.lstm_layer(num_words, nb_category)


#compile & model traning
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='acc')

log_dir = os.path.join(PATH_LOGS,
                       datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        
tensorboard_callbacks = TensorBoard(log_dir=log_dir, histogram_freq=1)


model.fit(X_train,y_train,epochs=10, 
          validation_data=(X_test,y_test), 
          callbacks=tensorboard_callbacks)


#%% 
#Step 7) Model evaluation
predicted_advanced = np.empty([len(X_test), 5])
for index , test in enumerate(X_test):
    predicted_advanced[index,:] = model.predict(np.expand_dims(test,axis=0))

#%% 
#Step 8) Model analysis
y_pred = np.argmax(predicted_advanced, axis=1)
y_true = np.argmax(y_test, axis=1)

me = ModelEvaluation()
me.report_metrics(y_true,y_pred)

#%% 
#Step 9)Model deployment
model.save(MODEL_SAVE_PATH)








