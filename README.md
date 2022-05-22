![badge](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

![badge](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

# Articles_Texts_Categorization
 Analyse the articles/texts and return their categories
 
# Datasets
Credit to susanli2016 for the datasets

URL = https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv

Text documents are essential as they are one of the richest sources of data for businesses. Text documents often contain crucial information which might shape the market trends or influence the investment flows. Therefore, companies often hire analysts to monitor the trend via articles posted online, tweets on social media platforms such as Twitter or articles from newspaper. However, some companies may wish to only focus on articles related to technologies and politics. Thus, filtering of the articles into different categories is required.

Often the categorization of the articles is conduced manually and retrospectively; thus, causing the waste of time and resources due to this arduous task. Hence, your job as a machine learning engineer is tasked to categorize unseen articles into 5 categories namely Sport, Tech, Business, Entertainment and Politics.
 
# Model Creation

 1) This model is developed by using Embeding Bidirectional LSTM layers in order to achieve 99.87% accuracy and 0.92 F1 score.

 ![images](https://github.com/umiira96/Articles_Texts_Categorization/blob/main/images/model_execution.png "model_execution.png")

 ![images](https://github.com/umiira96/Articles_Texts_Categorization/blob/main/images/training_loss_accuracy.png "training_loss-accuracy.png")

 ![images](https://github.com/umiira96/Articles_Texts_Categorization/blob/main/images/accuracy_and_F1_score.png "accuracy_and_F1_score.png")

 2) TensorFlow library is used to develop and train the model.
 3) The graph using Tensorboard.

 ![images](https://github.com/umiira96/Articles_Texts_Categorization/blob/main/images/graph_tensorboard_callbacks.png "graph_tensorboard_callbacks.png")

# How to use it
* Clone repo and run it
* train.assesment3.py is a script that trains the data
* deploy.assesment3.py is a script for deployment
* You may insert your articles/texts and then enter to return it's category







