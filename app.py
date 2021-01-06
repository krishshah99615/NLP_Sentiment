import streamlit as st
import tensorflow as tf
import os
import pandas as pd
import pickle
import cv2
import numpy as np
from  tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



tokenizer=pickle.load( open ("tokenizer.pkl", "rb"))
model=  tf.keras.models.load_model('model.h5', compile = False)
print('model loaded')




st.write("""
  # NLP Sentiment
""")
sentence = st.text_input('Input your sentence here:') 
pr = st.button('predict')
image1 = cv2.cvtColor(cv2.imread('distribution.png'),cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(cv2.imread('loss.png'),cv2.COLOR_BGR2RGB)
image3 = cv2.cvtColor(cv2.imread('acc.png'),cv2.COLOR_BGR2RGB)
image4 = cv2.cvtColor(cv2.imread('cm.png'),cv2.COLOR_BGR2RGB)

def import_and_predict(sentence,tokenizer,model):
  s=[]
  s.append(sentence)
  s_seq=tokenizer.texts_to_sequences(s)
  s_padded=pad_sequences(s_seq,maxlen=80,padding='post')
  label = {'0':'joy','1':'anger','2':'love','3':'sad','4':'fear','5':'surprise'}
  ans=label[str(int(np.argmax(model.predict(s_padded))))]

  return ans


if pr:
  
  p = import_and_predict(sentence,tokenizer , model)
  st.success("Given sentence has sentiment "+str(p))

else:
  st.text("Pls enter text")


st.text("Accuracy Train :0.9796  ")

st.text("Accuracy Test : 0.9319999814033508 ")

st.text("Accuracy Validation : 0.9360 ")
st.image([image1,image2,image3,image4], caption=['Distribution','loss','accuracy','confusion matrix'])

