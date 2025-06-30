import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

## load the model
model=load_model('next_word_lstm.h5')

## load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer=pickle.load(handle)
    
## function to predict the new word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len (token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):] ## Ensure the sequnece length matches max_sequnece
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0) ##verbose shows, kitna info dikhana hai during training
    predicted_word_index = np.argmax(predicted, axis=1) ## argmax gives the most likely indexs        
    ## have used axis =1, as prediction is 2D array
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:                
            return word
    return None
    
## Streamlit app code 

st.title("Next Word Prediction with LSTM and Early Stopping")
input_text=st.text_input("Enter the sequence of words", "To be or not be") ##this second sentence is the dedault value
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1 ## will retrive the mex seqence length
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"Next word: {next_word}")