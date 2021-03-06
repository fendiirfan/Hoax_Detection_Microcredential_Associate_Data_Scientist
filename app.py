import model_app
import streamlit as st
import numpy as np



st.title('Tool Deteksi Berita Hoax di Media Digital')
st.text("")
st.text("")
st.text("Platfowm ini berfungsi untuk mendeteksi suatu judul berita dapat diklasifikasikan sebagai Hoax atau tidak")
st.text("")

# inputan user
user_input = st.text_input('Input Sentence')
model = ['SVM']
selectedModel = st.selectbox('Please Select the Model : ',model)

# tombols
button = st.button('Prediction')

invalidInput = (len(user_input) <= 3 or
                len(user_input.split(' ')) > 200)

if button==True:
    if invalidInput:
        st.write('Please Input the Correct Sentence')
    else:
        # DISPLAY LOADING PROGRESS FROM 0-20s
        my_bar = st.progress(0)

        model_app.progressBar(my_bar,0,20)
        
        # class model
        model = model_app.model(selectedModel,user_input)

        # DISPLAY LOADING PROGRESS FROM 20-70
        model_app.progressBar(my_bar,20,70)
        
        # PREDICTION
        predict = model.predict_text()

        # DISPLAY LOADING PROGRESS FROM 70-100
        model_app.progressBar(my_bar,70,100)
        
        # DISPLAY PREDICTION
        st.write('Prediksi : Kalimat '+ predict+'\n' )