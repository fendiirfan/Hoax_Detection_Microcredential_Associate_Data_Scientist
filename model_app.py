# import tensorflow as tf
import pandas as pd
import numpy as np
import re
import string
import time
import pickle

class model:
    def __init__(self, selectedModel, text):
        self.text = text
        self.selectedModel = selectedModel
        if selectedModel=='SVM':
            self.model = pickle.load(open('model/'+selectedModel, 'rb'))
        # else:
        #     self.model = tf.keras.models.load_model(model_path)
        
    def preprocessing(self):
        text = self.text
        def clean_text(text):
            text = re.sub('http.*','',text)
            text = re.sub('(@\w+|#\w+)','',text)
            #will replace the html characters with " "
            text=re.sub('<.*?>', '', text)  
            #To remove the punctuations
            text = text.translate(str.maketrans(' ',' ',string.punctuation))
            #will consider only alphabets
            text = re.sub('[^a-zA-Z]',' ',text)  
            #will replace newline with space
            text = re.sub("\n"," ",text)
            #will convert to lower case
            text = text.lower()
            # will replace a word
            text = re.sub("(username|user|url|rt|xf|fx|xe|xa)\s|\s(user|url|rt|xf|fx|xe|xa)","",text)
            # will repalce repated char
            text = re.sub(r'(\w)(\1{2,})', r"\1", text)
            # will replace single word
            text = re.sub(r"\b[a-zA-Z]\b","",text)
            # will replace space more than one
            text = re.sub('(s{2,})',' ',text)
            text = re.sub(' +', ' ', text)
            # will join the words
            text=' '.join(text.split())
            return text

        def applyKamusAlayandStopWord(text):
            # load dicionary
            kamusAlay = pd.read_csv('resource\colloquial-indonesian-lexicon.csv')
            stopWord = pd.read_csv('resource\stopwords_id_satya.txt', header = None)
            stopWord = stopWord[0].to_list()
            kamusTambahan = pd.read_csv('resource\Kamu-Alay.csv')

            # split
            text = text.split(' ')
            # apply kamusAlay
            temp=[]
            for i in range(len(text)):

                try:
                    # kamus alay
                    index = kamusAlay.index[kamusAlay['slang'] == text[i]][0]
                    text[i] = kamusAlay['formal'][index]
                    # stop word
                except:
                    pass

                try:
                    # kamus alay
                    index = kamusTambahan.index[kamusTambahan['kataAlay'] == text[i]][0]
                    text[i] = kamusTambahan['kataBaik'][index]
                    # stop word
                except:
                    pass

                try:
                    if text[i] in stopWord:
                        text.remove(text[i])
                except:
                    pass

                
            text = ' '.join(text)
            return text
        
        text = clean_text(self.text) # clean text
        text = applyKamusAlayandStopWord(text) # repalce kata alay
        return text
    
    def convert_text(self,text):
        if self.selectedModel=='SVM':
            with open('model/tokenizer_SVM', 'rb') as handle:
                tokenizer = pickle.load(handle)
            return tokenizer.transform([text])
        # else:
        #     max_length = 14
        #     padding_type="post"
        #     truncating_type="post"

        #     with open('model/tokenizer_deep_learning.pkl', 'rb') as handle:
        #         tokenizer = pickle.load(handle)

        #     x_sequences = tokenizer.texts_to_sequences([text])
            
        #     return tf.keras.preprocessing.sequence.pad_sequences(x_sequences,maxlen = max_length, padding = padding_type, truncating = truncating_type)


    def predict_text(self):
        className = {
            0 : 'NORMAL',
            1 : 'BULLYING',
        }
        text = self.preprocessing()
        converted_text = self.convert_text(text)
        prediksi = self.model.predict(converted_text)
        if self.selectedModel=='SVM':
            return className[prediksi[0]]
        else:
            return [prediksi,className[np.argmax(prediksi)]]

def progressBar(my_bar,start,end):
    for percent_complete in range(start,end):
        time.sleep(0.1)
            
        my_bar.progress(percent_complete + 2)