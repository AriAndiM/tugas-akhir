# Terminal Alt+1
import streamlit as st
import streamlit.components.v1 as html
import pandas as pd
import numpy as np
import string
import re
import pickle
import joblib
import warnings
import nltk
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from streamlit_option_menu import option_menu
from collections import defaultdict
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
# string.punctuation
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

with st.container():
    st.write(f'<h1 style="text-align:center;"><span style="color:#1e81b0">Pengaruh Penggunaan Kata Sifat Pada Analisis Sentimen Ulasan Pantai Jawa Timur Menggunakan</span><span style="color:#1e81b0"><i> Skip-gram</i> dan <i>Support Vector Machine</i>(SVM)</span></h1>', unsafe_allow_html=True)
    teks_input = st.text_area("**Masukkan Teks**")
    submit = st.button("Prediksi", type="primary")
    if submit:
        if teks_input:
            df_mentah = pd.DataFrame({'Ulasan': [teks_input]})
            # df_mentah

            # Cleaning
            def cleaning(text):
              # untuk mengganti link menjadi spasi
              ulasan = re.sub(r'https\S*', ' ', text)
              # untuk mengganti tanda baca dengan spasi
              ulasan = re.sub(r'[{}]'.format(string.punctuation), ' ', ulasan)
              # untuk mengganti karakter selain a-z seperti emot menjadi spasi
              ulasan = re.sub('[^a-zA-Z]', ' ', ulasan)
              # mengganti newline dengan spasi
              ulasan = re.sub('\n', ' ', ulasan)
              # mengganti kata hanya 1 huruf dengan spasi
              ulasan = re.sub(r'\b[a-zA-Z]\b', ' ', ulasan)
              # menghilangkan spasi berlebih
              ulasan = ' '.join(ulasan.split())
              return ulasan

            def case_folding(text):
              ulasan = text.lower()
              return ulasan

             # Tokenisasi
            def tokenization(text):
                token = word_tokenize(text)
                return ','.join(token)

            # Normalisasi
            path_kbba = 'Kamus/kbba.xlsx'
            kbba = pd.read_excel(path_kbba)

            kbba_dict = {}
            for index,row in kbba.iterrows():
              kbba_dict[row['slang']] = row['baku']

            def normalization(text):
              normalized_token = []
              for kata in text.split(','):
                # print(text)
                if kata in kbba_dict:
                  baku = kbba_dict[kata]
                  # print(text, '->', baku)
                  banyak_kata = len(baku.split())
                  if banyak_kata > 1:
                    kata_baku = baku.split()
                    print(baku)
                    for idx, row in enumerate(kata_baku):
                      normalized_token.append(kata_baku[idx])
                  else:
                    normalized_token.append(baku)

                if kata not in kbba_dict:
                  normalized_token.append(kata)

              return ','.join(normalized_token)

            # Stopword removal
            stoplist = stopwords.words('indonesian')
            # proses stopword removal
            def stopword_removal(x):
              result = []
              # memisahkan antar kata berdasarkan spasi
              for i in x.split(','):
                # dilakukan pengecekan pada stoplist
                if i not in stoplist:
                  # memasukkan kata yang tidak ada di stoplist ke array
                  result.append(i)
              return ','.join(result)

            # Stemming
            Fact = StemmerFactory()
            Stemmer = Fact.create_stemmer()

            def stemming(text):
              hasil = []
              for kata in text.split(','):
                stemming_kata = Stemmer.stem(kata)
                hasil.append(stemming_kata)
              return ','.join(hasil)

            df_mentah['Clean']= df_mentah['Ulasan'].apply(cleaning)
            # df_clean = df_mentah[['Clean']]
            df_clean['Case Folding']= df_clean['Clean'].apply(case_folding)
            # st.write(df_clean['Case Folding'])
            df_clean['Tokenization'] = df_clean['Case Folding'].apply(tokenization)
            # st.write(df_clean['Tokenization'])
            df_clean['Normalization'] = df_clean['Tokenization'].apply(normalization)
            # st.write(df_clean['Normalization'])
            df_clean['Stopword Removal']=df_clean['Normalization'].apply(stopword_removal)
            # st.write(df_clean['Stopword Removal'])
            df_clean['Stemming'] = df_clean['Stopword Removal'].apply(stemming)
            # st.write(df_clean['Stemming'])

            # skip-gram
            model_w2v = Word2Vec.load("Ekstraksi Fitur\skenario 1 model w6.model")

            data = df_clean['Stemming']
            hasil_preprocess = [row.split(',') for row in data]

            document_word_vectors = []  # list untuk menyimpan pasangan kata dan vector untuk setiap dokumen
            document_vectors = []  # list untuk menyimpan vector rata-rata dari setiap dokumen

            for doc in hasil_preprocess:
                if isinstance(doc, list):
                    word_vectors = []
                    for word in doc:
                        if word in model_w2v.wv:
                            word_vector = model_w2v.wv[word]  # mendapatkan vector kata dari model
                            word_vectors.append((word, word_vector))  # menambahkan pasangan kata dan vector ke dalam list
                        else:
                            # Jika kata tidak ada dalam model, tambahkan vektor nol
                            word_vector = np.zeros(model_w2v.vector_size)
                            word_vectors.append((word, word_vector))

                    if word_vectors:
                        total_vector = 0
                        for pasangan_kata_vector in word_vectors:
                            vector = pasangan_kata_vector[1]
                            total_vector += vector

                        doc_vector = total_vector / len(word_vectors)
                        document_word_vectors.append(word_vectors)  # menambahkan pasangan kata dan vector untuk dokumen saat ini
                        document_vectors.append(doc_vector)  # menambahkan vector rata-rata ke dalam list
                    else:
                        document_vectors.append(None)

            # st.write('Vector Kata')
            # for i in document_word_vectors:
            #     st.write(i)

            # st.write('Vector Dokumen')
            if document_vectors is not None:
                document_vectors_df = pd.DataFrame(document_vectors)
                st.write(document_vectors_df)
            else:
                data = {i+1: [0] for i in range(100)}
                document_vectors_df = pd.DataFrame(data)
                # document_vectors_df
                st.write(document_vectors_df)

            #klasifikasi menggunakan SVM
            with open('Model/Skenario 1 _ w6 _ c10g1 rbf.pkl','rb') as r:
                model_svm = pickle.load(r)

            pred = model_svm.predict(document_vectors_df)
            if pred == 0:
                st.write(f'Ulasan "{teks_input}" memiliki <span style="background-color:#fb4c4c; padding: 5px; border-radius: 5px; color:white;">**Sentimen Negatif**</span>', unsafe_allow_html=True)
            if pred == 1:
                st.write(f'Ulasan "{teks_input}" memiliki <span style="background-color:#1e81b0; padding: 5px; border-radius: 5px; color:white;">**Sentimen Positif**</span>', unsafe_allow_html=True)

        else :
            st.warning('Anda Belum Masukkan Teks', icon="⚠️")
