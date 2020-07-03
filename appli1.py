# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 07:31:19 2020

@author: Bellemiss972
"""
from sklearn.ensemble import RandomForestClassifier
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.externals import joblib

#charger le modèle
model=joblib.load('model_iris.pkl')


#Prédiction (préparation)
def predict(sepal_length,sepal_width,petal_length,petal_width):
    predictions=model.predict(sepal_length,sepal_width,petal_length,petal_width)
    return predictions


def name():
    
    #Titre de la page
    st.header('**Iris classification**')
    st.write("Cet outil est l'aboutissement d'un projet de machine-learning sur les fleurs d'iris. Le modèle prédictif a obtenu un score de **précision de 97%**. Essayez-le.")
    from PIL import Image
    image=Image.open('fleur.jpg')
    logo=Image.open('logo.png')
    #st.image(image,use_column_width=False)
    st.sidebar.image(logo,use_column_width=True)
    selection=st.sidebar.selectbox(
        "Comment voulez-vous prédire vos données ?",
    ("En ligne", "Importer un fichier"))

    
    #créer la sidebar
    
    st.sidebar.info("Cette application est créee pour prédire le type d'iris")
    st.sidebar.success('https://www.agence-marketic.fr')
    st.sidebar.image(image,use_column_width=True)
    
  
    #Personnaliser la page principale
        
    st.title ("**Prédiction du type d'iris selon les caractéristiques suivantes**")
        
    if selection=='En ligne':
        sepal_length=st.slider('Indiquer la longueur des sépales (mm)',min_value=0.5,max_value=5.5,value=2.2)	
        sepal_width=st.slider('Indiquer la largeur des sépales (mm)',min_value=0.5,max_value=5.5,value=2.2)
        petal_length=st.slider('Indiquer la longueur des pétales (mm)',min_value=0.1,max_value=5.5,value=1.5)
        petal_width=st.slider('Indiquer la largeur des pétales(mm)',min_value=0.1,max_value=4.9,value=3.1)
        
#Prédiction finale
        resultat=""
    
        #variables={'sepal_length':sepal_length,'sepal_width':sepal_width,'petal_length':petal_length,'petal_width':petal_width}
        #df = pd.DataFrame([variables])  
    
        
        if st.button("Prédire"):
            resultat=model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
            st.success("""Le type d'iris est **{}**""".format(resultat)) 
           

    if selection == "Importer un fichier":
       
        batch = st.file_uploader("Importer un fichier de type csv", type=["csv"])
        if batch is not None:
            data = pd.read_csv(batch)
            predictions=model.predict_proba(data)
            predictions=model.predict(data)
            
            #test
            pred=pd.Series(predictions.reshape(150,))
            dataframe=pd.DataFrame([[data,predictions]])
            concat=pd.concat([data,pred], axis=1)
            concat.columns=['longueur_sepal','largeur_sepal','longueur_petal','largeur_petal','predictions']
            st.write(concat)
          
            

   # Pour lancer l'app web
 
if __name__=='__main__' : name()