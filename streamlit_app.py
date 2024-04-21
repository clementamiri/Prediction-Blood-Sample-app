import streamlit as st
import pickle
import pandas as pd 
import numpy as np
from PIL import Image 

df = pd.read_csv("tab/df.csv")


st.sidebar.title('Sommaire')
pages = ["Présentation",'Modèle','Prédire ses analyses de sang']
page = st.sidebar.radio("aller vers", pages)

#-------------------------------------------------------------------------Présentation-----------------------------------------------------------------------------------------

if page == pages[0]:
    cover = Image.open("image/sang.png")
    st.image(cover, use_column_width=True)
    st.markdown("<center><b>Application pour prédire le diagnostic en fonction des résultats d'une prise de sang.</b></center>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("*Attention cette application utilise des données centrées, c'est une démonstration de machine learning et n'a pas d'objectif médical.*")
    

#-------------------------------------------------------------------------Modèle---------------------------------------------------------------------------------------------------
if page == pages[1]:
    st.title('Modèle Machine Learning')
    text_xgb = "XGBoost (eXtreme Gradient Boosting) est un modèle de Machine Learning très populaire chez les Data Scientists.\
        Il s'agit d'un modèle amélioré de l'algorithme d'amplification de gradient (Gradient Boost). \
            Cet algorithme d'apprentissage machine est également utilisé pour résoudre les problématiques courantes d'entreprises tout en se basant sur une quantité minimale de ressources."
    st.write(text_xgb, text_align='justify')
    
    text_dataframe = pd.read_csv('tab/df.csv')
    st.write("Dataframe utilisé pour prédire le diagnostic. Les données ont été centré réduite.")
    st.dataframe(text_dataframe.head())
    
    st.write("XGBoostClassifier utilisé avec ces hyperparamètres suivant:")
    algo = Image.open("image/code_xgb.png")
    st.image(algo, use_column_width=True) 
    
    st.write("Graphique avec les variables qui expliquent le mieux le diagnostic de la prise de sang")
    tableau = Image.open("image/important.png")
    st.image(tableau, use_column_width=True)
    
    st.markdown("**Matrice de Confusion**")
    matrice = pd.read_csv("tab/matrice.csv")
    st.dataframe(matrice)
    
    st.markdown('**Rapport de Classification**')
    report = pd.read_csv('tab/report.csv')
    st.dataframe(report)
    
    st.write("Les résultats montrent que le modèle est capable de discriminer les caractéristiques des maladies mesurées par la prise de sang.")
     

#-------------------------------------------------------------------------Classifier------------------------------------------------------------------------------------------------
if page == pages[2]:
    st.title('Prédiction Classifier')
    @st.cache_data

    def charger_modele(model_name):
    # Charger le modèle à partir du fichier Pickle
        with open(model_name, 'rb') as fichier_modele:
            modele = pickle.load(fichier_modele)
            return modele
    
    model = charger_modele("xgb_streamlit.pkl")
    
    col1, col2 = st.columns(2)
    
    with col1:
        Glucose = st.slider("Glucose", min_value=0.0, max_value=1.0, step=0.01)
        Cholesterol = st.slider('Cholesterol', min_value=0.0, max_value=1.0, step=0.01)
        Hemoglobin = st.slider('Hemoglobin', min_value=0.0, max_value=1.0, step=0.01)
        Platelets = st.slider('Platelets', min_value=0.0, max_value=1.0, step=0.01)
        White_Blood_Cells = st.slider('White_Blood_Cells', min_value=0.0, max_value=1.0, step=0.01)
        Red_Blood_Cells = st.slider('Red Blood Cells', min_value=0.0, max_value=1.0, step=0.01)
        Hematocrit = st.slider('Hematocrit', min_value=0.0, max_value=1.0, step=0.01)
        Mean_Corpuscular_Volume = st.slider('Mean Corpuscular Volume', min_value=0.0, max_value=1.0, step=0.01)
        Mean_Corpuscular_Hemoglobin = st.slider('Mean Corpuscular Hemoglobin', min_value=0.0, max_value=1.0, step=0.01)
        Mean_Corpuscular_Hemoglobin_Concentration = st.slider('Mean Corpuscular Hemoglobin Concentration', min_value=0.0, max_value=1.0, step=0.01)
        Insulin = st.slider('Insulin', min_value=0.0, max_value=1.0, step=0.01)
        BMI = st.slider('BMI', min_value=0.0, max_value=1.0, step=0.01)
    with col2:
        Systolic_Blood_Pressure = st.slider('Systolic Blood Pressure', min_value=0.0, max_value=1.0, step=0.01)
        Diastolic_Blood_Pressure = st.slider('Diastolic Blood Pressure', min_value=0.0, max_value=1.0, step=0.01)
        Triglycerides = st.slider('Triglycerides', min_value=0.0, max_value=1.0, step=0.01)
        HbA1c = st.slider('HbA1c', min_value=0.0, max_value=1.0, step=0.01)
        LDL_Cholesterol = st.slider('LDL Cholesterol', min_value=0.0, max_value=1.0, step=0.01)
        HDL_Cholesterol = st.slider('HDL Cholesterol', min_value=0.0, max_value=1.0, step=0.01)
        ALT = st.slider('ALT', min_value=0.0, max_value=1.0, step=0.01)
        AST = st.slider('AST', min_value=0.0, max_value=1.0, step=0.01)
        Heart_Rate = st.slider('Heart Rate', min_value=0.0, max_value=1.0, step=0.01)
        Creatinine = st.slider('Creatinine', min_value=0.0, max_value=1.0, step=0.01)
        Troponin = st.slider('Troponin', min_value=0.0, max_value=1.0, step=0.01)
        C_reactive_Protein = st.slider('C-reactive Protein', min_value=0.0, max_value=1.0, step=0.01)
    
    st.markdown("""
    <style>
        /* Ajuster la largeur des sliders */
        .st-df div[data-baseweb="slider"] .st-cc {
            width: 100%; /* Ajustez la largeur selon vos besoins */
        }
    </style>
    """, unsafe_allow_html=True)

    col = df.drop(columns=['Disease', 'label'], axis=1)
    prediction_reg = model.predict([[Glucose, Cholesterol, Hemoglobin, Platelets, White_Blood_Cells, Red_Blood_Cells, Hematocrit, Mean_Corpuscular_Volume, Mean_Corpuscular_Hemoglobin, Mean_Corpuscular_Hemoglobin_Concentration, Insulin, BMI, Systolic_Blood_Pressure, Diastolic_Blood_Pressure, Triglycerides,HbA1c, LDL_Cholesterol, HDL_Cholesterol, ALT, AST, Heart_Rate, Creatinine, Troponin, C_reactive_Protein]])
    dico = {0: 'Anemia', 1: 'Diabetes', 3: 'Thalasse', 2: 'Healthy', 4: 'Thromboc'}
    
    if st.button("Afficher la prédiction"):
        if prediction_reg != 2:
            st.write("Interprétation de la prise de sang montre un risque de ", dico[prediction_reg[0]])
        else:
            st.write("Les resultats indiquent que vous ête en bonne santé.")
