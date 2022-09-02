import pickle
import numpy as np
import streamlit as st


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data['model']
le_country = data['le_country']
le_education = data['le_education']

def show_predict_page():
    st.title('Software Engineer Salary Prediction')
    st.write('Based on 2020 stackoverflow data analytics')
    countries = (
        'United States',
        'India',
        'United Kingdom',
        'Germany',
        'Canada',
        'Brazil',
        'France',
        'Spain',
        'Australia',
        'Netherlands',
        'Poland',
        'Italy',
        'Russina federation',
        'Sweden',
    )
    education = (
        'Less than a Bachelors',
        'Bachelor`s degree',
        'Master`s degree',
        'Post grad',
    )
    
    country = st.selectbox('Country',countries)
    education = st.selectbox('Level of Education',education)
    
    experience = st.slider('Years of Experience', 0, 50, 2)
    
    calc = st.button('Calculate Salary')
    if calc:
        x = np.array([[country, education, experience]])
        x[:, 0] = le_country.transform(x[:,0])
        x[:, 1] = le_education.transform(x[:,1])
        x = x.astype(float)
        
        salary = regressor.predict(x)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}") 