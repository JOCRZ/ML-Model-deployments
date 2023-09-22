import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data//Salary_Data.csv')

x = np.array(data['YearsExperience']).reshape(-1,1)
lr = LinearRegression()
lr.fit(x,np.array(data['Salary']))

st.title("Salary Predictor")

nav = st.sidebar.radio("Navigation",['Home','Prediction'])
if nav == 'Home':
    st.image('data//salary.jpg')
    if st.checkbox('Show Table'):
        st.table(data)

    st.markdown(""" Scatter plot""")
    val = st.slider("Filter data using years",0,11)
    data = data.loc[data["YearsExperience"]>= val]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=data, x="YearsExperience", y="Salary", ax=ax)
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    st.pyplot(fig)


if nav == 'Prediction':
    st.header("Know your Salary")
    val = st.number_input("Enter you exp",0.00,12.00,step=2.5)
    val = np.array(val).reshape(1,-1)
    pred = lr.predict(val)[0]

    if st.button("Predict"):
        st.success(f'Your predicted salary is  {round(pred)}')



