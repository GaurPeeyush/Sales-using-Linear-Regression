import pandas as pd
import streamlit as st
import joblib

st.title('First app')

#st.sidebar.title('adding 2 numbers')
#st.sidebar.slider('Pick a number', 0, 10)

model = joblib.load('reg_model.pkl')

num1 = st.number_input("Enter Price")
num2 = st.number_input("Enter AdvCost")

input = pd.DataFrame({
    'Price': [num1],
    'AdvCost': [num2]
})

output = model.predict(input)

st.metric('Sales', output)


