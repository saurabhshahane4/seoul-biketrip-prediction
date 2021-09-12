# pip install streamlit
import streamlit as st
from model import predict_duration
import numpy as np

st.set_page_config(page_title="Seoul Bike Trip Duration Prediction App",
                   page_icon="🛴", layout="wide")


with st.form("prediction_form"):

    st.header("Enter the Deciding Factors:")

    distance = st.number_input("Distance: ", value=0, format="%d")
    haversine = st.number_input("Haversine: ")
    phour = st.slider("Pickup Hour: ", 0, 23, value=0, format="%d")
    pmin = st.slider("Pickup Minute: ", 0, 59, value=0, format="%d")
    dhour = st.slider("Dropoff Hour: ", 0, 23, value=0, format="%d")
    dmin = st.slider("Dropoff Minute: ", 0, 59, value=0, format="%d")
    temp = st.number_input("Temp: ")
    humid = st.number_input("Humid: ")
    solar = st.number_input("Solar: ")
    dust = st.number_input("Dust: ")

    submit_val = st.form_submit_button("Predict Duration")

if submit_val:
    # If submit is pressed == True
    attribute = np.array([distance, haversine, phour,
                        pmin, dhour,
                        dmin, temp,
                        humid, solar, dust]).reshape(1,-1)


    if attribute.shape == (1,10):
        print("attrubutes valid")
        

        value = predict_duration(attributes= attribute)


        st.header("Here are the results:")
        st.success(f"The Duration predicted is {value} mins")