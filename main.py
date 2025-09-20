import streamlit as st
import joblib
import numpy as np

model = joblib.load('beans_best_model_rf.joblib')
scaler = joblib.load('beans_scaler.joblib')
le = joblib.load('beans_label_encoder.joblib')

st.title("Dry Beans Classifier")
st.subheader("*By* :rainbow[Swayam Sodha]")
st.write("Enter bean physical measurements:")

# Example for dynamic inputs based on dataset features
features = ['Area','Perimeter','MajorAxisLength','MinorAxisLength','AspectRation','Eccentricity','ConvexArea','EquivDiameter','Extent','Solidity','roundness','Compactness','ShapeFactor1','ShapeFactor2','ShapeFactor3','ShapeFactor4']
vals = []
for f in features:
    vals.append(st.number_input(f, value=0.0))

if st.button('Predict'):
    x = np.array(vals).reshape(1,-1)
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)
    st.success(f'Predicted class: {le.inverse_transform(pred)[0]}')

