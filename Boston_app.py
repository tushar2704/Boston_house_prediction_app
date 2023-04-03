# California House Price Prediction Application

# Importing the required libraries
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# Application body
st.title("California House Price Prediction Application")
st.write(" This application predicts the California House Prices")
# Loading the dataset
housing = fetch_california_housing()
# Defining feature and target variable
X=pd.DataFrame(housing.data, columns=housing.feature_names)
y=pd.DataFrame(housing.target, columns=['MedHouseVal'])
y=y.values
# Sidebar
st.sidebar.header("Please specify input parameters")

# Defining user_input_features

def user_input_features():
    MedInc = st.sidebar.slider("MedInc", X.MedInc.min(), X.MedInc.max(), X.MedInc.mean())
    HouseAge =st.sidebar.slider("HouseAge", X.HouseAge.min(), X.HouseAge.max(), X.HouseAge.mean())
    AveRooms =st.sidebar.slider("AveRooms", X.AveRooms.min(), X.AveRooms.max(), X.AveRooms.mean())
    Population =st.sidebar.slider("Population", X.Population.min(), X.Population.max(), X.Population.mean())
    AveOccup =st.sidebar.slider("AveOccup", X.AveOccup.min(), X.AveOccup.max(), X.AveOccup.mean())
    Latitude =st.sidebar.slider("Latitude", X.Latitude.min(), X.Latitude.max(), X.Latitude.mean())
    Longitude =st.sidebar.slider("Longitude", X.Longitude.min(), X.Longitude.max(), X.Longitude.mean())
    
    
    data = {'MedInc': MedInc,
            'HouseAge': HouseAge,
            'AveRooms': AveRooms,
            'Population': Population,
            'AveOccup': AveOccup,
            'Latitude': Latitude,
            'Longitude': Longitude}
    
    features =pd.DataFrame(data, index=[0])
    return features

#df = user_input_features()

# Main Panel

st.header("Selected input parameters")
st.write(df)
st.write("___")

# Building Regression Model
model = RandomForestRegressor()
model.fit(X, y)

# Predictions
predictions =model.predict(user_input_features())

# Predication display on App
st.header("Predication of MedHouseVal")
st.write(predictions)
st.write("---")

# Explaining the model's prediction using SHAP values
explainer =shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Shap plot
st.header("Feature Importance")
plt.title("Feature importance based on SHAP values")
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write("---")

# Shap values
plt.title("Feature importance based on SHAP values(BAR)")
shap.summary_plot(shap_values, X, plot_type='bar')
st.pyplot(bbox_inches='tight')
