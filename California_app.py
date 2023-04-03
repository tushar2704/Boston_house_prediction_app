# California House Price Prediction Application

# Importing the required libraries
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
st.set_option('deprecation.showPyplotGlobalUse', False)
# Application body
st.title("California House Price Prediction Application")
st.write(" This application predicts the California House Prices")
# Loading the dataset
housing = fetch_california_housing()
# Defining feature and target variable
X=pd.DataFrame(housing.data, columns=housing.feature_names)
y=pd.DataFrame(housing.target, columns=['MedHouseVal'])
y = y.values.ravel()
# Sidebar
st.sidebar.header("Please specify input parameters")

# Defining user_input_features

MedInc = st.sidebar.slider("MedInc",int(X.MedInc.min()), int(X.MedInc.max()), int(X.MedInc.mean()),key="MedInc")
HouseAge =st.sidebar.slider("HouseAge", int(X.HouseAge.min()), int(X.HouseAge.max()), int(X.HouseAge.mean()),key="HouseAge")
AveRooms =st.sidebar.slider("AveRooms", int(X.AveRooms.min()), int(X.AveRooms.max()), int(X.AveRooms.mean()),key="AveRooms")
AveBedrms =st.sidebar.slider("AveBedrms", int(X.AveBedrms.min()), int(X.AveBedrms.max()), int(X.AveBedrms.mean()),key="AveBedrms")
Population =st.sidebar.slider("Population", int(X.Population.min()), int(X.Population.max()), int(X.Population.mean()),key="Population")
AveOccup =st.sidebar.slider("AveOccup", int(X.AveOccup.min()), int(X.AveOccup.max()), int(X.AveOccup.mean()),key="AveOccup")
Latitude =st.sidebar.slider("Latitude", int(X.Latitude.min()), int(X.Latitude.max()), int(X.Latitude.mean()),key="Latitude")
Longitude =st.sidebar.slider("Longitude", int(X.Longitude.min()), int(X.Longitude.max()), int(X.Longitude.mean()),key="Longitude")

def user_input_features():
    
    data = {'MedInc': MedInc,
            'HouseAge': HouseAge,
            'AveRooms': AveRooms,
            'AveBedrms':AveBedrms,
            'Population': Population,
            'AveOccup': AveOccup,
            'Latitude': Latitude,
            'Longitude': Longitude}
    
    features =pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

st.header("Selected input parameters")
st.write(user_input_features())
st.write("___")

# Building Regression Model
model = RandomForestRegressor()
model.fit(X, y)

# Predictions
predictions =model.predict(df)

# Predication display on App
st.header("Predication of MedHouseVal")
st.write(predictions+" in 1000s")
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
