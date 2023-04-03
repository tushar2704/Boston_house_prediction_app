# Boston House Price Prediction Application

# Importing the required libraries
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

# Application body
st.title("Boston House Price Prediction Application")
st.write(" This application predicts the Boston House Prices")
# Loading the dataset
boston = datasets.load_boston()
# Defining feature and target variable
X=pd.DataFrame(boston.data, columns=boston.feature_names)
y=pd.DataFrame(boston.target, columns=['MEDV'])

# Sidebar
st.sidebar.header("Please specify input parameters")

# Defining user_input_features

def user_input_features():
    CRIM =st.sidebar.slider("CRIM", X.CRIM.min(), X.CRIM.max(), X.CRIM.mean())# Format .slider("",min val, max val , default val)
    ZN =st.sidebar.slider("ZN", X.ZN.min(), X.ZN.max(), X.ZN.mean())
    INDUS =st.sidebar.slider("INDUS", X.INDUS.min(), X.INDUS.max(), X.INDUS.mean())
    CHAS =st.sidebar.slider("CHAS", X.CHAS.min(), X.CHAS.max(), X.CHAS.mean())
    NOX =st.sidebar.slider("NOX", X.NOX.min(), X.NOX.max(), X.NOX.mean())
    RM =st.sidebar.slider("RM", X.RM.min(), X.RM.max(), X.RM.mean())
    AGE =st.sidebar.slider("AGE", X.AGE.min(), X.AGE.max(), X.AGE.mean())
    DIS =st.sidebar.slider("DIS", X.DIS.min(), X.DIS.max(), X.DIS.mean())
    RAD =st.sidebar.slider("RAD", X.RAD.min(), X.RAD.max(), X.RAD.mean())
    TAX =st.sidebar.slider("TAX", X.TAX.min(), X.TAX.max(), X.TAX.mean())
    PTRATIO =st.sidebar.slider("PTRATIO", X.PTRATIO.min(), X.PTRATIO.max(), X.PTRATIO.mean())
    B =st.sidebar.slider("B", X.B.min(), X.B.max(), X.B.mean())
    LSTAT =st.sidebar.slider("LSTAT", X.LSTAT.min(), X.LSTAT.max(), X.LSTAT.mean())
    
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    
    features =pd.DataFrame(data, index=[0])
    return features

df=user_input_features()

# Main Panel

st.header("Selected input parameters")
st.write(df)
st.write("___")

# Building Regression Model
model = RandomForestRegressor()
model.fit(X, y)

# Predictions
predictions =model.predict(df)

# Predication display on App
st.header("Predication of MEDV")
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