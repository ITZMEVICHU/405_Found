import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Title and description
st.title("Spending Limit Predictor")
st.write("This app predicts spending limits based on earnings and earning potential.")

# Upload a CSV file for training
st.subheader("Upload Training Data:")
uploaded_train_file = st.file_uploader("Upload a CSV file for training:", type=["csv"])

if uploaded_train_file is not None:
    # Display uploaded training data
    st.write("Uploaded Training Data:")
    train_data = pd.read_csv(uploaded_train_file)
    st.write(train_data)

    # Training the model
    X = train_data[['earnings', 'earning_potential']]
    y = train_data['spending_limit']
    
    model = RandomForestRegressor()
    model.fit(X, y)

    # Upload a CSV file for testing
    st.subheader("Upload Testing Data:")
    uploaded_test_file = st.file_uploader("Upload a CSV file for testing:", type=["csv"])

    if uploaded_test_file is not None:
        # Display uploaded testing data
        st.write("Uploaded Testing Data:")
        test_data = pd.read_csv(uploaded_test_file)
        st.write(test_data)

        # Predict spending limits for testing data
        X_test = test_data[['earnings', 'earning_potential']]
        y_pred = model.predict(X_test)

        # Add predictions to the testing data
        test_data['predicted_spending_limit'] = y_pred

        # Display the testing data with predictions in a table
        st.write("Testing Data with Predicted Spending Limit:")
        st.table(test_data)

        # Calculate Mean Squared Error and R-squared for evaluation
        y_true = train_data['spending_limit']
        y_train, y_test = train_test_split(y_true, test_size=0.2, random_state=0)
        y_test_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)

        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"R-squared (R2): {r2}")
