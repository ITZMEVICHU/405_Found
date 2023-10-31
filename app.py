import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Add a title and logo
st.set_page_config(
    page_title="Spending Limit Predictor",
    page_icon="🛒",  # You can use a custom icon or emoji here
)

# Display the logo
st.image("logo.png")


# Title and description
st.title("Spending Limit Predictor")
st.write("This app predicts spending limits based on different sets of features.")

# Create a sidebar for the menu
menu_selection = st.sidebar.selectbox("Menu", ["Problem Statement", "ROI", "Visualization"])

if menu_selection == "Problem Statement":
    # Problem Statement section
    st.header("Problem Statement")
    st.markdown(
        """
        <div style="text-align: justify; text-justify: inter-word;">
        Marketing managers at a retail company wanted to develop a targeted marketing plan and demonstrate a return on investment (ROI) for their marketing spend. To predict the customer's spending limit based on their earnings and earning potential, the marketing team turned to their data science team for a machine learning application (MLOps). In a meeting with the management team of the retail company, it was suggested that the machine learning model be trained and tested with a variety of data sets as needed.

        As a result, business users should be able to upload training data to MLOps and select features through the user interface (UI). Users should also be able to upload and preview test data to test the model. Explanations AI functionality should be implemented by MLOps to help business users understand what the model outcomes mean. To simplify and better understand model outcomes, business users requested visual data analysis functionality.
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.subheader("Upload Training Data:")
    uploaded_train_file = st.file_uploader("Upload a CSV file for training:", type=["csv"])

    if uploaded_train_file is not None:
        # Display uploaded training data
        st.write("Uploaded Training Data:")
        train_data = pd.read_csv(uploaded_train_file)
        st.write(train_data)

        # Select features using a dropdown input
        selected_features = st.selectbox(
            "Select Features for Modeling:",
            ("Earnings and Earning potential", "Earnings and Savings", "Earnings and CreditScore")
        )

        if selected_features == "Earnings and Earning potential":
            features = ['earnings', 'earning_potential']
        elif selected_features == "Earnings and Savings":
            features = ['earnings', 'Savings']
        elif selected_features == "Earnings and CreditScore":
            features = ['earnings', 'CreditScore']

        X = train_data[features]
        y = train_data['spending_limit']

        # Training the model
        model = RandomForestRegressor()
        model.fit(X, y)

        # Upload a CSV file for testing
        st.subheader("Upload Testing Data:")
        uploaded_test_file = st.file_uploader("Upload a CSV file for testing:", type=["csv"])

        if uploaded_test_file is not None:
            # Display uploaded testing data
            st.write("Uploaded Testing Data:")
            test_data = pd.read_csv(uploaded_test_file)  # Define test_data here
            st.write(test_data)

            # Predict spending limits for testing data
            X_test = test_data[features]
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

elif menu_selection == "Visualization":
    # Visualization section
    st.header("Visualization")
    st.write("This section provides visualizations based on the features tested.")
    st.subheader("Upload Testing Data:")
    uploaded_test_file = st.file_uploader("Upload a CSV file for testing:", type=["csv"])
    test_data = pd.read_csv(uploaded_test_file)
    # Add visualization code here
    import matplotlib.pyplot as plt

    if 'test_data' in locals():  # Check if test_data is defined
        # Example scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(test_data['earnings'], test_data['predicted_spending_limit'], label="Predicted Spending Limit", color='blue')
        plt.scatter(test_data['earnings'], test_data['spending_limit'], label="Actual Spending Limit", color='red', marker='x')
        plt.xlabel("Earnings")
        plt.ylabel("Spending Limit")
        plt.title("Earnings vs. Spending Limit")
        plt.legend()

        # Display the plot in Streamlit
        st.pyplot(plt)
    else:
        st.warning("Please upload and process testing data to generate visualizations.")
    # You can add your visualizations here.

