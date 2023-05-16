import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_model(data):
    # Split the data into features (X) and target variable (y)
    X = data[['company_name', 'job_title', 'min_experience']]
    y = data['avg_salary']

    # Convert categorical variables to numerical using one-hot encoding
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)


    return model,X_encoded

# Call the train_model function with the provided data
data = pd.read_csv('data_scientist_salaries_dataset.csv')
model,X_encoded = train_model(data)


def main():
    # Set the title and sidebar
    st.title('Salary Prediction for Data Scientist')
    st.title('Options')

    # Load the data
    data = pd.read_csv('data_scientist_salaries_dataset.csv')

    # Train the model
    model,X_encoded = train_model(data)

    # Add input fields for the features
    company = st.selectbox('Company', data['company_name'].unique())
    job_title = st.selectbox('Job Title', data['job_title'].unique())
    experience = st.selectbox('Minimum Experience (in years)', data['min_experience'].unique())

    # Make a prediction for the input values
    input_data = pd.DataFrame([[company, job_title, experience]], columns=['company_name', 'job_title', 'min_experience'])
    input_data_encoded = pd.get_dummies(input_data, drop_first=True)
    input_data_encoded = input_data_encoded.reindex(columns=X_encoded.columns, fill_value=0)
    prediction = model.predict(input_data_encoded)

    # Display the prediction result
    st.write('Predicted Avg Salary (in Lakhs):', round(prediction[0], 2))

    #subheader
    st.write('By: :blue[Mayuresh Harihar]')

    btn_click = st.button("Connect with me")

    if btn_click == True:
        st.write(":blue[LinkedIn]: (https://www.linkedin.com/in/mayuresh-harihar/)")

        st.write(":green[GitHub]: (https://github.com/hariharmayuresh)")

        st.write(":red[Instagram]: (https://www.instagram.com/hariharmayuresh/)")



if __name__ == '__main__':
    main()