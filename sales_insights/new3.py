import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from textblob import TextBlob
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(page_title="Sales Insights", layout="centered")

# Sample in-memory user database (replace with a real database in production)
users_db = {
    "Maha": {"password": "Streamlit@24"}  # Sample user for testing
}

# Add login function


def login():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users_db and users_db[username]['password'] == password:
            st.success("Login successful!")
            st.session_state["logged_in"] = True
            return True
        else:
            st.error("Invalid credentials. Please try again.")
            return False
    return False

# Add signup function


def signup():
    st.title("Sign Up Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if password != confirm_password:
            st.error("Passwords do not match.")
        elif username in users_db:
            st.error("Username already exists. Please choose another one.")
        else:
            # Add the new user to the in-memory database
            users_db[username] = {'password': password}
            st.success("Sign up successful! You can now log in.")

# Function to display login/signup selection


def show_login_signup():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        choice = st.selectbox("Choose action", ["Login", "Sign Up"])
        if choice == "Login":
            return login()
        elif choice == "Sign Up":
            signup()
            return False
    else:
        return True

# Sentiment analysis function


def analyze_sentiment(text):
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0:
        return "Positive"
    elif blob.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Save feedback to database


def save_feedback_to_db(name, phone, feedback, sentiment):
    c.execute("INSERT INTO feedback (name, phone, feedback, sentiment)"
              "VALUES (?, ?, ?, ?)", (name, phone, feedback, sentiment))
    conn.commit()

# Load feedback from database


def load_feedback_from_db():
    c.execute("SELECT name, phone, feedback, sentiment FROM feedback")
    rows = c.fetchall()
    return pd.DataFrame(
        rows, columns=["Name", "Phone", "Feedback", "Sentiment"])

# Customer feedback section


def customer_feedback_section():
    st.subheader("Customer Feedback")
    with st.form("feedback_form"):
        name = st.text_input("Your Name")
        phone = st.text_input("Your Phone Number")
        feedback = st.text_area("Your Feedback")
        submit_feedback = st.form_submit_button("Submit Feedback")
        if submit_feedback:
            if feedback:
                sentiment = analyze_sentiment(feedback)
                save_feedback_to_db(name, phone, feedback, sentiment)
                st.success("Thank you for your feedback!")
            else:
                st.warning("Please enter your feedback before submitting.")

    feedback_data = load_feedback_from_db()
    if not feedback_data.empty:
        st.write("### Customer Feedback and Sentiment Analysis")
        st.dataframe(feedback_data)
        sentiment_counts = feedback_data["Sentiment"].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values,
                    ax=ax, palette="coolwarm")
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

# File upload and data visualization section


def upload_and_analysis():
    st.title("Sales and Performance Analytics")
    st.subheader(
        "A Real-Time Data Insight Platform for Revenue and Customer Insights")

    # Sidebar for file upload
    st.sidebar.header("Upload your CSV/Excel file")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        # Load CSV or Excel files
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)

        st.write("### Data Visualizations")

        # Display basic data information
        st.write("#### Dataset Overview")
        st.write(df.head())
        st.write("Number of rows:", df.shape[0])
        st.write("Number of columns:", df.shape[1])

        # Visualization options
        st.write("#### Select Visualization Type")
        chart_type = st.selectbox("Choose chart type", ["Bar Chart",
                                                        "Line Plot",
                                                        "Scatter Plot",
                                                        "Pie Chart"])

        if chart_type == "Bar Chart":
            x_axis = st.selectbox("Select X-axis", df.columns)
            y_axis = st.selectbox("Select Y-axis", df.columns)
            if x_axis and y_axis:
                fig = plt.figure(figsize=(10, 6))
                sns.barplot(data=df, x=x_axis, y=y_axis)
                plt.title(f"Bar Chart: {y_axis} vs {x_axis}")
                st.pyplot(fig)

        elif chart_type == "Line Plot":
            x_axis = st.selectbox("Select X-axis", df.columns)
            y_axis = st.selectbox("Select Y-axis", df.columns)
            if x_axis and y_axis:
                fig = plt.figure(figsize=(10, 6))
                sns.lineplot(data=df, x=x_axis, y=y_axis)
                plt.title(f"Line Plot: {y_axis} vs {x_axis}")
                st.pyplot(fig)

        elif chart_type == "Scatter Plot":
            x_axis = st.selectbox("Select X-axis", df.columns)
            y_axis = st.selectbox("Select Y-axis", df.columns)
            color = st.selectbox("Select color", [None] + list(df.columns))
            if x_axis and y_axis:
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color)
                st.plotly_chart(fig)

        elif chart_type == "Pie Chart":
            pie_column = st.selectbox(
                "Select Column for Pie Chart", df.columns)
            if pie_column:
                pie_data = df[pie_column].value_counts()
                fig = plt.figure(figsize=(8, 8))
                plt.pie(pie_data, labels=pie_data.index,
                        autopct='%1.1f%%', startangle=90)
                plt.title(f"Pie Chart of {pie_column}")
                st.pyplot(fig)

# Sales prediction section


def sales_prediction(df):
    st.write("### Sales Prediction Model")

    # Ensure 'Sales' and 'Revenue' columns are in the dataset
    if "Sales" in df.columns and "Revenue" in df.columns:
        X = df[["Revenue"]]
        y = df["Sales"]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Show predictions on test data
        st.write("#### Predictions on Test Data")
        y_pred = model.predict(X_test)
        test_data = pd.DataFrame({"Actual Sales": y_test,
                                  "Predicted Sales": y_pred})
        st.write(test_data.head())

        # Display model metrics
        st.write("#### Model Performance")
        st.write("Model Coefficient:", model.coef_[0])
        st.write("Model Intercept:", model.intercept_)

        # Plot predictions vs actual values
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, color="blue", label="Predicted vs Actual")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                "k--", lw=2)
        ax.set_xlabel("Actual Sales")
        ax.set_ylabel("Predicted Sales")
        ax.set_title("Prediction vs Actual")
        st.pyplot(fig)

# Connect to SQLite database


conn = sqlite3.connect('feedback.db')
c = conn.cursor()

# Create a feedback table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                phone TEXT,
                feedback TEXT,
                sentiment TEXT
            )''')
conn.commit()

# Main application logic
if show_login_signup():
    # Sidebar for navigation
    page = st.sidebar.selectbox("Choose a page", ["Sales Analytics",
                                                  "Customer Feedback",
                                                  "Sales Prediction"])

    # Page navigation logic
    if page == "Sales Analytics":
        upload_and_analysis()
    elif page == "Customer Feedback":
        customer_feedback_section()
    elif page == "Sales Prediction":
        uploaded_file = st.sidebar.file_uploader(
            "Upload your sales data for prediction", type=["csv", "xlsx"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                sales_prediction(df)
            elif uploaded_file.name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
                sales_prediction(df)
