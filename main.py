import streamlit as st
import json
import sqlite3
import hashlib

# Function to create the Users table
def create_users_table():
    conn = sqlite3.connect('gym_app.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS Users (
        id INTEGER PRIMARY KEY,
        username TEXT NOT NULL,
        password TEXT NOT NULL
    );''')
    conn.commit()
    conn.close()

# Function to insert a new user
def insert_user(username, password):
    conn = sqlite3.connect('gym_app.db')
    c = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()  # Hash the password
    c.execute("INSERT INTO Users (username, password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()
    conn.close()

# Function to check login credentials
def check_login(username, password):
    conn = sqlite3.connect('gym_app.db')
    c = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()  # Hash the password
    c.execute("SELECT * FROM Users WHERE username=? AND password=?", (username, hashed_password))
    user = c.fetchone()
    conn.close()
    return user

# Create Users table
create_users_table()

# Initialize session state
if 'user_authenticated' not in st.session_state:
    st.session_state.user_authenticated = False

# Load gym exercise data
with open('exercise.json', 'r') as exercise_file:
    exercises = json.load(exercise_file)

# Load gym meal data
with open('gym_meals.json', 'r') as meal_file:
    meals = json.load(meal_file)

# Page layout
st.set_page_config(page_title="Gym App", layout="wide")

# Logo
logo_path = "gym.jpg"  # replace with your logo path
st.image(logo_path, width=200)

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Home", "Exercise List", "Meal List", "Pricing", "Sign Up", "Login", "Logout"])

# User authentication
authenticated = st.session_state.user_authenticated

# Home page
if page == "Home":
    st.title("Welcome to the Gym App!")
    st.write("Explore gym exercises and meals.")

# Exercise List page
elif page == "Exercise List" and authenticated:
    st.title("Gym Exercise List")
    st.write("Browse a variety of gym exercises.")

    # Display exercise list
    st.write("### Exercise List")
    for exercise in exercises:
        st.write(f"- {exercise}")

# Meal List page
elif page == "Meal List" and authenticated:
    st.title("Gym Meal List")
    st.write("Discover nutritious gym meals.")

    # Display meal list
    st.write("### Meal List")
    for meal in meals:
        st.write(f"- {meal['name']} (Category: {meal['category']}")

# Pricing page
elif page == "Pricing" and authenticated:
    st.title("Pricing")
    st.write("Choose a plan that fits your needs:")
    st.write("- Basic Plan: $10/month")
    st.write("- Premium Plan: $20/month")
    st.write("- Ultimate Plan: $30/month")

    # Add payment form
    st.write("### Payment Information")
    payment_method = st.radio("Select Payment Method", ["Credit Card", "UPI"])

    if payment_method == "Credit Card":
        credit_card_number = st.text_input("Credit Card Number", type="password")
        expiration_date = st.text_input("Expiration Date (MM/YY)", type="password")
        cvv = st.text_input("CVV", type="password")

        if st.button("Make Payment"):
            # Process payment logic goes here
            st.success("Payment successful!")

    elif payment_method == "UPI":
        upi_id = st.text_input("UPI ID")
        if st.button("Make Payment"):
            # Process UPI payment logic goes here
            st.success("Payment successful!")

# Sign Up page
elif page == "Sign Up":
    st.title("Sign Up")
    new_username = st.text_input("Username")
    new_password = st.text_input("Password", type="password", key="signup")
    if st.button("Sign Up"):
        # Insert new user into the Users table
        insert_user(new_username, new_password)
        st.success(f"User {new_username} signed up successfully. Please log in.")

# Login page
elif page == "Login":
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password", key="login")
    if st.button("Login"):
        # Check user credentials
        user = check_login(username, password)
        if user:
            st.session_state.user_authenticated = True
            st.success(f"Logged in as {username}.")
        else:
            st.error("Invalid username or password.")

# Logout
elif page == "Logout":
    st.session_state.user_authenticated = False
    st.success("Logged out successfully.")

# Handle authentication
if authenticated is False and page not in ["Sign Up", "Login"]:
    st.warning("You need to log in to access the Exercise List, Meal List, and Pricing pages.")
    st.sidebar.radio("Navigation", ["Home", "Sign Up", "Login"])
