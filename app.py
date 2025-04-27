import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import joblib

#==============================================================================================|

# Customizing header
st.markdown("""
 <style>
        .header2 {
            font-size: 40px;
            color: #4CAF50;
            font-weight: bold;
            cursor: pointer;
            text-align:center;
        
           transition: all 0.3s ease;  /* Smooth transition */
        }
        .header2:hover {
            color: #FF5733;  /* Change to a different color on hover */
            text-decoration: underline;  /* Underline text on hover */
            transform: scale(1.1);  /* Increase font size on hover */
        }
        .main{
            background-color: #4CAF50;  /* Light blue color */
            
        }
    </style>
    <h1  class="header2">Welcome to the AI & ML Playground</h1>
    <h3  style="text-align: center; color:white;background-color:#4CAF50">Explore, Experiment, and Learn with Interactive Models!</h3>
    
    """, unsafe_allow_html=True)

# Main content area for your ML models (you can replace this with your actual app content)
st.write("Here, you can explore various machine learning models like Regression, Classification, and more!")

#--------------------------------------current time----------------------------------------------------|


# Get the current date and time
current_time = datetime.now()

# Display the current date and time in the top-right corner

st.markdown(f"""
    <style>
        .datetime {{
        position: absolute;
            top: 10px;
            right: 10px;
            font-size: 18px;
            font-weight: bold;
            color:#EE82EE;
            display: inline-block;
            animation: fadeInOut 5s infinite;
        }}

        @keyframes fadeInOut {{
            0% {{ opacity: 0; }}
            50% {{ opacity: 1; }}
            100% {{ opacity: 0; }}
        }}
    </style>
    <div class="datetime">Current Date and Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}</div>
""", unsafe_allow_html=True)
# In markdown HTML and inline CSS included

#----------------------------------------sidebar--------------------------------------------------|

# Sidebar content
st.sidebar.title("AI & ML Playground")
st.sidebar.header("Explore Machine Learning Models")
st.sidebar.text("Select the model from below.")

# Add options to sidebar (you can modify these as per your project)
model_choice = st.sidebar.selectbox(
    "Choose a Model to Explore",
    ["Not Yet choose the model","Regression", "Classification", "Recommendation", "NLP","Create a Report"]
)


#------------------------------------------------------------------------------------------|
# Main content area
if model_choice=="Not Yet choose the model":
    import streamlit as st
    import random
    import time

# Set Page Title
    st.title("🎮 Fun Game Hub")

# Sidebar for Game Selection
    game = st.radio("Choose a Game:", ["Rock, Paper, Scissors", "Number Guessing"])

# ------------------- 1️⃣ Rock, Paper, Scissors -------------------
    if game == "Rock, Paper, Scissors":
        st.subheader("✊📄✂️ Rock, Paper, Scissors Game")

        choices = ["Rock", "Paper", "Scissors"]
        user_choice = st.selectbox("Choose your move:", choices)
        computer_choice = random.choice(choices)

        if st.button("Play!"):
            st.write(f"🤖 Computer chose: {computer_choice}")
            if user_choice == computer_choice:
               st.write("⚖️ It's a tie!")
            elif (user_choice == "Rock" and computer_choice == "Scissors") or \
                (user_choice == "Paper" and computer_choice == "Rock") or \
                (user_choice == "Scissors" and computer_choice == "Paper"):
                st.success("🎉 You Win!")
            else:
               st.error("😞 You Lose!")

# ------------------- 2️⃣ Number Guessing -------------------
    elif game == "Number Guessing":
        st.subheader("🔢 Number Guessing Game")
        number = random.randint(1, 10)

        user_guess = st.number_input("Guess a number between 1 and 10:", min_value=1, max_value=100, step=1)
        if st.button("Check"):
            if user_guess == number:
               st.success("🎉 Correct! You guessed the number!")
            elif user_guess > number:
               st.warning("📉 Too high! Try again.")
            else:
                st.warning("📈 Too low! Try again.")



    
 
elif model_choice == "Regression":
    st.title("Regression Models")
    st.write("Here are two regression models you can explore:")

    # Sub-selection for Regression Models (Weather Forecasting or Car Price Prediction)
    regression_choice = st.radio(
        "Select Regression Model",
        ["Select a model", "🏠 House Price Prediction", "🚗Car Price Prediction"]
    )

    if regression_choice == "🏠 House Price Prediction":
        
        # Load the saved house price model
        model = joblib.load('house_price_model.pkl')

        st.title("🏠 House Price Prediction")
        st.write("Enter details below to predict the house price:")

        #  inputs
        size = st.number_input("Enter house size in sqft:", step=100, value=1000)
        bedrooms = st.number_input("Enter number of bedrooms:", step=1, value=2)

       # Predict button
        if st.button("Predict Price"):
           
           input_data = np.array([[size, bedrooms]])
           prediction = model.predict(input_data)
           st.write(f"💰 Predicted Price: $.{prediction[0]:,.2f}")
        





 #==========================================Car Price Prediction====================================================================|       

    elif regression_choice == "🚗Car Price Prediction":
        st.subheader("🚗Car Price Prediction")
        st.write("This model predicts the price of a car based on features like make, model, mileage, and other attributes.")
        # Add interactivity for Car Price Prediction model here
        # For now, just showing a placeholder message
        st.write("You can play with the car price prediction data here (this is a placeholder for now).")
       
        # Load the model
        model = joblib.load('car_price_prediction_model.pkl')

# Define the Streamlit app
        st.title("Car Price Prediction App")

        st.header("Enter Car Details")

# Input fields for user data
        year = st.number_input("Year of Purchase", min_value=2000, max_value=2023, step=1, value=2015)
        present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1, value=5.0)
        kms_driven = st.number_input("KMs Driven", min_value=0, step=500, value=30000)
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
        seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
        transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
        owner = st.number_input("Number of Previous Owners", min_value=0, max_value=3, step=1, value=0)

# Encoding user inputs
        fuel_mapping = {"Petrol": 0, "Diesel": 1, "CNG": 2}
        seller_mapping = {"Dealer": 0, "Individual": 1}
        trans_mapping = {"Manual": 0, "Automatic": 1}

        user_data = pd.DataFrame({
       'Year': [year],
    'Present_Price': [present_price],
    'Kms_Driven': [kms_driven],
    'Fuel_Type': [fuel_mapping[fuel_type]],
    'Seller_Type': [seller_mapping[seller_type]],
    'Transmission': [trans_mapping[transmission]],
    'Owner': [owner]
       })

# Prediction button
        if st.button("Predict Selling Price"):
           prediction = model.predict(user_data)
           st.subheader("Predicted Selling Price")
           st.write(f"🚗 ₹ {prediction[0]:.2f} lakhs")
    
    
        else:
           st.write("Enter car details in the sidebar and click 'Predict Selling Price'.")


    else:
        st.write("Please select a regression model to explore.")
# ================================================================================================|
elif model_choice == "Classification":
    import streamlit as st
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, confusion_matrix

# Load model and encofromders
    model_path = "EAgradient_boosting_model.pkl"
    encoder_path = "EAlabel_encoders.pkl"

    gb_model = joblib.load(model_path)
    label_encoders = joblib.load(encoder_path)

# Define UI layout
    st.title("Employee Attrition Prediction")
    st.sidebar.header("Enter Employee Details")

# Input fields
    age = st.sidebar.number_input("Age", min_value=18, max_value=65, value=30)
    years_at_company = st.sidebar.number_input("Years at Company", min_value=0, max_value=40, value=5)
    monthly_income = st.sidebar.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
    job_satisfaction = st.sidebar.slider("Job Satisfaction", 1, 5, 3)
    performance_rating = st.sidebar.slider("Performance Rating", 1, 5, 3)
    overtime = st.sidebar.selectbox("Overtime", ["Yes", "No"])
    distance_from_home = st.sidebar.number_input("Distance from Home", min_value=1, max_value=50, value=10)
    education_level = st.sidebar.slider("Education Level", 1, 5, 3)
    job_level = st.sidebar.slider("Job Level", 1, 5, 3)
    company_reputation = st.sidebar.slider("Company Reputation", 1, 5, 3)

# Convert input into DataFrame
    input_data = pd.DataFrame({
     "Age": [age],
     "Years at Company": [years_at_company],
     "Monthly Income": [monthly_income],
     "Job Satisfaction": [job_satisfaction],
     "Performance Rating": [performance_rating],
     "Overtime": [label_encoders["Overtime"].transform([overtime])[0]],
     "Distance from Home": [distance_from_home],
     "Education Level": [education_level],
     "Job Level": [job_level],
     "Company Reputation": [company_reputation]
    })

# Prediction
    if st.sidebar.button("Predict Attrition"):
       prediction = gb_model.predict(input_data)[0]

       attrition_result = "Left" if prediction == 1 else "Stayed"
       st.write(f"### Prediction: {attrition_result}")

    # Confusion Matrix Plot
       y_test = [0, 1, 0, 1]  # Dummy values for labels
       y_pred = [0, 1, 1, 0]  # Dummy values for predictions
       conf_matrix = confusion_matrix(y_test, y_pred)
       plt.figure(figsize=(6, 4))
       sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'])
       plt.xlabel('Predicted')
       plt.ylabel('Actual')
       plt.title('Confusion Matrix')
       st.pyplot(plt)

#==================================================================================================|
elif model_choice == "Recommendation":
   
    import streamlit as st
    import pandas as pd
    import numpy as np
    import pickle
    from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
    df = pd.read_csv("disney_plus_titles.csv")

# Load pre-trained models
    content_sim = pickle.load(open("RD_content_model.pkl", "rb"))
    collab_model = pickle.load(open("RD_collaborative_model.pkl", "rb"))

# Function to get content-based recommendations
    def get_content_recommendations(movie_title, num_recommendations=5):
        
        if movie_title not in df["title"].values:
            
            return ["Movie not found. Please try another title."]
    
        movie_idx = df[df["title"] == movie_title].index[0]
        similar_movies = list(enumerate(content_sim[movie_idx]))
        sorted_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
        recommended_titles = [df.iloc[i[0]]["title"] for i in sorted_movies]
    
        return recommended_titles

# Function to get collaborative filtering recommendations (based on random user)
    def get_collab_recommendations(num_recommendations=5):
        movie_indices = np.argsort(collab_model[0])[-num_recommendations:][::-1]
        recommended_titles = [df.iloc[i]["title"] for i in movie_indices]
        return recommended_titles

# ------------------ Streamlit UI ------------------

# Custom CSS for Styling
    st.markdown("""
       <style>
        body {background-color: #f5f5f5; font-family: Arial, sans-serif;}
        .main-title {color: #ff4b4b; text-align: center; font-size: 40px; font-weight: bold;}
        .sub-title {color: #666; text-align: center; font-size: 20px; margin-bottom: 30px;}
        .stButton > button {background-color: #ff4b4b; color: white; font-size: 18px; border-radius: 10px; padding: 10px 20px;}
        .stTextInput>div>div>input {font-size: 18px; padding: 10px; border-radius: 10px;}
        .stMarkdown {text-align: center;}
      </style>
     """, unsafe_allow_html=True)

# Title
    st.markdown("<p class='main-title'>🎬 Movie Recommendation System</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Find the best movies based on your interest! 🍿</p>", unsafe_allow_html=True)

# Movie Search Input
    movie_input = st.text_input("Enter a movie name to get recommendations:", "")

# Content-Based Recommendation Button
    if st.button("Get Content-Based Recommendations 🎥"):
        
        if movie_input:
            
           recommendations = get_content_recommendations(movie_input)
           st.subheader("Recommended Movies Based on Content:")
           for movie in recommendations:
               st.write(f"✅ {movie}")
        else:
            st.warning("Please enter a movie name!")

# Collaborative Filtering Recommendation Button
    if st.button("Get Personalized Recommendations 👥"):
        recommendations = get_collab_recommendations()
        st.subheader("Recommended Movies Based on Users:")
        for movie in recommendations:
           st.write(f"⭐ {movie}")

# Footer
    st.markdown("<br><p style='text-align:center; font-size:14px; color:gray;'>Developed by Nithish, Yashas and Team 🚀</p>", unsafe_allow_html=True)

   
#====================================================================================================|
elif model_choice == "NLP":
    # Custom CSS for styling
    st.markdown("""
    <style>
    body {
        background-color: #1c1c1c; /* Dark background */
        color: white;
    }
    .stTextArea label, .stSelectbox label, .stButton button {
        font-size: 18px;
        font-weight: bold;
    }
    .stTitle {
        color: #800080 !important; /* Red title */
        text-align: center;
        font-size: 36px;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #cc0000;
    }
    </style>
    
    
    """, unsafe_allow_html=True)
    st.title("Natural Language Processing")
    st.markdown("<h1 class='stTitle'>🍽️ Zomato Review Sentiment Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:20px;color:#00ff00 '>Analyze customer reviews using AI-powered models.</p>", unsafe_allow_html=True)


    
    
    st.write("Explore NLP models for tasks like text classification, sentiment analysis, etc.")
    import streamlit as st
    import joblib
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    import re
    import nltk
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords
    nltk.download('stopwords')

# Load Pretrained Models and Vectorizer
    vectorizer = joblib.load("tfidf_vectorizer.pkl")  # TF-IDF Vectorizer
    models = {
    "Logistic Regression": joblib.load("logistic_regression_model.pkl"),
    "Naive Bayes": joblib.load("naive_bayes_model.pkl"),
    #"SVM": joblib.load("svm_model.pkl"),
    "Random Forest": joblib.load("random_forest_model.pkl")
    }

# Function to clean user input text
    def clean_text(text):
       text = re.sub(r'[^a-zA-Z ]', '', text)
       text = text.lower()
       words = text.split()
       words = [word for word in words if word not in stopwords.words('english')]
       return " ".join(words)

# Streamlit UI Layout
   # st.set_page_config(page_title="Zomato Review Sentiment", layout="wide")
    
# User Input for Review Prediction
    user_review = st.text_area("Enter your Zomato review:", "")

# Model Selection
    selected_model = st.selectbox("Choose a model:", list(models.keys()))

# Predict Sentiment
    if st.button("Predict Sentiment"):
        if user_review:
           cleaned_review = clean_text(user_review)
           st.subheader("🔹 WordCloud of Reviews")
           review_vectorized = vectorizer.transform([cleaned_review])
           prediction = models[selected_model].predict(review_vectorized)[0]
           sentiment = "😊 Liked (Positive)" if prediction == 1 else "😞 Not Liked (Negative)"
           st.success(f"Predicted Sentiment: {sentiment}")
           # WordCloud Visualization
   
           wordcloud = WordCloud(width=800, height=400, background_color="white").generate(cleaned_review)
           plt.figure(figsize=(10, 5))
           plt.imshow(wordcloud, interpolation="bilinear")
           plt.axis("off")
           st.pyplot(plt)
           
        else:
           st.warning("Please enter a review!")



   
#=================================================================================================================|
elif model_choice=="Create a Report":
    import streamlit as st
    import pandas as pd
    import plotly.express as px

# Title
    st.title("Create a Report by uploading csv file")

# File Upload
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
       df = pd.read_csv(uploaded_file)
       st.write("Data Preview:")
       st.write(df.head())
    
    # Show basic stats
       st.write(" Summary Statistics:")
       st.write(df.describe())
    
    # Select Column for Analysis
       numeric_cols = df.select_dtypes(include=['number']).columns
       category_cols = df.select_dtypes(include=['object']).columns
    
       if not numeric_cols.any():
           st.error("No numerical columns found for visualization.")
       else:
        # Sales Trend (if Date column exists)
            if 'Date' in df.columns:
               df['Date'] = pd.to_datetime(df['Date'])
               df = df.sort_values(by='Date')
               fig = px.line(df, x='Date', y=numeric_cols[0], title=f"{numeric_cols[0]} Over Time")
               st.plotly_chart(fig)
        
        # Bar Chart for Top Categories
            if len(category_cols) > 0:
                category = st.selectbox("Select a Category Column", category_cols)
                metric = st.selectbox("Select a Metric Column", numeric_cols)
                top_categories = df.groupby(category)[metric].sum().reset_index()
                fig = px.bar(top_categories, x=category, y=metric, title=f"Top {category} by {metric}")
                st.plotly_chart(fig)
        
        # Scatter Plot
            x_axis = st.selectbox("Select X-axis", numeric_cols, index=0)
            y_axis = st.selectbox("Select Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter Plot: {x_axis} vs {y_axis}")
            st.plotly_chart(fig)
        
            st.success("Dashboard Loaded Successfully ✅")

 
#==========================================================================================================|
# Function to apply styles for dark mode or light mode
def apply_styles(dark_mode: bool):
    if dark_mode:
        # Dark mode CSS for the entire website
        st.markdown("""
            <style>
                body {
                    background-color: #121212;
                    color: white;
                }
                .sidebar .sidebar-content {
                    background-color: #333;
                    color: white;
                }
                .stButton>button {
                    background-color: #444;
                    color: white;
                }
                .stTextInput>div>div>input {
                    background-color: #444;
                    color: white;
                }
                .stCheckbox {
                    color: white;
                }
                .stMarkdown, .stText {
                    color: white;
                }
            </style>
        """, unsafe_allow_html=True)
    else:
        # Light mode CSS for the entire website
        st.markdown("""
            <style>
                body {
                    background-color: #ffffff;
                    color: black;
                }
                .sidebar .sidebar-content {
                    background-color: #f0f0f0;
                    color: black;
                }
                .stButton>button {
                    background-color: #f0f0f0;
                    color: black;
                }
                .stTextInput>div>div>input {
                    background-color: white;
                    color: black;
                }
                .stCheckbox {
                    color: black;
                }
                .stMarkdown, .stText {
                    color: black;
                }
            </style>
        """, unsafe_allow_html=True)



# Add a toggle button for light and dark modes
st.markdown("""
    <style>
        .stCheckbox {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;  /* Ensures it's on top */
            padding: 10px;  /* Adds some space around the checkbox */
        }
    </style>
""", unsafe_allow_html=True)

# Create the dark mode toggle button
dark_mode = st.checkbox("Enable Dark Mode")

# Apply the corresponding styles
apply_styles(dark_mode)

#=====================================================================Chatbot============================================================|

import streamlit as st

# Chatbot function with nested options
def chatbot_response(user_input):
    responses = {
        "website": "This website is an AI/ML playground where you can test different models, visualize data, and generate reports.",
        "models": "We have multiple AI/ML models: Regression, Classification, Clustering, Sentiment Analysis, Word Cloud, and a Recommendation System. Which one do you want details about?",
        "regression": "Regression models are used for predicting continuous values, like sales forecasting or temperature prediction.",
        "classification": "Classification models categorize data into predefined labels, such as spam detection or medical diagnosis.",
        "clustering": "Clustering models group similar data points together, useful in customer segmentation or anomaly detection.",
        "sentiment analysis": "Sentiment Analysis detects emotions in text, like positive, neutral, or negative feedback analysis.",
        "word cloud": "Word Cloud visually represents frequently occurring words in a dataset.",
        "recommendation system": "Recommendation Systems suggest relevant content, like movie or product recommendations."
    }
    return responses.get(user_input.lower(), "Sorry, I didn't understand that. Try selecting an option!")

# Initialize session state
if 'chatbot_open' not in st.session_state:
    st.session_state['chatbot_open'] = False
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Open chatbot
if not st.session_state['chatbot_open']:
    if st.button("🤖 Open Chatbot"):
        st.session_state['chatbot_open'] = True

# Chatbot UI with nested options
if st.session_state['chatbot_open']:
    option = st.radio("What do you want to know?", ["Website Info", "Available Models"])

    if option == "Website Info":
        st.session_state['conversation'].append("Bot: " + chatbot_response("website"))
    elif option == "Available Models":
        model_choice = st.selectbox("Select a model to know more:", ["Regression", "Classification", "Clustering", "Sentiment Analysis", "Word Cloud", "Recommendation System"])
        if st.button("Get Model Info"):
            st.session_state['conversation'].append("Bot: " + chatbot_response(model_choice))

    # Display conversation history
    for message in st.session_state['conversation']:
        st.write(message)

    # Close chatbot button below conversation
    if st.button("❌ Close Chatbot"):
        st.session_state['chatbot_open'] = False
        st.session_state['conversation'] = []

#==================================================Rating======================================================|
# Add custom CSS for the bottom-right position of the rating section


if "rating_submitted" not in st.session_state:
    st.session_state["rating_submitted"] = False
    st.session_state["rating"] = 3  # Default rating value

if st.button("Rate This App 🌟"):
    # Rating section
    with st.form("rating_form"):
        st.subheader("We value your feedback!")
        st.session_state["rating"] = st.slider("How would you rate this app? (1-5 stars)", min_value=1, max_value=5, value=3, step=1, key="rating_slider")
        submit_button = st.form_submit_button("Submit Rating")
        if submit_button:
            st.session_state["rating_submitted"] = True

if st.session_state.get("rating_submitted", False):
    st.success(f"Thank you for rating us {st.session_state['rating']} star(s)! 🌟")

#=================================================Footer======================================================|
# Footer section
st.markdown("""
    <hr>
    <footer style="text-align: center; padding: 10px; font-size: 14px;">
        <p>Built with Passion and Powered by AI</p>
        <p>Designed and Developed by Nithish, Yashas & team</p>
        <p>📧 Contact: <a href="mailto:nithish699734@gmail.com" target="_blank">nithish699734@gmail.com</a></p>

    </footer>
    """, unsafe_allow_html=True)




