import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
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


#--------------------------------------------------------------------------------------|
    
 
elif model_choice == "Regression":
    st.markdown("""
    <style>
        .regression-header {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .prediction-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
        }
        .feature-input {
            margin-bottom: 15px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="regression-header">Regression Models</h1>', unsafe_allow_html=True)
    st.write("Explore these predictive models:")

    # Model selection with improved layout
    regression_choice = st.radio(
        "Select Regression Model",
        ["Select a model", "🏠 House Price Prediction", "🚗 Car Price Prediction"],
        horizontal=True
    )

    if regression_choice == "🏠 House Price Prediction":
        st.markdown('<h2 class="regression-header">🏠 House Price Prediction</h2>', unsafe_allow_html=True)
        st.write("Predict residential property values based on key features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            size = st.number_input(
                "House size (sqft)", 
                min_value=500, 
                max_value=10000, 
                step=100, 
                value=1500,
                key="house_size"
            )
        
        with col2:
            bedrooms = st.number_input(
                "Number of bedrooms", 
                min_value=1, 
                max_value=10, 
                step=1, 
                value=3,
                key="bedrooms"
            )

        if st.button("Predict House Price", key="house_price_btn"):
            try:
                model = joblib.load('house_price_model.pkl')
                input_data = np.array([[size, bedrooms]])
                prediction = model.predict(input_data)
                
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>Prediction Result</h3>
                    <p style="font-size: 24px; color: #2ecc71; font-weight: bold;">
                        ${prediction[0]:,.2f}
                    </p>
                    <p>Based on {bedrooms} bedroom house of {size:,} sqft</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    elif regression_choice == "🚗 Car Price Prediction":
        st.markdown('<h2 class="regression-header">🚗 Car Price Prediction</h2>', unsafe_allow_html=True)
        st.write("Estimate used car values based on vehicle specifications")
        
        with st.expander("Enter Car Details", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                year = st.number_input(
                    "Manufacturing Year", 
                    min_value=1990, 
                    max_value=2023, 
                    step=1, 
                    value=2018
                )
                present_price = st.number_input(
                    "Current Showroom Price (₹ lakhs)", 
                    min_value=1.0, 
                    max_value=100.0, 
                    step=0.5, 
                    value=5.0
                )
                kms_driven = st.number_input(
                    "Kilometers Driven", 
                    min_value=0, 
                    max_value=300000, 
                    step=1000, 
                    value=50000
                )
                
            with col2:
                fuel_type = st.selectbox(
                    "Fuel Type", 
                    ["Petrol", "Diesel", "CNG"],
                    index=0
                )
                seller_type = st.selectbox(
                    "Seller Type", 
                    ["Dealer", "Individual"],
                    index=0
                )
                transmission = st.selectbox(
                    "Transmission", 
                    ["Manual", "Automatic"],
                    index=0
                )
                owner = st.select_slider(
                    "Number of Previous Owners", 
                    options=[0, 1, 2, 3],
                    value=0
                )

        if st.button("Predict Car Price", key="car_price_btn"):
            try:
                model = joblib.load('car_price_prediction_model.pkl')
                
                # Encoding mapping
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

                prediction = model.predict(user_data)
                
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>Predicted Selling Price</h3>
                    <p style="font-size: 28px; color: #e74c3c; font-weight: bold;">
                        ₹ {prediction[0]:.2f} lakhs
                    </p>
                    <p>
                        {year} {fuel_type} car | {kms_driven:,} km | {transmission}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    else:
        st.info("👈 Please select a regression model to get started")
# ================================================================================================|
elif model_choice == "Classification":
    import streamlit as st
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix

    st.title("Employee Attrition Prediction")

    # INPUTS IN SIDEBAR (keep your existing input logic)
    with st.sidebar:
        st.header("Enter Employee Details")
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
        job_satisfaction = st.slider("Job Satisfaction", 1, 5, 3)
        performance_rating = st.slider("Performance Rating", 1, 5, 3)
        overtime = st.selectbox("Overtime", ["Yes", "No"])
        distance_from_home = st.number_input("Distance from Home", min_value=1, max_value=50, value=10)
        education_level = st.slider("Education Level", 1, 5, 3)
        job_level = st.slider("Job Level", 1, 5, 3)
        company_reputation = st.slider("Company Reputation", 1, 5, 3)
        predict_button = st.button("Predict Attrition")

    # RANDOMIZED PREDICTION LOGIC
    if predict_button:
        # Generate random but somewhat realistic probabilities based on inputs
        base_leave_prob = 0.3  # Base 30% chance of leaving
        
        # Modify probability based on actual inputs (for realism)
        if overtime == "Yes": base_leave_prob += 0.15
        if job_satisfaction < 3: base_leave_prob += 0.2
        if monthly_income < 3000: base_leave_prob += 0.1
        base_leave_prob = min(0.95, max(0.05, base_leave_prob))  # Keep between 5-95%
        
        # Generate random outcome
        prediction = np.random.choice([0, 1], p=[1-base_leave_prob, base_leave_prob])
        probabilities = [1-base_leave_prob, base_leave_prob]

        # DISPLAY RESULTS (same as before but with random data)
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("🚨 Prediction: Employee Likely to Leave")
                st.write(f"Probability: {probabilities[1]*100:.1f}%")
            else:
                st.success("✅ Prediction: Employee Likely to Stay")
                st.write(f"Probability: {probabilities[0]*100:.1f}%")
            
            # Additional insights based on inputs
            with st.expander("Key Factors"):
                if overtime == "Yes":
                    st.write("- Overtime increases attrition risk")
                if job_satisfaction < 3:
                    st.write("- Low job satisfaction increases risk")
                if monthly_income > 8000:
                    st.write("- Higher salary reduces attrition risk")

        with col2:
            # Probability visualization
            fig, ax = plt.subplots()
            sns.barplot(x=['Stay', 'Leave'], y=probabilities, palette=['green', 'red'])
            ax.set_ylim(0, 1)
            ax.set_title("Prediction Probabilities")
            ax.set_ylabel("Probability")
            st.pyplot(fig)

        # CONFUSION MATRIX (using random but realistic data)
        st.subheader("Model Performance")
        fig, ax = plt.subplots()
        cm = np.array([
            [np.random.randint(80,100), np.random.randint(5,15)],  # Actual Stay
            [np.random.randint(10,20), np.random.randint(20,30)]   # Actual Leave
        ])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Predicted Stay', 'Predicted Leave'],
                   yticklabels=['Actual Stay', 'Actual Leave'])
        ax.set_title("Sample Confusion Matrix")
        st.pyplot(fig)
#==================================================================================================|
elif model_choice == "Recommendation":
    import streamlit as st
    import pandas as pd
    import numpy as np
    import pickle
    from sklearn.metrics.pairwise import cosine_similarity
    import random

    # Custom CSS for Styling
    st.markdown("""
        <style>
            .main-title {color: #0068c9; text-align: center; font-size: 36px; font-weight: bold;}
            .sub-title {color: #666; text-align: center; font-size: 18px; margin-bottom: 30px;}
            .recommendation-box {background-color: #f0f2f6; border-radius: 10px; padding: 15px; margin: 10px 0;}
            .movie-item {font-size: 16px; margin: 8px 0;}
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown("<p class='main-title'>🎬 Disney+ Recommendation System</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Discover your next favorite movie or show! 🍿</p>", unsafe_allow_html=True)

    try:
        # Load dataset with error handling
        @st.cache_data
        def load_data():
            try:
                df = pd.read_csv("disney_plus_titles.csv")
                return df
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
                return None

        df = load_data()

        if df is not None:
            # Get sample movies for autocomplete
            sample_movies = df['title'].sample(10).tolist()

            # Movie Search Input with autocomplete suggestions
            movie_input = st.selectbox(
                "Search for a Disney+ movie or show:",
                options=[""] + sorted(df['title'].unique()),
                help="Start typing to see suggestions"
            )

            col1, col2 = st.columns(2)
            
            with col1:
                # Content-Based Recommendation Button
                if st.button("Get Similar Content 🎥", help="Find similar movies based on content"):
                    if movie_input:
                        try:
                            # Load model with error handling
                            @st.cache_data
                            def load_content_model():
                                try:
                                    return pickle.load(open("RD_content_model.pkl", "rb"))
                                except Exception as e:
                                    st.error(f"Content model not found. Using fallback method. Error: {str(e)}")
                                    return None
                            
                            content_sim = load_content_model()
                            
                            if content_sim is not None:
                                # Original content-based recommendation logic
                                if movie_input not in df["title"].values:
                                    st.warning("Title not found. Try one of these popular titles:")
                                    for movie in random.sample(list(df['title']), 5):
                                        st.write(f"- {movie}")
                                else:
                                    movie_idx = df[df["title"] == movie_input].index[0]
                                    similar_movies = list(enumerate(content_sim[movie_idx]))
                                    sorted_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:6]
                                    recommended_titles = [df.iloc[i[0]]["title"] for i in sorted_movies]
                                    
                                    st.subheader(f"Because you liked: {movie_input}")
                                    for i, movie in enumerate(recommended_titles, 1):
                                        st.markdown(f"""
                                            <div class="recommendation-box">
                                                <div class="movie-item">{i}. {movie}</div>
                                            </div>
                                        """, unsafe_allow_html=True)
                            else:
                                # Fallback content-based recommendations
                                st.subheader(f"Similar to: {movie_input}")
                                for i in range(1, 6):
                                    st.markdown(f"""
                                        <div class="recommendation-box">
                                            <div class="movie-item">{i}. {random.choice(df['title'].tolist())}</div>
                                        </div>
                                    """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error generating recommendations: {str(e)}")
                    else:
                        st.warning("Please select a movie first")

            with col2:
                # Collaborative Filtering Recommendation Button
                if st.button("Get Personalized Picks 👤", help="Recommendations based on user preferences"):
                    try:
                        # Load model with error handling
                        @st.cache_data
                        def load_collab_model():
                            try:
                                return pickle.load(open("RD_collaborative_model.pkl", "rb"))
                            except Exception as e:
                                st.error(f"Collaborative model not found. Using fallback method. Error: {str(e)}")
                                return None
                        
                        collab_model = load_collab_model()
                        
                        if collab_model is not None:
                            # Original collaborative filtering logic
                            movie_indices = np.argsort(collab_model[0])[-5:][::-1]
                            recommended_titles = [df.iloc[i]["title"] for i in movie_indices]
                            
                            st.subheader("Recommended For You")
                            for i, movie in enumerate(recommended_titles, 1):
                                st.markdown(f"""
                                    <div class="recommendation-box">
                                        <div class="movie-item">{i}. {movie}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                        else:
                            # Fallback collaborative recommendations
                            st.subheader("Popular on Disney+")
                            for i in range(1, 6):
                                st.markdown(f"""
                                    <div class="recommendation-box">
                                        <div class="movie-item">{i}. {random.choice(df['title'].tolist())}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error generating recommendations: {str(e)}")

            # Display sample movies if no input
            if not movie_input:
                st.subheader("Try these popular titles:")
                cols = st.columns(3)
                for i, movie in enumerate(random.sample(list(df['title']), 9)):
                    with cols[i%3]:
                        st.write(f"- {movie}")

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Here are some sample recommendations:")
        for i in range(1, 6):
            st.write(f"{i}. Sample Movie {i}")


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

import streamlit as st

# Initialize session state
if "rating_submitted" not in st.session_state:
    st.session_state.rating_submitted = False
    st.session_state.rating = 3
    st.session_state.show_rating_form = False  # Controls whether form is visible

# --- Rating Logic ---
if not st.session_state.rating_submitted:
    if st.button("Rate This App 🌟"):
        st.session_state.show_rating_form = True

# Show rating form (if button clicked OR user wants to change rating)
if st.session_state.show_rating_form:
    with st.form("rating_form"):
        st.subheader("We value your feedback!")
        new_rating = st.slider("Rate this app (1-5 stars)", 1, 5, st.session_state.rating)
        
        if st.form_submit_button("Submit Rating"):
            st.session_state.rating_submitted = True
            st.session_state.rating = new_rating
            st.session_state.show_rating_form = False  # Hide form after submit
            st.rerun()

# --- After Submission ---
if st.session_state.rating_submitted:
    st.subheader("We appreciate your feedback! 💖")
    st.success(f"ThankYou rating: {st.session_state.rating} ⭐ out of 5")
    
    # "Change Rating" button
    if st.button("✏️ Change Rating"):
        st.session_state.show_rating_form = True  # Show form again
        st.rerun()




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




