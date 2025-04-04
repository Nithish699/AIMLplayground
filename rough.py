import streamlit as st
from datetime import datetime

# Sidebar content
st.sidebar.title("AI & ML Playground")
st.sidebar.header("Explore Machine Learning Models")
st.sidebar.text("Select the model you'd like to explore from the options below.")

# Current Date and Time
current_time = datetime.now()

# Add CSS animation for in and out effect
st.markdown("""
    <style>
        .datetime {
            font-size: 18px;
            font-weight: bold;
            color: #4CAF50;
            display: inline-block;
            animation: fadeInOut 5s infinite;
        }

        @keyframes fadeInOut {
            0% { opacity: 0; }
            50% { opacity: 1; }
            100% { opacity: 0; }
        }
    </style>
    <div class="datetime">Current Date and Time: {}</div>
""".format(current_time.strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)
