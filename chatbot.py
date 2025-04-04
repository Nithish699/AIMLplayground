import streamlit as st

# Chatbot function
def chatbot_response(user_input, conversation_history):
    # Simple rule-based responses
    responses = {
        "hi": "Hello! How can I help you today?",
        "hello": "Hi there! How can I assist you?",
        "bye": "Goodbye! Have a great day!",
        "models": "We have several AI/ML models for prediction, classification, and more. Select a model from the sidebar.",
        "help": "You can ask me about the models or any help regarding the project!"
    }
    
    # Get the response or a fallback message
    response = responses.get(user_input.lower(), "Sorry, I didn't understand that. Try asking something else!")
    
    # Append user input and response to the conversation history
    conversation_history.append(f"You: {user_input}")
    conversation_history.append(f"Bot: {response}")
    
    return response, conversation_history
