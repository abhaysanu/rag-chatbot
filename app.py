import streamlit as st
import requests

st.set_page_config(page_title="LangGraph ChatBot", page_icon="💬")
st.title("💬 LangGraph ChatBot")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", "", placeholder="Type your message here...")
    submitted = st.form_submit_button("Send")

# On send
if submitted and user_input:
    # Send POST request to FastAPI
    response = requests.post(
        "http://127.0.0.1:8000/chat",
        json={
            "message": user_input,
            "chat_history": st.session_state.chat_history
        }
    )

    if response.status_code == 200:
        result = response.json()
        bot_reply = result.get("response", "No response.")
        
        # Save to history
        st.session_state.chat_history.append({"user": user_input, "bot": bot_reply})
    else:
        st.error("Something went wrong! Please check the backend.")

# Show chat history
st.subheader("Conversation")
for turn in st.session_state.chat_history:
    st.markdown(f"**You:** {turn['user']}")
    st.markdown(f"**Bot:** {turn['bot']}")
    st.markdown("---")
