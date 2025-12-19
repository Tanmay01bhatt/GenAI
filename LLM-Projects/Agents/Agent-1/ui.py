# ui.py
import re
import streamlit as st
from agent import get_agent_response

st.set_page_config(page_title="LangChain Chat Agent", layout="wide")

st.title("LangChain Chat Agent")

# -----------------------
# Initialize chat history
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "text": "Hi I'm your LangChain agent. Ask me anything!"}
    ]

# -----------------------
# Input box at bottom
# -----------------------
if user_input := st.chat_input("Type your message..."):
    # User message
    st.session_state.messages.append({"role": "user", "text": user_input}) #Store user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Agent response with spinner
    with st.spinner(" Thinking..."):
        try:
            response = get_agent_response(user_input)
        except Exception as e:
            response = f"Agent error: {e}"

    st.session_state.messages.append({"role": "assistant", "text": response})  #Store agent message
    with st.chat_message("assistant"):                         #Display agent response
        text = response
        yt_links = re.findall(r"(https?://www\.youtube\.com/watch\?v=[\w-]+)", text)  #Find YouTube links in the response
        for link in yt_links:   # For each YouTube link: embed video
            col, _ = st.columns([2, 3])     #Create two columns: first column gets width 2 ,The second column gets width 3

            with col: 
                st.video(link)    #embeds a playable YouTube video.
            text = text.replace(link, "") #Remove link from text 
            
        if text.strip():
            st.markdown(text)    #Display leftover text without YouTube links
# -----------------------
# Display chat history
# -----------------------
for msg in st.session_state.messages:   #Iterate through saved messages
    with st.chat_message(msg["role"]):
        text = msg["text"]
        # Show all YouTube links
        yt_links = re.findall(r"(https?://www\.youtube\.com/watch\?v=[\w-]+)", text)
        for link in yt_links:
            col, _ = st.columns([2, 3])
            with col:
                st.video(link)
            text = text.replace(link, "")
            
        # # Show leftover plain text
        if text.strip():
            st.markdown(text)

