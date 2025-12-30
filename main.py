import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load Environment Variables
GOOGLE_API_KEY = os.getenv("genai")



# Page Config
st.set_page_config(
    page_title="AI Chatbot Mentor",
    page_icon="🤖",
    layout="wide"
)

st.title("👋 Welcome to AI Chatbot Mentor")
st.write(
    "Your personalized AI learning assistant.\n"
    "Please select a learning module to begin your mentoring session."
)


# Module Selection
modules = [
    "Python",
    "SQL",
    "Power BI",
    "Exploratory Data Analysis (EDA)",
    "Machine Learning (ML)",
    "Deep Learning (DL)",
    "Generative AI",
    "Agentic AI"
]

selected_module = st.selectbox("📌 Select a Learning Module", modules)

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Reset chat when module changes
if "current_module" not in st.session_state:
    st.session_state.current_module = selected_module

if st.session_state.current_module != selected_module:
    st.session_state.messages = []
    st.session_state.current_module = selected_module

# LLM Initialization
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

# System Instruction (Domain Control)
system_instruction = f"""
You are an AI mentor strictly specialized in the {selected_module} domain.

RULES:
- Answer ONLY questions related to {selected_module}.
- If the question is outside the selected module, respond EXACTLY with:
"Sorry, I don’t know about this question. Please ask something related to the selected module."
- Do NOT explain why the question is irrelevant.
"""

# Display Chat History
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# Chat Input
user_input = st.chat_input("Ask your question...")

if user_input:
    # Add user message
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    # Prepare full context
    chat_context = [SystemMessage(content=system_instruction)] + st.session_state.messages

    # Generate response
    with st.chat_message("assistant"):
        try:
            response = llm.invoke(chat_context)
            st.write(response.content)
            st.session_state.messages.append(AIMessage(content=response.content))
        except Exception as e:
            st.error(f"Error: {e}")

# Download Chat History
# ----------------------------
if st.session_state.messages:
    chat_text = "\n".join(
        [
            f"You: {m.content}" if isinstance(m, HumanMessage)
            else f"AI: {m.content}"
            for m in st.session_state.messages
        ]
    )

    st.download_button(
        label="📥 Download Conversation",
        data=chat_text,
        file_name=f"{selected_module}_chat_history.txt",
        mime="text/plain"
    )
