import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# ------------------ BASIC SETUP ------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("genai")
st.set_page_config(page_title="AI Chatbot Mentor", page_icon="🤖")

# ------------------ SESSION STATE INIT ------------------
if "module" not in st.session_state:
    st.session_state.module = None

if "chat" not in st.session_state:
    st.session_state.chat = []

if "memory" not in st.session_state:
    st.session_state.memory = []

# ------------------ WELCOME SCREEN ------------------
if st.session_state.module is None:
    st.title("👋 Welcome to AI Chatbot Mentor")
    st.write("Your personalized AI learning assistant.")
    st.write("Please select a learning module to begin your mentoring session.")

    module = st.selectbox(
        "📌 Select a Module",
        [
            "Python",
            "SQL",
            "Power BI",
            "Exploratory Data Analysis (EDA)",
            "Machine Learning (ML)",
            "Deep Learning (DL)",
            "Generative AI (Gen AI)",
            "Agentic AI"
        ]
    )

    if st.button("Start Mentoring"):
        st.session_state.module = module

        # SYSTEM PROMPT (CORE OF DOMAIN CONTROL)
        system_prompt = f"""
You are an AI Mentor dedicated ONLY to the selected learning module: {module}.

Your responsibility:
- Decide whether the user's question belongs to the subject {module} itself.

Rules:
- If the question is about the concepts, ideas, methods, or topics that DEFINE {module}, answer it.
- If the question is about any topic that does NOT belong to the subject {module}, do NOT answer it.
- Do NOT assume a topic belongs to {module} just because it is commonly used together with it.
- If the question is NOT about {module}, reply ONLY with this exact sentence:

Sorry, I don’t know about this question. Please ask something related to the selected module.

- For valid questions, explain clearly and in a beginner-friendly way.
"""



        st.session_state.memory.append(("system", system_prompt))
        st.rerun()

# ------------------ MODULE CHAT INTERFACE ------------------
else:
    st.title(f"Welcome to {st.session_state.module} AI Mentor 🎯")
    st.write(f"I am your dedicated mentor for **{st.session_state.module}**.")
    st.write("How can I help you today?")

    # Display chat history
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User input
    user_input = st.chat_input("Ask your question here")

    if user_input:
        # Store user message
        st.session_state.chat.append({"role": "user", "content": user_input})
        st.session_state.memory.append(("user", user_input))

        with st.chat_message("user"):
            st.write(user_input)

        # Call LLM
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        response = model.invoke(st.session_state.memory)

        # Store AI response
        st.session_state.chat.append({"role": "ai", "content": response.content})
        st.session_state.memory.append(("ai", response.content))

        with st.chat_message("ai"):
            st.write(response.content)

    # ------------------ DOWNLOAD CHAT FEATURE (MANDATORY) ------------------
    if st.session_state.chat:
        conversation_text = ""
        for msg in st.session_state.chat:
            role = "User" if msg["role"] == "user" else "AI"
            conversation_text += f"{role}: {msg['content']}\n\n"

        st.download_button(
            label="📥 Download Conversation",
            data=conversation_text,
            file_name=f"{st.session_state.module}_Chat_History.txt",
            mime="text/plain"
        )