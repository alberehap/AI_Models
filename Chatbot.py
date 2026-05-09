import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.chat-box {
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
}

.user-message {
    background-color: #1E293B;
}

.bot-message {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Chatbot")
st.write("Mini ChatGPT using Hugging Face Transformers")

@st.cache_resource
def load_model():

    chatbot = pipeline(
        "text-generation",
        model="gpt2"
    )

    return chatbot

generator = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input(
    "Type your message:"
)

if st.button("Send"):

    if user_input != "":

        # Store user message
        st.session_state.messages.append(
            ("You", user_input)
        )

        # Generate AI response
        response = generator(
            user_input,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7
        )

        bot_reply = response[0]["generated_text"]

        # Store bot response
        st.session_state.messages.append(
            ("Bot", bot_reply)
        )

# --------------------------------
# Display Chat
# --------------------------------
for sender, message in st.session_state.messages:

    if sender == "You":

        st.markdown(
            f"""
            <div class="chat-box user-message">
            <b>🧑 You:</b><br>{message}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="chat-box bot-message">
            <b>🤖 Bot:</b><br>{message}
            </div>
            """,
            unsafe_allow_html=True
        )