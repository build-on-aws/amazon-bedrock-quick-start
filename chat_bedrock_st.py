import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.bedrock import Bedrock
import time

st.title("ChatBedrock")

def prompt_fixer(prompt: str) -> str:
    # Add headers to start and end of prompt
    return "\n\nHuman: " + prompt + "\n\nAssistant:"

@st.cache_resource
def load_llm():
    llm = Bedrock(model_id="anthropic.claude-v2")
    llm.model_kwargs = {"temperature": 0.7, "max_tokens_to_sample": 2048}

    model = ConversationChain(llm=llm, verbose=True, memory=ConversationBufferMemory())

    return model


model = load_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        prompt = prompt_fixer(prompt)
        result = model.predict(input=prompt)

        # Simulate stream of response with milliseconds delay
        for chunk in result.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
