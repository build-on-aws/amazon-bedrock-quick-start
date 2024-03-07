import base64
import json
import logging

import boto3
import streamlit as st
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

st.sidebar.title("Building with Bedrock")  # Title of the application
st.sidebar.subheader("Q&A for the uploaded image")

REGION = "us-west-2"


def add_chat_history_message(input: str, output: str):
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    st.session_state['history'].append({
        'input': input,
        'output': output
    })


def show_chat_history():
    if 'history' not in st.session_state:
        return
    for msg in st.session_state['history']:
        if 'input' in msg:
            st.chat_message(name='user').write(msg['input'])
        if 'output' in msg:
            st.chat_message(name='ai').write(msg['output'])


def get_chat_history():
    if 'history' not in st.session_state:
        return ''
    res = ''
    for msg in st.session_state['history']:
        if 'input' in msg:
            res += f"user: {msg['input']}"
        if 'output' in msg:
            res += f"ai: {msg['output']}"

    return res


def clear_chat_history_message():
    if 'history' in st.session_state:
        del st.session_state['history']


def run_multi_modal_prompt(bedrock_runtime, model_id, messages, max_tokens):
    """
    Invokes a model with a multimodal prompt.
    Args:
        bedrock_runtime: The Amazon Bedrock boto3 client.
        model_id (str): The model ID to use.
        messages (JSON): The messages to send to the model.
        max_tokens (int): The maximum  number of tokens to generate.
    Returns:
        None.
    """

    system_input = """
    You are Claude, an AI assistant created by Anthropic to be helpful,harmless, and honest. 
    Your goal is to provide informative and substantive responses to queries while avoiding potential harms.
    You should answer the questions in the same language with user input text.
    You should answer the question according to the history chat messages in <history>
    """

    system_input.format(history_messages=get_chat_history())

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_input,
            "messages": messages
        }
    )

    response = bedrock_runtime.invoke_model(
        body=body, modelId=model_id)
    response_body = json.loads(response.get('body').read())

    return response_body


def main():
    """
    Entrypoint for Anthropic Claude multimodal prompt example.
    """

    try:

        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=REGION,
        )

        model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
        max_tokens = 1000
        st.sidebar.header("What image would you like to analyst?")
        uploaded_file = st.sidebar.file_uploader("Upload an image",
                                                 type=['jpg', 'jpeg', 'png', 'gif', 'webp'],
                                                 on_change=clear_chat_history_message)
        content_image = None
        if uploaded_file:
            st.sidebar.image(uploaded_file)
            content_image = base64.b64encode(uploaded_file.read()).decode('utf8')

        # Read reference image from file and encode as base64 strings.

        input_text = st.chat_input(placeholder="What do you want to know?")
        if content_image:
            if input_text:
                show_chat_history()
                st.chat_message(name='user').write(input_text)
                input_text_with_history = input_text + f"<history>{get_chat_history()}<history>"
                message = {"role": "user",
                           "content": [
                               {"type": "image", "source": {"type": "base64",
                                                            "media_type": "image/jpeg", "data": content_image}},
                               {"type": "text", "text": input_text_with_history}
                           ]}

                messages = [message]

                with st.spinner('I am thinking about this...'):
                    response = run_multi_modal_prompt(bedrock_runtime, model_id, messages, max_tokens)

                st.chat_message(name='assistant').write(response.get("content")[0].get("text"))
                add_chat_history_message(input_text, response.get("content")[0].get("text"))
                logger.info(json.dumps(response, indent=4))

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)


if __name__ == "__main__":
    main()
