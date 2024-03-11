import base64
import json
import logging

import boto3
import streamlit as st
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

st.sidebar.title("Building with Bedrock")  # Title of the application
st.sidebar.subheader("Q&A for the uploaded image")

REGION = "us-west-2"


def save_chat_history_message(history: list):
    st.session_state['history'] = history


def has_history():
    return 'history' in st.session_state


def show_chat_history():
    if 'history' not in st.session_state:
        return
    for msg in st.session_state['history']:
        if 'content' not in msg:
            continue
        if type(msg['content']) is list:
            for item in msg['content']:
                if item['type'] == "text":
                    st.chat_message(name=msg['role']).write(item['text'])
                elif item['type'] == "image":
                    continue
                    # st.chat_message(name=msg['role']).image(item['source']['data'])
        else:
            st.chat_message(name=msg['role']).write(msg['content'])


def get_chat_history():
    if not has_history():
        return []
    return st.session_state['history']


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
    """

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
        max_tokens = 4096
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
                message = {"role": "user",
                           "content": [
                               {"type": "text", "text": input_text}
                           ]}
                if not has_history():
                    message["content"].append({"type": "image",
                                               "source": {"type": "base64",
                                                          "media_type": "image/jpeg",
                                                          "data": content_image}})

                messages = []

                # Get History Messages
                if has_history():
                    messages.extend(get_chat_history())
                messages.append(message)
                with st.spinner('I am thinking about this...'):
                    response = run_multi_modal_prompt(bedrock_runtime, model_id, messages, max_tokens)

                st.chat_message(name='assistant').write(response.get("content")[0].get("text"))
                messages.append({
                    "role": "assistant",
                    "content": response.get("content")[0].get("text")
                })
                save_chat_history_message(messages)
                logger.debug(json.dumps(response, indent=4))

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)


if __name__ == "__main__":
    main()
