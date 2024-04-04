import base64
import json
import logging

import boto3
import streamlit as st
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

st.sidebar.title("Building with Bedrock")  # Title of the application
st.sidebar.subheader("Go game explanation Demo")

REGION = "us-west-2"

request_input = 'Analyze the situation of this Go game. Analyze the respective advantages and disadvantages of the black and white sides. Output it in Markdown format.'
output_template = """
<Example>
## Overview
...
## White
### Advantages
1. ...
2. ...
3. ...
### Disadvantages
1. ...
2. ...
3. ...
## Black
### Advantages
1. ...
2. ...
3. ...
### Disadvantages
1. ...
2. ...
3. ...
## Conclusion
...
</Example>
"""


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

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
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
        st.sidebar.header("Game situation")
        input_image = "chess.jpg"
        input_text = f'{request_input} {output_template}'
        st.sidebar.image(input_image)
        st.sidebar.write(request_input)
        st.header("Analyze results")
        # Read reference image from file and encode as base64 strings.
        with open(input_image, "rb") as image_file:
            content_image = base64.b64encode(image_file.read()).decode('utf8')

        message = {"role": "user",
                   "content": [
                       {"type": "image", "source": {"type": "base64",
                                                    "media_type": "image/jpeg", "data": content_image}},
                       {"type": "text", "text": input_text}
                   ]}

        messages = [message]

        response = run_multi_modal_prompt(
            bedrock_runtime, model_id, messages, max_tokens)
        st.write(response.get("content")[0].get("text"))
        logger.info(json.dumps(response, indent=4))

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)


if __name__ == "__main__":
    main()
