import streamlit as st
import boto3
import json
from PIL import Image
import io


st.title("Building with Bedrock")  # Title of the application
st.subheader("Stable Diffusion Demo")


# List of Stable Diffusion Preset Styles
sd_presets = [
    "None",
    "3d-model",
    "analog-film",
    "anime",
    "cinematic",
    "comic-book",
    "digital-art",
    "enhance",
    "fantasy-art",
    "isometric",
    "line-art",
    "low-poly",
    "modeling-compound",
    "neon-punk",
    "origami",
    "photographic",
    "pixel-art",
    "tile-texture",
]

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

# Bedrock api call to stable diffusion
def generate_image(text, style):
    """
    Purpose:
        Uses Bedrock API to generate an Image
    Args/Requests:
         text: Prompt
         style: style for image
    Return:
        image: base64 string of image
    """
    body = {
        "text_prompts": [{"text": text}],
        "cfg_scale": 10,
        "seed": 0,
        "steps": 50,
        "style_preset": style,
    }

    if style == "None":
        del body["style_preset"]

    body = json.dumps(body)

    modelId = "stability.stable-diffusion-xl"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("artifacts")[0].get("base64")
    return results


# Turn base64 string to image with PIL
def base64_to_pil(base64_string):
    """
    Purpose:
        Turn base64 string to image with PIL
    Args/Requests:
         base64_string: base64 string of image
    Return:
        image: PIL image
    """
    import base64

    imgdata = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(imgdata))
    return image


# select box for styles
style = st.selectbox("Select Style", sd_presets)
# text input
prompt = st.text_input("Enter prompt")


#  Generate image from prompt,
if st.button("Generate Image"):
    image = base64_to_pil(generate_image(prompt, style))
    st.image(image)