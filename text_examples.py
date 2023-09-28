import boto3
import json

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

def claude_prompt_format(prompt: str) -> str:
    # Add headers to start and end of prompt
    return "\n\nHuman: " + prompt + "\n\nAssistant:"

# Call AI21 labs model
def run_mid(prompt):
    prompt_config = {
        "prompt": prompt,
        "maxTokens": 5147,
        "temperature": 0.7,
        "stopSequences": [],
    }

    body = json.dumps(prompt_config)

    modelId = "ai21.j2-mid"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("completions")[0].get("data").get("text")
    return results


# Call Claude model
def call_claude(prompt):
    prompt_config = {
        "prompt": claude_prompt_format(prompt),
        "max_tokens_to_sample": 4096,
        "temperature": 0.5,
        "top_k": 250,
        "top_p": 0.5,
        "stop_sequences": [],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-v2"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("completion")
    return results


# Call Cohere model
def call_cohere(prompt):
    prompt_config = {
        "prompt": prompt,
        "max_tokens": 2048,
        "temperature": 0.7,
        # "return_likelihood": "GENERATION"
    }

    body = json.dumps(prompt_config)

    modelId = "cohere.command-text-v14"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("generations")[0].get("text")
    return results

def summarize_text(text):
    """
    Function to summarize text using a generative AI model.
    """
    prompt = f"Summarize the following text: {text}"
    result = run_mid(prompt)  # Assuming run_mid is the function that executes the model
    return result

def generate_code():
    """
    Function to generate Python code for uploading a file to Amazon S3.
    """
    prompt = "Write a Python function that uploads a file to Amazon S3"
    result = call_claude(
        prompt
    )  # Assuming call_claude is the function that executes the model
    return result

def perform_qa(text):
    """
    Function to perform a Q&A operation based on the provided text.
    """
    prompt = (
        f"How many models does Amazon Bedrock support given the following text: {text}"
    )
    result = call_cohere(
        prompt
    )  # Assuming call_cohere is the function that executes the model
    return result


if __name__ == "__main__":
    # Sample text for summarization
    text = "This April, we announced Amazon Bedrock as part of a set of new tools for building with generative AI on AWS. Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies, including AI21 Labs, Anthropic, Cohere, Stability AI, and Amazon, along with a broad set of capabilities to build generative AI applications, simplifying the development while maintaining privacy and security Today, I'm happy to announce that Amazon Bedrock is now generally available! I'm also excited to share that Meta's Llama 2 13B and 70B parameter models will soon be available on Amazon Bedrock."

    print("\n=== Summarization Example ===")
    summary = summarize_text(text)
    print(f"Summary: {summary}")

    print("\n=== Code Generation Example ===")
    code_snippet = generate_code()
    print(f"Generated Code:\n{code_snippet}")

    print("\n=== Q&A Example ===")
    answer = perform_qa(text)
    print(f"Answer: {answer}")