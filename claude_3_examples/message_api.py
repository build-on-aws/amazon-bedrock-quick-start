import boto3
import json
import time

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
)


def call_claude_sonet(prompt):

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("content")[0].get("text")
    return results

def summarize_text(text):
    """
    Function to summarize text using a generative AI model.
    """
    prompt = f"Summarize the following text in 50 words or less: {text}"
    result = call_claude_sonet(prompt)
    return result


def sentiment_analysis(text):
    """
    Function to return a JSON object of sentiment from a given text.
    """
    prompt = f"Giving the following text, return only a valid JSON object of sentiment analysis. text: {text} "
    result = call_claude_sonet(prompt)
    return result


def perform_qa(question, text):
    """
    Function to perform a Q&A operation based on the provided text.
    """
    prompt = f"Given the following text, answer the question. If the answer is not in the text, 'say you do not know': {question} text: {text} "
    result = call_claude_sonet(prompt)
    return result


if __name__ == "__main__":
    # Sample text for summarization
    text = "Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Stability AI, and Amazon via a single API, along with a broad set of capabilities you need to build generative AI applications with security, privacy, and responsible AI. Using Amazon Bedrock, you can easily experiment with and evaluate top FMs for your use case, privately customize them with your data using techniques such as fine-tuning and Retrieval Augmented Generation (RAG), and build agents that execute tasks using your enterprise systems and data sources. Since Amazon Bedrock is serverless, you don't have to manage any infrastructure, and you can securely integrate and deploy generative AI capabilities into your applications using the AWS services you are already familiar with"

    print("\n=== Summarization Example ===")
    summary = summarize_text(text)
    print(f"Summary:\n {summary}")
    time.sleep(2)

    print("\n=== Sentiment Analysis Example ===")
    sentiment_analysis_json = sentiment_analysis(text)
    print(f"{sentiment_analysis_json}")
    time.sleep(2)

    print("\n=== Q&A Example ===")

    q1 = "How many companies have models in Amazon Bedrock?"
    print(q1)
    answer = perform_qa(q1, text)
    print(f"Answer: {answer}")
    time.sleep(2)

    q2 = "Can Amazon Bedrock support RAG?"
    print(q2)
    answer = perform_qa(q2, text)
    print(f"Answer: {answer}")
    time.sleep(2)

    q3 = "When was Amzozn Bedrock announced?"
    print(q3)
    answer = perform_qa(q3, text)
    print(f"Answer: {answer}")