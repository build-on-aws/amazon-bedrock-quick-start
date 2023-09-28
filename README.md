# Quickly build Generative AI applications with Amazon Bedrock

This repository contains code samples for building diverse AI applications using Amazon Bedrock's foundation models. Learn how to accelerate projects in image and text generation and beyond.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites
* Python 3.9 or higher
* pip
* [Model Access in Amazon Bedrock](https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/modelaccess)

### Installation

Clone the repo

```bash
git clone https://github.com/build-on-aws/amazon-bedrock-quick-start.git
```

Install required packages

```bash
pip install -r requirements.txt
```

## Usage

This repository contains various code samples demonstrating how to build AI applications using Amazon Bedrock's foundation models. Here's how to use each:

### Image Generation

To generate images using Stable Diffusion, run the following command:

```bash
streamlit run sd_sample_st.py
```

This will launch a Streamlit app where you can enter text prompts to generate corresponding images.

### Text Examples

Run this Python script to see different text-based applications like text summarization, code generation, and Q&A:

```bash
python text_examples.py
```

This script will output results for each of these applications, showcasing the versatility of foundation models in text-based tasks.

### Chatbot

To interact with a chatbot built using Amazon Bedrock, LangChain, and Streamlit, run:

```bash
streamlit run chat_bedrock_st.py
```

This launches a Streamlit app where you can have a conversation with the chatbot, witnessing AI-powered conversational capabilities firsthand.

### RAG Example

To see how Retrieval Augmented Generation (RAG) works with LangChain, execute:

```bash
python rag_example.py
```

This will demonstrate how RAG augments foundation models by retrieving and incorporating external data into the generated content.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.