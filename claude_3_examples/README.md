# Getting Started with Claude 3 on Amazon Bedrock

This repository contains code samples for using Cladue 3 on Amazon Bedrock.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites
* Python 3.9 or higher
* pip
* [Model Access in Amazon Bedrock](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/modelaccess)

### Text Examples

Run this Python script to see different text-based applications like text summarization, code generation, and Q&A:

```bash
python text_examples.py
```

### Image Generation

To generate images using Stable Diffusion and have Claude 3 capation the image, run the following command:

```bash
streamlit run image_api_st.py
```
<div align="center"><img src="cat_image.png" alt="Cat jumping into water"></div>

