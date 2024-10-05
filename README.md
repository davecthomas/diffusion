# Diffusion - Auto-generate prompts and images based on a specific topic

A simple program that hits the OpenAI completions and images APIs to both generate image prompts and images.

# Install

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
vi .env
```

# Run the program

Make sure you set up your .env file before running.

```bash
python diffusion.py
```

## Settings

```bash
OPENAI_API_KEY
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_USER
OPENAI_COMPLETIONS_MODEL=gpt-4o-mini
```
