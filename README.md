# Build a Large Language Model from Scratch

This repository contains my own implementation of the code snippets that appear in [Sebastian Raschka](https://github.com/rasbt)'s: **Build a Large Language Model (from Scratch)** book.

It is part of my follow up of the sessions of the study group organized by [Santi Viquez](https://www.santiviquez.com) around the book. More information can be found in its Discord channel: [AI from scratch](https://discord.com/channels/1299408818681286699/).

Although these type of books are traditionally followed with Jupyter notebooks, I opted for a object oriented version where classes are created in files (see the [scratch/](scratch/) folder) as I found that it is easier to test.

## Installation

This repository can be installed as a regular Python project, only that I don't plan to upload it to the Python Package Index as it's meant for pedagogical purposes, rather than for production use cases.

```bash
git clone https://github.com/elcapo/llm-from-scratch
cd llm-from-scratch
```

## Usage

### Preprocessor

The preprocessor takes a big chunk of text and returns a list of strings (typically words) where spaces have been removed.

```python
from scratch.tokenizers.preprocessor import Preprocessor

preprocessor = Preprocessor()
preprocessor.preprocess('Hello beautiful world!')

# Returns
['Hello', 'beautiful', 'world']
```

### Tokenizers

The tokenizers are classes with a `encode` and a `decode` method.

#### Simple Tokenizer

The simple tokenizer is an easy to read tokenizer which code has no dependencies and does the job done.

```python
from scratch.tokenizers.simple_tokenizer import SimpleTokenizer

vocabulary = ['I', 'like', 'what']
tokenizer = SimpleTokenizer(vocabulary)

tokenizer.encode('I like what I like') # Returns: [0, 1, 2, 0, 1]
tokenizer.decode([0, 1, 2, 0, 1]) # Returns: 'I like what I like'
```

#### Tiktoken Tokenizer

The tiktoken tokenizer uses OpenAI's [tiktoken](https://github.com/openai/tiktoken).

```python
from scratch.tokenizers.tiktoken_tokenizer import TiktokenTokenizer

tokenizer = TiktokenTokenizer()

tokenizer.encode('I like what I like') # Returns: [40, 588, 644, 314, 588]
tokenizer.decode([40, 588, 644, 314, 588]) # Returns: 'I like what I like'
```

## Virtual Environment

Before installing the dependencies, it is recommended to create and activate a virtual environment.

```python
# Create a virtual environment in the `.venv` folder
python -m venv .venv

# Activate the new virtual environment
source .venv/bin/activate

# Install the dependencies
pip install -r requirements.txt
```

## Tests

In order to facilitate the readability of the tests, most of the examples used in them are literal copies of the values (strings and vectors) that appear in the book.

```bash
pytest
```
