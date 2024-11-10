# Build a Large Language Model from Scratch

This repository contains my own implementation of the code snippets that appear in [Sebastian Raschka](https://github.com/rasbt)'s: **Build a Large Language Model (from Scratch)** book.

It is part of my follow up of the sessions of the study group organized by [Santi Viquez](https://www.santiviquez.com) around the book.

I opted for a object oriented version where classes are created in files (see the [scratch](scratch) folder) as I found that it is easier to test.

## Installation

This repository can be installed as a regular Python project, only that I don't plan to upload it to the Python Package Index as it's meant for pedagogical purposes, rather than for production use cases.

```bash
git clone https://github.com/elcapo/llm-from-scratch
cd llm-from-scratch
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
