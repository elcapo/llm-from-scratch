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

The tokenizers are classes with a `encode` and a `decode` method where:

- the `encode` method receives a text and transforms it into a vector of numbers,
- the `decode` method receives a vector of numbers returned by the encoding function and transforms it back into the original text.

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

### Dataset and Dataloader

#### Dataset

The dataset class prepares a text for using it to train a Large Language Model to "guess the next token". To do so, it gives us access to a given source text through a window where we always see two parts of the text:

- the one we'll feed the model as its input
- the one the model will see as the result

For instance, from the sentence **How to build a Large Language Model**, we could have:

- How to build a Large Language
- to build a Large Language **Model**

```python
from scratch.gpt_dataset import GptDataset
from scratch.tokenizers.tiktoken_tokenizer import TiktokenTokenizer

text = 'How to build a Large Language Model from scratch'
dataset = GptDataset(text, TiktokenTokenizer(), max_length=5)

for pairs in dataset:
    start = pairs[0].tolist()
    target = pairs[1].tolist()
    print("Start:", tokenizer.decode(start), ">",
        "Target:", tokenizer.decode(target))

# Start: How to build a Large > Target:  to build a Large Language
# Start:  to build a Large Language > Target:  build a Large Language Model
# Start:  build a Large Language Model > Target:  a Large Language Model from
# Start:  a Large Language Model from > Target:  Large Language Model from scratch
```

#### Dataloader

The dataloader helper prepares an iterator that traverses a given dataset and helps us preparing batches for the training phase.

```python
from scratch.dataloader import create_dataloader
from scratch.tokenizers.tiktoken_tokenizer import TiktokenTokenizer

tokenizer = TiktokenTokenizer()
dataloader = create_dataloader(
    'This repository can be installed as a regular Python project',
    batch_size=2,
    max_length=4,
    stride=1,
    shuffle=False)

for n, batch in enumerate(dataloader):
    start = batch[0].tolist()
    target = batch[1].tolist()

    print("Batch", n + 1)

    print("Start:", tokenizer.decode(start[0]), ">",
        "Target:", tokenizer.decode(target[0]))

    print("Start:", tokenizer.decode(start[1]), ">",
        "Target:", tokenizer.decode(target[1]))

# Batch 1
# Start: This repository can be > Target:  repository can be installed
# Start:  repository can be installed > Target:  can be installed as
# Batch 2
# Start:  can be installed as > Target:  be installed as a
# Start:  be installed as a > Target:  installed as a regular
# Batch 3
# Start:  installed as a regular > Target:  as a regular Python
# Start:  as a regular Python > Target:  a regular Python project
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
