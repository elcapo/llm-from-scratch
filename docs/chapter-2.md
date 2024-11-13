# Build a Large Language Model from Scratch

## Working with text data

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

tokenizer.encode('I like what I like')
tokenizer.decode([0, 1, 2, 0, 1])
```

This would return:

```python
[0, 1, 2, 0, 1]
'I like what I like'
```

#### Tiktoken Tokenizer

The tiktoken tokenizer uses OpenAI's [tiktoken](https://github.com/openai/tiktoken).

```python
from scratch.tokenizers.tiktoken_tokenizer import TiktokenTokenizer

tokenizer = TiktokenTokenizer()

tokenizer.encode('I like what I like')
tokenizer.decode([40, 588, 644, 314, 588])
```

This would return:

```python
[40, 588, 644, 314, 588]
'I like what I like'
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
from scratch.dataset import Dataset
from scratch.tokenizers.tiktoken_tokenizer import TiktokenTokenizer

text = 'How to build a Large Language Model from scratch'
dataset = Dataset(text, TiktokenTokenizer(), max_length=5)

for pairs in dataset:
    start = pairs[0].tolist()
    target = pairs[1].tolist()
    print("Start:", tokenizer.decode(start))
    print("Target:", tokenizer.decode(target), "\n")
```

This would print:

```
Start: How to build a Large
Target:  to build a Large Language

Start:  to build a Large Language
Target:  build a Large Language Model

Start:  build a Large Language Model
Target:  a Large Language Model from

Start:  a Large Language Model from
Target:  Large Language Model from scratch
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
        "Target:", tokenizer.decode(target[1]), "\n")
```

This code would print:

```
Batch 1
Start: This repository can be > Target:  repository can be installed
Start:  repository can be installed > Target:  can be installed as

Batch 2
Start:  can be installed as > Target:  be installed as a
Start:  be installed as a > Target:  installed as a regular

Batch 3
Start:  installed as a regular > Target:  as a regular Python
Start:  as a regular Python > Target:  a regular Python project
```

### Embeddings

#### Token Embeddings

The token embedding initially assigns a random vector to each token id of the input, as the `Embedding` layer is basically a lookup that returns the coefficients that correspond to the ids of the input tokens.

This coefficients will be adjusted during the training phase.

```python
from torch import randint
from scratch.embeddings.token_embeddings import TokenEmbeddings

embedding_layer = TokenEmbeddings(vocab_size=50257, output_dim=256)
random_input = randint(low=0, high=50257-1, size=(8, 4))
embeddings = embedding_layer.embed(random_input)

embeddings.shape
```

This would return:

```python
torch.Size([8, 4, 256])
```

#### Positional Embeddings

The positional embeddings are meant to add information about the position of each token in the input sequence. They also start as vectors with random values that are assigned to each token only that in this case, the generated value only depends on the position of each token, not on the values of the tokens themselves.

The coefficients of this layer will also be adjusted during the training phase.

```python
from scratch.embeddings.positional_embeddings import PositionalEmbeddings

embedding_layer = PositionalEmbeddings(context_length=4, output_dim=256)
embeddings = embedding_layer.embed()

embeddings.shape
```

This would return:

```python
torch.Size([4, 256])
```

#### Input Embeddings

The input embeddings are the result of adding the results of the token embeddings plus the positional embeddings.

```python
from torch import randint
from scratch.embeddings.input_embeddings import InputEmbeddings

embedding_layer = InputEmbeddings(vocab_size=50257, context_length=4, output_dim=256)
random_input = randint(low=0, high=50257-1, size=(8, 4))
embeddings = embedding_layer.embed(random_input)

embeddings.shape
```

This would return:

```python
torch.Size([8, 4, 256])
```
