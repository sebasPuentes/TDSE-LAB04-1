# LLM Text Preprocessing and Embeddings

Educational project implementing text preprocessing fundamentals for Large Language Models, including tokenization, Byte-Pair Encoding (BPE), and embedding layers. Based on Chapter 2 of "Build a Large Language Model From Scratch" by Sebastian Raschka.

## Getting Started

These instructions will help you set up and run the Jupyter notebook on your local machine for learning and experimentation with LLM text preprocessing techniques.

### Prerequisites

Required software and libraries to run this project:

```bash
Python 3.8 or higher
PyTorch 2.0+
tiktoken 0.5.0+
pandas
requests
jupyter
```

### Installing

Step-by-step guide to set up the development environment:

1. Clone or download this repository

```bash
git clone https://github.com/sebasPuentes/TDSE-LAB04-1
cd TDSE-LAB04-1
```

2. Install required Python packages

3. Launch Jupyter notebook

```bash
jupyter notebook embeddings.ipynb
```

## Running the Notebook

The notebook is divided into sequential sections that build upon each other:

1. **Tokenization**: Learn how to break text into tokens
2. **BPE Encoding**: Implement BytePair encoding for handling unknown words
3. **Data Sampling**: Create sliding window datasets for training
4. **Embeddings**: Transform tokens into continuous vector representations

Execute cells in order from top to bottom.

### Key Concepts Demonstrated

**Tokenization and Vocabulary**
- Breaking text into processable units
- Building vocabularies from text corpora
- Handling special tokens and unknown words

**BytePair Encoding (BPE)**
- Subword tokenization for open vocabulary
- Using OpenAI's tiktoken library
- Encoding and decoding text efficiently

**Sliding Window Data Loading**
- Creating training examples with context windows
- Understanding stride and max_length parameters
- Preparing batched inputs for model training


## Understanding max_length and stride Parameters

When creating the dataloader with:

```python
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
```

`max_length` defines the context window size how many tokens the model sees at once. With `max_length=4`, each training example contains 4 consecutive tokens as input, and the next 4 tokens as targets.

Larger context allows the model to see more information at once but requires more memory and computation. Lower context allows faster training.

In the other hand, `stride` determines how many positions to move forward when creating the next training example. It controls the overlap between consecutive samples.

**When stride = max_length**:
- Uses each token exactly once
- Faster training, less redundancy

**When stride < max_length**:
- Creates more training examples from the same data
- Each token appears in multiple contexts
- Can improve learning but may cause overfitting

**When stride = 1**:
- Maximum data augmentation
- Highest risk of overfitting
- Significantly more training examples

## Built With

* [PyTorch](https://pytorch.org/) - Deep learning framework for embedding layers and tensor operations
* [tiktoken](https://github.com/openai/tiktoken) - OpenAI's BPE tokenizer with Rust-based performance
* [pandas](https://pandas.pydata.org/) - Data analysis and manipulation
* [Jupyter](https://jupyter.org/) - Interactive development environment


## Acknowledgments

* Code and concepts from "Build a Large Language Model From Scratch" by Sebastian Raschka
* Original code repository: [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
* Text corpus: "The Verdict" by Edith Wharton
* OpenAI's tiktoken library for efficient BPE tokenization


