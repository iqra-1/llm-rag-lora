# Fine-Tuning a Retrieval-Augmented Generation Model for Legal Text with LoRA and Quantization

This repository contains an implementation of a fine-tuned Large Language Model (LLM) integrated with a Retrieval-Augmented Generation (RAG) pipeline. It uses quantized models for efficient training and inference, along with a dataset containing legal text for text completion and information retrieval tasks.

## Features

- **Fine-Tuning with LoRA:** Reduces the number of trainable parameters for faster and memory-efficient training.
- **Quantized Model Support:** Utilizes 4-bit quantized models to reduce memory usage while maintaining performance.
- **RAG Implementation:** Combines a retriever and a fine-tuned LLM to provide answers based on relevant documents.
- **FAISS Indexing:** For fast and efficient similarity-based search.
- **Text Splitting and Processing:** Processes documents with tools like `NLTKTextSplitter` and `Html2TextTransformer`.

## Installation

To use this project, ensure you have Python 3.8 or higher installed. Then, install the required dependencies:

```bash
pip install --upgrade pip
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install transformers torch datasets
pip install trl peft accelerate bitsandbytes
pip install faiss-gpu langchain langchain-community nltk sentence-transformers html2text
pip install playwright
playwright install
```

## Dataset

The model is fine-tuned on the [MultiLegalSBD dataset](https://huggingface.co/datasets/rcds/MultiLegalSBD), which contains legal text in multiple languages.

## Usage

### 1. Fine-Tuning the Model

This notebook demonstrates fine-tuning a quantized Llama-based model using the Low-Rank Adaptation (LoRA) technique.

```python
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from unsloth import UnslothTrainer, UnslothTrainingArguments

# Load the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Define training arguments
trainer = UnslothTrainer(
    model=model,
    train_dataset=dataset,
    args=UnslothTrainingArguments(
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=5e-5,
        output_dir="outputs",
    ),
)

trainer.train()
```

### 2. Retrieval-Augmented Generation (RAG)

The RAG pipeline retrieves relevant information from legal documents and uses the fine-tuned model to generate answers.

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load and process documents
docs = AsyncChromiumLoader(["https://www.justia.com/"]).load()
docs_transformed = Html2TextTransformer().transform_documents(docs)
chunked_documents = NLTKTextSplitter(chunk_size=10).split_documents(docs_transformed)

# Create FAISS index
db = FAISS.from_documents(chunked_documents, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 4})

# Query the RAG system
query = "What is the law on assault?"
retrieved_docs = retriever.get_relevant_documents(query)
```

## Results

The model achieves effective fine-tuning with minimal hardware requirements and provides reliable answers for queries based on indexed documents.

## Notebook

The code for this project is provided in the Jupyter Notebook file: **`FinalRAG+FineTuned_LLM.ipynb`**.

## Contributing

Contributions are welcome! Feel free to open an issue or create a pull request.

## License

This project is licensed under the MIT License.

