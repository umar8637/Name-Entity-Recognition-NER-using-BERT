# Named Entity Recognition using BERT

This project demonstrates how to use BERT for Named Entity Recognition (NER) on the CoNLL-2003 dataset. The code involves loading the dataset, tokenizing the input data, adjusting labels, training a BERT model, and evaluating its performance.

## Requirements

To get started, install the necessary libraries:

- `datasets`
- `transformers`
- `seqeval`

## Dataset

The CoNLL-2003 dataset is used for NER. It can be loaded using the `datasets` library, which provides access to the dataset and its features such as `pos_tags`, `chunk_tags`, and `ner_tags`.

## Tokenizer

The BERT tokenizer is used to tokenize the input data. It is essential to handle the special tokens added by the tokenizer (e.g., `[CLS]`, `[SEP]`) and subword tokens. 

## Handling Tokenization and Labels

There is a mismatch between the tokenized input IDs and the labels due to the addition of special tokens and subword tokens. The solution involves:
- Adding `-100` for special tokens (for `[CLS]` and `[SEP]`).
- Masking the subwords.

The `-100` index is ignored during training by PyTorch's `torch.nn.CrossEntropyLoss` class.

## Model

A BERT "uncased" model for token classification is used. The model is initialized with a pre-trained BERT base model and is configured for token classification with the number of labels set to match the dataset.

## Training Arguments

The training arguments specify the training configuration, including evaluation strategy, learning rate, batch sizes, number of epochs, and weight decay.

## Data Collator

The data collator handles padding and prepares the input sequences and labels for training. It ensures that the input data is properly formatted for the model.

## Evaluation Metric

The `seqeval` library is used for evaluating the performance of the model. It can evaluate tasks such as named-entity recognition, part-of-speech tagging, and semantic role labeling.

## Training

The training process involves initializing a `Trainer` with the model, training arguments, datasets, data collator, tokenizer, and compute metrics function. The model is trained and evaluated on the validation dataset.

## Saving the Model

After training, the model and tokenizer are saved for future use.

## Inference

The trained model can be used for inference on new text inputs. A pipeline is created using the trained model and tokenizer, which can then be used to perform NER on new examples.
