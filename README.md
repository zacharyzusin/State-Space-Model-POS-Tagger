# State Space Model POS Tagger
In this project, I developed a Part-of-Speech (POS) tagging system using a selective State Space Model architecture. The code consisted of a single-layer network with an embedding dimension of 64 and handled sequences of length 4. I made use of pre-trained word2vec embeddings and I worked with the CoNLL 2003 dataset, splitting the original 46 POS tags into four essential categories: Nouns (including pronouns), Verbs, Adjectives/Adverbs, and an Other category.

## Prerequisites

- MATLAB

## Project Structure

```
.
├── TextPreprocessor.m      # Handles data loading and preprocessing
├── StateSpaceTagger.m      # Core state space model implementation
├── StateSpaceOptimizer.m   # Training and optimization logic
├── main.m                  # Entry point script
├── train_data.csv         # Training dataset
├── valid_data.csv         # Validation dataset
├── test_data.csv          # Test dataset
└── embeddings.csv         # Pre-trained word embeddings
```

## Running the Project

1. Verify that your data files are in the correct location and format as specified above.

2. Run the project by executing the main script

## Model Configuration

The default configuration can be modified by adjusting the following parameters:

### StateSpaceTagger.m
- `hidden_dim`: Dimension of the state space (default: 64)
- `output_dim`: Number of POS tag classes (default: 4)
- `context_size`: Size of the context window (default: 4)

### StateSpaceOptimizer.m
- `learn_rate`: Learning rate for optimization (default: 0.001)
- `mini_batch_size`: Size of mini-batches (default: 32)
- `max_epochs`: Maximum number of training epochs (default: 50)
- `early_stop_limit`: Number of epochs without improvement before early stopping (default: 5)

## POS Tag Classes

The model uses a simplified tag set with 4 classes:
1. Nouns (including proper nouns, common nouns)
2. Verbs (all verb forms)
3. Modifiers (adjectives and adverbs)
4. Others (remaining POS tags)

## Output

The training process will display:
- Per-epoch loss and accuracy metrics
- Per-class precision, recall, and accuracy
- Early stopping notifications when triggered
- Final model performance on test set
