import torch 
from torchtext.legacy.data import Field, TabularDataset, BucketIterator

SRC = Field(tokenize = "spacy", tokenizer_language = "en", init_token = '<sos>', eos_token = '<eos>', lower = True)
TRG = Field(tokenize = "spacy", tokenizer_language = "fr", init_token = '<sos>', eos_token = '<eos>', lower = True)

dataset = TabularDataset(
    path = "fra-eng/fra.txt",
    format = 'tsv',
    fields = [('src', SRC), ('trg', TRG)]
)

train_data, valid_data, test_data = dataset.split(split_ratio = [0.8, 0.15, 0.05])

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device
)

print(train_iterator, valid_iterator, test_iterator)