import torch 
import torch.nn as nn
from torchtext.legacy.data import Field, TabularDataset, BucketIterator

from modelComponents import attention, encoder, decoder, seq2seq

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for _, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

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

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
LR = 0.001
CLIP = 1

attention_module = attention.AttentionLayer(HID_DIM)
enc = encoder.Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = decoder.Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attention_module)
model = seq2seq.Seq2Seq(enc, dec, device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = LR)
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

N_EPOCHS = 10
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f}')
