# NeuroLingua

NeuroLingua is a machine translation system built using PyTorch. It leverages the power of LSTM networks and introduces attention mechanisms to improve the translation from source to target languages.

## Features
- LSTM based encoder-decoder architecture.
- Incorporates attention mechanisms for enhanced context capture.
- Easily configurable model parameters for experimentation.
- Utilizes the TorchText library for efficient data handling.

## Requirements
- Python 3.9 or above
- PyTorch
- TorchText
- spaCy (with `en_core_web_sm` and `fr_core_news_sm` models)

## Quick Start
1. Clone the repository:
```bash
git clone [Your Repo URL]
cd NeuroLingua
```

2. Install the dependencies:
```bash
pip install torch torchtext spacy
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

3. Run the main script:
```bash
python main.py
```

## Structure
- `attention.py`: Defines the attention mechanisms for the model.
- `encoder.py`: Contains the Encoder LSTM implementation.
- `decoder.py`: Contains the Decoder LSTM with attention.
- `seq2seq.py`: Combines the encoder and decoder into a single sequence-to-sequence model.
- `main.py`: The main driver script which loads data, defines the model, and trains it.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
```
