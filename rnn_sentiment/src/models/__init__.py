from src.models.rnn             import VanillaRNN
from src.models.lstm            import LSTMClassifier
from src.models.attention_lstm  import AttentionLSTM

MODEL_REGISTRY = {
    "rnn":      VanillaRNN,
    "lstm":     LSTMClassifier,
    "attention": AttentionLSTM,
}