from model_types import BERT


class SentimentEngine:

    MODEL_TYPES = {
        "bert" : BERT
    }

    def __init__(self, model_type, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
