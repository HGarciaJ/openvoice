import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

class SEExtractor:
    def __init__(self, model_name="facebook/wav2vec2-large-xlsr-53"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)

    def extract_features(self, speech):
        inputs = self.processor(speech, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def __call__(self, speech):
        return self.extract_features(speech)
