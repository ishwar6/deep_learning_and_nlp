import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class MachineTranslationModel(nn.Module):
    """A simple machine translation model using BERT for encoding and a decoder."""
    def __init__(self):
        super(MachineTranslationModel, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.decoder = nn.GRU(input_size=768, hidden_size=512, num_layers=1, batch_first=True)
        self.fc = nn.Linear(512, 10000)  # Assuming a vocab size of 10,000

    def forward(self, input_ids, attention_mask):
        """Forward pass through the model."""
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states, _ = self.decoder(encoder_outputs.last_hidden_state)
        logits = self.fc(hidden_states)
        return logits

def translate_sentence(model, tokenizer, sentence):
    """Translates a given sentence using the machine translation model."""
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    outputs = model(inputs['input_ids'], inputs['attention_mask'])
    return outputs.argmax(dim=-1)

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = MachineTranslationModel()
    model.eval()
    sentence = 'Hello, how are you?'
    translation = translate_sentence(model, tokenizer, sentence)
    print('Translated Sentence IDs:', translation.tolist())