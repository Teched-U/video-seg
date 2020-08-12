# You can choose either V1 or V2 as the BERT feature extractor for Chinese sentences
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import torch

# https://huggingface.co/bert-base-chinese?text=巴黎是%5BMASK%5D国的首都%E3%80%82
from transformers import AutoTokenizer, AutoModelWithLMHead

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
bert_model = AutoModelWithLMHead.from_pretrained("bert-base-chinese")

# V1
inputs = bert_tokenizer("您好", return_tensors="pt")
outputs = bert_model(**inputs)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
print(last_hidden_states[0][0])
print("Hahaha")

# V2
input_str = "你好呀"
with torch.no_grad():
    inputs = bert_tokenizer.tokenize(input_str)
    if len(inputs) > 510: inputs = inputs[:510]

    inputs = [bert_tokenizer.cls_token] + inputs + [bert_tokenizer.sep_token]
    inputs = bert_tokenizer.convert_tokens_to_ids(inputs)
    inputs = torch.tensor(inputs, dtype=torch.long)
    inputs = torch.unsqueeze(inputs, 0)

    # self.bert_model(inputs.squeeze())
    outputs = bert_model(inputs)
    last_hidden_states = outputs[0]
    print(last_hidden_states[0][0])

    # return last_hidden_states[0][0]

### English Model
# from transformers import BertTokenizer, BertModel
# import torch
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
#
# # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# inputs = tokenizer("您好", return_tensors="pt")
# outputs = model(**inputs)
#
# last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
# print(last_hidden_states)
# print("Hahaha")