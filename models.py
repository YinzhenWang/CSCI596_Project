from transformers import BertTokenizer, BertModel
import numpy as np

def get_model(name):
    if name == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

def run_model(model_name, text, layer, head):
    tokenizer, model = get_model(model_name)
    encoded_input = tokenizer(text, return_tensors='pt',)
    output = model(**encoded_input,output_attentions=True)
    if head == 'average':
        attentions = output['attentions'][layer][0].mean(0)
    else:
        attentions = output['attentions'][layer][0][head]
    tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
    attentions = attentions[1:-1, 1:-1].detach().numpy()
    tokens = tokens[1:-1]
    #softmax scale to 0-1 for each token
    for i in range(attentions.shape[0]):
        print(np.exp(attentions[i]),np.sum(np.exp(attentions[i])))
        attentions[i] = np.exp(attentions[i])/np.sum(np.exp(attentions[i]))
    return attentions, tokens


