from transformers import BertTokenizer, BertModel
import numpy as np

def get_model(name):
    print(name)
    try:
        tokenizer = BertTokenizer.from_pretrained(name)
        model = BertModel.from_pretrained(name)
    except:
        raise Exception("Model name is not valid")
    return tokenizer, model

def run_model(text, model_name, layer, head):
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
        attentions[i] = attentions[i]/np.max(attentions[i])
    return attentions, tokens


