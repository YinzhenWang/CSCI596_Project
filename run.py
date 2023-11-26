from models import run_model
from draw import initialize_window, render


def run(text,model_name,layer,head):
    attentions, tokens = run_model(text,model_name,layer, head)
    window = initialize_window(model_name, layer, head)
    render(window, tokens, attentions)
    return None


text = "I am a student in University of Southern California"
model = "dslim/bert-base-NER"
layer = 11
head = 4
run(text,model,layer,head)