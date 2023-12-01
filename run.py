from models import run_model
from draw import initialize_window, render_image
import sys

def run(text,model_name,layer,head):
    attentions, tokens = run_model(text,model_name,layer, head)
    # window = initialize_window(model_name, layer, head, False)
    # render_image(window, tokens, attentions)
    window = initialize_window(model_name, layer, head, False)
    image, image_name = render_image(window, tokens, attentions)
    return image_name

if __name__ == "__main__":
    text = sys.argv[1]
    model_name = sys.argv[2]
    layer = int(sys.argv[3])
    head = int(sys.argv[4])
    image_name = run(text,model_name,layer,head)
    print(image_name)
# text = "I am a student in USC majoring in computer science"
# # model = "dslim/bert-base-NER"
# model = "bert-base-uncased"
# layer = 11
# head = 4
# run(text,model,layer,head)
