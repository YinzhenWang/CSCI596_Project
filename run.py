from models import run_model
from draw import initialize_window, render_image
import sys
import argparse

def run(text,model_name,layer,head):
    attentions, tokens = run_model(text,model_name,layer, head)
    # window = initialize_window(model_name, layer, head, False)
    # render_image(window, tokens, attentions)
    window = initialize_window(model_name, layer, head, False)
    image, image_name = render_image(window, tokens, attentions)
    return image_name

if __name__ == "__main__":
    # python run.py -t "The cat sat on the mat" -m "dslim/bert-base-NER" -l 11 -he 4
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", "-t", help="text to be processed")
    parser.add_argument("--model_name", "-m", help="model name")
    parser.add_argument("--layer", "-l", help="layer")
    parser.add_argument("--head", "-he", help="head")

    args = parser.parse_args()
    image_name = run(args.text,args.model_name,int(args.layer),int(args.head))
    print(image_name)
