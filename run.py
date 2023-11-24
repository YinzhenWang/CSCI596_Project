from models import run_model
from draw import initialize_window, render
text = "I am a student majoring in computer science"
attentions,tokens = run_model('bert-base-uncased', text,11,0)
print(len(tokens), attentions)
window = initialize_window()
render(window, tokens, attentions)