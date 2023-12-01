from models import run_model
import gradio as gr
import multiprocessing

from draw import initialize_window, render_image
 
def create_image_process(text, model, layer, head, result_queue):
    attentions, tokens = run_model(text, model, layer, head)
    window = initialize_window(model, layer, head, False)
    image, image_name = render_image(window, tokens, attentions)
    result_queue.put(image_name)

def run(text, model, layer, head):
    result_queue = multiprocessing.Queue()

    process = multiprocessing.Process(
        target=create_image_process,
        args=(text, model, layer, head, result_queue),
    )

    process.start()
    process.join()

    # Retrieve the result from the queue
    image = result_queue.get()
    return image

demo = gr.Interface(
    fn = run,
    inputs = [
        gr.Textbox(label="Text"),
        gr.Dropdown(
            ["bert-base-uncased", "dog", "bird"], label="Model name", info="choose a number"
        ),
        gr.Slider(0, 11, step=1, value=11, label="layer", info="Choose between 0 and 11"),
        gr.Slider(0, 11, step=1, value=4, label="head", info="Choose between 0 and 11"),

    ],
    outputs = gr.Image()
)


if __name__ == "__main__":
    demo.launch(share=True)