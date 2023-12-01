import streamlit as st
import subprocess
from run import run

def main():
    # Initialize default values
    default_text = 'I am a student in USC majoring in computer science'
    default_model = 'bert-base-uncased'
    default_layer = 11
    default_head = 4

    # Use st.session_state to persist state across Streamlit sessions
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False

    if 'image_result' not in st.session_state:
        st.session_state.image_result = 0

    with st.sidebar:
        input_text = st.text_input('Input text', default_text)
        input_model = st.selectbox('Model name', ('bert-base-uncased', ''))
        input_layer = st.slider('Layer', 0, 11, default_layer)
        input_head = st.slider('Head', 0, 11, default_head)

        if st.button('Submit'):
            st.session_state.submitted = True
            command = ['python', 'run.py', input_text, input_model, str(input_layer), str(input_head)]
            process = subprocess.run(command, capture_output=True, text=True)
            st.session_state.image_result = process.stdout.strip()

    st.header("Output:")
    
    # Run subprocess with default values if not submitted
    if not st.session_state.submitted:
        command = ['python', 'run.py', default_text, default_model, str(default_layer), str(default_head)]
        process = subprocess.run(command, capture_output=True, text=True)
        st.session_state.image_result = process.stdout.strip()

    if st.session_state.image_result:
        st.image(f'{st.session_state.image_result.strip()}')

if __name__ == "__main__":
    main()
