import subprocess
import sys
import os

## Turn around camel-kenlm wheel error
def install_packages():
    # Install regular packages
    packages = [
        "future", "six", "docopt", "cachetools", "numpy", "scipy", "pandas",
        "scikit-learn", "torch", "transformers", "editdistance", "requests",
        "emoji", "pyrsistent", "muddler"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)

    # Install camel-tools without dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "camel-tools", "--no-deps"])

# Run the install function
install_packages()

os.system('python -m spacy download en_core_web_sm')

import torch
import utils
from utils import camel_tokenizer, spacy_tokenizer
from Vocabulary import Vocabulary
from models import Seq2seq_with_attention, Encoder, Decoder, Attention
import gradio as gr

device = 'cuda' if torch.cuda.is_available() else 'cpu'

en_tokenizer = spacy_tokenizer()
en_vocab = Vocabulary(en_tokenizer, max_freq=2, unk=True, sos=True, eos=True)
ar_tokenizer = camel_tokenizer()
ar_vocab = Vocabulary(ar_tokenizer, max_freq=2, unk=True, sos=True, eos=True)

vocabs = torch.load('./seq2seq_with_attention_states.pt', weights_only=False, map_location=device)
    
en_vocab.set_vocabulary(vocabs['en_vocabulary'])
ar_vocab.set_vocabulary(vocabs['ar_vocabulary'])

seq2seq_with_attention = torch.load("./seq2seq_with_attention.bin", map_location=device, weights_only=False)

def pre_processor(text):
    preprocessed = utils.preprocess_en(text)
    en_tokens = torch.tensor(en_vocab(preprocessed)).unsqueeze(0).to(device)
    return en_tokens

def post_processor(raw_output):
    return ar_vocab.tokens_to_text(raw_output[1:-1])

@torch.no_grad
def lunch(raw_input, maxtries=30):
    en_tokens = pre_processor(raw_input)
    output = seq2seq_with_attention.translate(en_tokens, maxtries)
    return post_processor(output)


custom_css ='.gr-button {background-color: #bf4b04; color: white;}'
with gr.Blocks(css=custom_css) as demo:
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label='English Sentence')
            gr.Examples(['How are you?',
                         'She is a good girl.',
                         'Who is better than me?!'],
                        inputs=input_text, label="Examples: ")
        with gr.Column():
            output = gr.Textbox(label="Arabic Translation")
            
            start_btn = gr.Button(value='Submit', elem_classes=["gr-button"])
    start_btn.click(fn=lunch, inputs=input_text, outputs=output)

demo.launch()
