
# WIND: A Verifiable Zero-Watermark Framework for Protecting Creative Textual Data

This repository is the official implementation of our WIND, a novel verifiable and implicit watermarking scheme designed to safeguard the copyright of creative writing databases.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## Arguments -- tool.py
>You can change arguments setting in tool.py

## Model Framework
### Encoder
>You replace the encoder by changing the args.encoder, while WIND utilize 'sup-simcse-bert-base-uncased'.

### cons_model.py
>Contrastive learning.
### prompt_wm_gpt.py
>Construction of creative essence as condensed-lists leverage a LLM. You should fill your own API key in corresponding position.
### wmSpace.py
>Map extracted creative essence to a watermark.
