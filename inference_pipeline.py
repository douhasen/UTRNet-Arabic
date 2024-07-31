import streamlit as st
import math
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import Model  
from utils import CTCLabelConverter, AttnLabelConverter  
from dataset import NormalizePAD  
from datetime import datetime
import pytz
import argparse  

# Function to load and preprocess the image
def load_image(image_file):
    img = Image.open(image_file)
    return img



# Function to perform OCR prediction
def predict_text(image, model, converter, device, opt):
    img = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    w, h = img.size
    ratio = w / float(h)
    if math.ceil(opt.imgH * ratio) > opt.imgW:
        resized_w = opt.imgW
    else:
        resized_w = math.ceil(opt.imgH * ratio)
    img = img.resize((resized_w, opt.imgH), Image.Resampling.BICUBIC)
    transform = NormalizePAD((1, opt.imgH, opt.imgW))
    img = transform(img)
    img = img.unsqueeze(0)
    # print(img.shape) # torch.Size([1, 1, 32, 400])
    batch_size = img.shape[0] # 1
    img = img.to(device)
    preds = model(img)
    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
    
    _, preds_index = preds.max(2)
    preds_str = converter.decode(preds_index.data, preds_size.data)[0]
    
    return preds_str

# Streamlit code
def main():
    st.title('OCR Model Interface')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = load_image(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Load model and configuration
        opt = load_model_config()  # Function to load model configuration
        device = torch.device('cpu')

        if 'CTC' in opt.Prediction:
            converter = CTCLabelConverter(opt.character)
        else:
            converter = AttnLabelConverter(opt.character)
        opt.num_class = len(converter.character)
        
        model = Model(opt)
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))
        model.eval()
        model = model.to(device)

        # Perform prediction
        if st.button('Predict'):
            with st.spinner('Predicting...'):
                result_text = predict_text(image, model, converter, device, opt)
                st.write(f'Predicted Text: {result_text}')

# Function to load model configuration (similar to read.py)
def load_model_config():
    opt = argparse.Namespace()
    opt.saved_model = "saved_models/UTRNet-Large/best_norm_ED.pth"
    opt.batch_max_length = 100
    opt.imgH = 32
    opt.imgW = 400
    opt.rgb = False
    opt.FeatureExtraction = "HRNet"
    opt.SequenceModeling = "DBiLSTM"
    opt.Prediction = "CTC"
    opt.num_fiducial = 20
    opt.input_channel = 1
    opt.output_channel = 32  # Adjusted based on HRNet feature extraction
    opt.hidden_size = 256

    # Load characters/vocab
    with open("ArabGlyphs.txt", "r", encoding="utf-8") as file:
        content = file.readlines()
        content = ''.join([str(elem).strip('\n') for elem in content])
        opt.character = content+" "

    # Set device to CPU
    opt.device = 'cpu'

    return opt

if __name__ == '__main__':
    main()


