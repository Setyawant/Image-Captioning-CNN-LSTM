import streamlit as st
import numpy as np
import os
import re
import json
import requests
from io import BytesIO
from PIL import Image

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Embedding, LSTM, Add, BatchNormalization
)

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# UI Layout
st.set_page_config(page_title="Image Description App", page_icon="🖼️", layout="centered")

# Python Function
def is_valid_url(url):
    """
    Validate whether a given string is a valid URL.

    This function checks if the input string follows the structure of a valid URL, 
    including an optional scheme (http/https), a valid domain, an optional port, 
    and an optional path.

    Parameters:
        url (str): The input string to validate as a URL.

    Returns:
        bool: True if the input is a valid URL, False otherwise.
    """
    regex = re.compile(
        r'^(https?://)?'  # Optional scheme (http or https)
        r'([a-zA-Z0-9.-]+)\.([a-zA-Z]{2,})'  # Domain name validation
        r'(:\d+)?(/.*)?$',  # Optional port and path
        re.IGNORECASE
    )
    return re.match(regex, url) is not None

def idx_to_word(integer, tokenizer):
    """
    Convert an integer index to its corresponding word using the tokenizer's vocabulary.

    This function looks up a given integer in the tokenizer's word index dictionary 
    and returns the corresponding word. If the integer is not found, it returns None.

    Parameters:
        integer (int): The index of the word to look up.
        tokenizer (Tokenizer): The tokenizer containing the word index mapping.

    Returns:
        str or None: The corresponding word if found, otherwise None.
    """
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    """
    Generate a caption for a given image using a trained image captioning model.

    This function takes an image feature vector and iteratively predicts the next word 
    in the sequence until reaching the maximum length or encountering the 'endseq' token.

    Parameters:
        model (keras.Model): The trained image captioning model.
        image (numpy.ndarray): The extracted image feature vector from a CNN.
        tokenizer (Tokenizer): The tokenizer used for text preprocessing.
        max_length (int): The maximum length of the generated caption.

    Returns:
        str: The generated caption without 'startseq' and 'endseq' tokens.
    """
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        
        # Predict the next word
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)  # Get the index of the most probable word
        
        # Convert index to word
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    
    # Remove startseq and endseq tokens
    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption


def generate_caption_from_path(image_path, model, feature_extractor, tokenizer, max_length):
    """
    Generate a caption for an image from a local file path.

    This function loads an image from the specified path, preprocesses it to match 
    the input requirements of the feature extractor model, extracts image features, 
    and then generates a descriptive caption using the trained image captioning model.

    Parameters:
        image_path (str): Path to the image file.
        model (keras.Model): The trained image captioning model.
        feature_extractor (keras.Model): The pretrained CNN model used to extract image features.
        tokenizer (Tokenizer): The tokenizer used to convert words into sequences.
        max_length (int): The maximum length of the generated caption.

    Returns:
        str: The generated caption describing the image.
    """
    # Load and preprocess the image
    image = load_img(image_path, target_size=(384, 384))
    image = img_to_array(image)
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    image = preprocess_input(image) 

    # Extract image features
    image_features = feature_extractor.predict(image, verbose=0)

    # Generate caption
    caption = predict_caption(model, image_features, tokenizer, max_length)
    return caption


def generate_caption_from_url(image_url, model, feature_extractor, tokenizer, max_length):
    """
    Generate a caption for an image from a given URL.

    This function downloads an image from the provided URL, preprocesses it to match 
    the input requirements of the feature extractor model, extracts image features, 
    and then generates a descriptive caption using the trained image captioning model.

    Parameters:
        image_url (str): URL of the image to process.
        model (keras.Model): The trained image captioning model.
        feature_extractor (keras.Model): The pretrained CNN model used to extract image features.
        tokenizer (Tokenizer): The tokenizer used to convert words into sequences.
        max_length (int): The maximum length of the generated caption.

    Returns:
        str: The generated caption describing the image.

    Raises:
        ValueError: If the image cannot be downloaded from the given URL.
    """
    # Download the image from the URL
    response = requests.get(image_url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download image from URL: {image_url}")

    # Load and preprocess the image
    image = Image.open(BytesIO(response.content)).resize((384, 384))
    image = img_to_array(image) 
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    image = preprocess_input(image) 

    # Extract image features
    image_features = feature_extractor.predict(image, verbose=0)

    # Generate caption
    caption = predict_caption(model, image_features, tokenizer, max_length)
    return caption

class ImageCaptioningModel:
    """
    A class to define and build an image captioning model using a pretrained CNN for feature extraction 
    and LSTM for sequence generation.

    The model consists of:
    - A feature extraction pipeline (encoder) for image processing.
    - A text processing pipeline (decoder) for caption generation.
    - A residual connection to improve information flow in the LSTM layers.

    Attributes:
        vocab_size (int): The size of the vocabulary used in the tokenizer.
        max_length (int): The maximum length of the generated sequence.
        model (Model): The compiled Keras model.

    Methods:
        build_model(): Constructs and compiles the image captioning model.
        get_model(): Returns the built model instance.
    """

    def __init__(self, vocab_size, max_length):
        """
        Initializes the ImageCaptioningModel with vocabulary size and maximum sequence length.

        Parameters:
            vocab_size (int): The size of the vocabulary used in the tokenizer.
            max_length (int): The maximum length of the generated sequence.
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.model = self.build_model()

    @staticmethod
    def swish(x):
        """
        Swish activation function.

        Swish is defined as x * sigmoid(x) and is known to perform better than ReLU in deep networks.

        Parameters:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Activated output.
        """
        return x * K.sigmoid(x)

    def build_model(self):
        """
        Builds the image captioning model architecture.

        The model consists of:
        - A feature extractor (encoder) using a dense projection of extracted CNN features.
        - A text processor (decoder) that uses an LSTM-based sequence generator.
        - A final dense layer to predict the next word in the sequence.

        Returns:
            Model: The compiled image captioning model.
        """
        # Encoder (Image Features)
        inputs1 = Input(shape=(1280,), name="image")  # Image feature input
        fe1 = Dense(512, activation=self.swish)(inputs1)  # Feature projection
        fe1 = BatchNormalization()(fe1)
        fe1 = Dropout(0.4)(fe1)
        fe2 = Dense(256, activation=self.swish)(fe1)
        fe2 = BatchNormalization()(fe2)

        # Sequence feature layers (Text Processing)
        inputs2 = Input(shape=(self.max_length,), name="text")  # Text input
        se1 = Embedding(self.vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.4)(se1)
        se3 = LSTM(256, return_sequences=True)(se2)
        se3 = BatchNormalization()(se3)
        se3 = Dropout(0.4)(se3)
        se4 = LSTM(256, return_sequences=True)(se3)
        se4 = BatchNormalization()(se4)
        se4 = Dropout(0.4)(se4)

        # Residual Connection on LSTM
        se5 = Add()([se3, se4])  # Residual connection
        se5 = LSTM(256)(se5)  # Final LSTM output
        se5 = BatchNormalization()(se5)

        # Decoder (Combining Image and Text Features)
        decoder1 = Add()([fe2, se5])
        decoder1 = BatchNormalization()(decoder1)
        decoder2 = Dense(256, activation=self.swish)(decoder1)
        decoder2 = BatchNormalization()(decoder2)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder2)

        # Define and compile the model
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer=Nadam(learning_rate=0.001))

        return model

    def get_model(self):
        """
        Returns the built image captioning model.

        Returns:
            Model: The compiled Keras model.
        """
        return self.model

# Load Feature Extractor Model
@st.cache_resource
def load_feature_extractor():
    """
    Load the pretrained feature extractor model.

    Returns:
        keras.Model: The loaded feature extraction model.
    """
    model_path = "model_feature_extractor.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Feature extractor model not found: {model_path}")

    try:
        return load_model(model_path, compile=False)
    except Exception as e:
        raise RuntimeError(f"Error loading feature extractor model: {e}")

# Load Image Captioning Model
@st.cache_resource
def load_captioning_model(vocab_size, max_length):
    """
    Load the image captioning model with predefined architecture and weights.

    Parameters:
        vocab_size (int): The size of the vocabulary.
        max_length (int): The maximum sequence length for captions.

    Returns:
        keras.Model: The loaded image captioning model with weights.
    """
    model = ImageCaptioningModel(vocab_size, max_length).get_model()
    weights_path = "image_captioning_weights.weights.h5"

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Captioning model weights not found: {weights_path}")

    try:
        model.load_weights(weights_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading image captioning model weights: {e}")

# Load Tokenizer
@st.cache_resource
def load_tokenizer():
    """
    Load the tokenizer from a JSON file.

    Returns:
        Tokenizer: The loaded tokenizer.
    """
    tokenizer_path = "model_tokenizer.json"

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    try:
        with open(tokenizer_path, "r") as file:
            tokenizer_dict = json.load(file)
        tokenizer_json = json.dumps(tokenizer_dict)
        return tokenizer_from_json(tokenizer_json)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Error decoding tokenizer JSON: {e}")

# Define Variables
vocab_size = 8768
max_length = 34

# Load all models
feature_extractor = load_feature_extractor()
captioning_model = load_captioning_model(vocab_size, max_length)
tokenizer = load_tokenizer()

# Application Title & Description
st.title("🖼️ Image Descriptor App")
st.write("""
Welcome to the **Image Descriptor App**!

Upload an image or enter an image URL, and the system will generate a descriptive sentence based on the content. 
This application uses a **pretrained CNN for feature extraction** and **LSTM for text generation**. 
The model was trained using the **Flickr8K dataset**, which contains 8,000 images with multiple human-annotated descriptions.  
""")


# Tabs for Image Upload & URL Input
tab1, tab2, tab3 = st.tabs(["🏠 About This App", "📂 Upload Image", "🌐 Image URL"])

with tab1:
    st.subheader("📝 Application Description")
    st.write("""
    **Image Descriptor App** is a system designed to automatically generate **descriptive text** for images.  
    Users can either **upload an image** directly or **provide an image URL**, and the system will generate a **textual description** based on the image content.

    This application utilizes a **pretrained Convolutional Neural Network (CNN)** to extract important visual features, 
    which are then processed by **LSTM (Long Short-Term Memory)** to generate **meaningful descriptions**.

    The goal of this app is to help users **interpret images more effectively**, making it useful for **accessibility, content automation**, and various other applications.
    """)

    st.subheader("🔧 Features")
    st.write("""
    ✅ Upload an image or enter an image URL\n
    ✅ Automatic image description generation\n
    ✅ Displays the uploaded image alongside its generated description\n
    ✅ User-friendly interface with an intuitive layout\n
    ✅ Built with Streamlit for seamless interaction
    """)

with tab2:
    st.subheader("📤 Upload an Image")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "webp"])
    
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        # Konversi file ke format yang bisa dibaca sebagai path
        image_bytes = BytesIO(uploaded_image.getvalue())
        
        if st.button("Generate Description for Uploaded Image"):
            st.subheader("📝 Generated Description")
            # Generate caption
            uploaded_image_capt = generate_caption_from_path(image_bytes, captioning_model, feature_extractor, tokenizer, max_length)
            st.success(f"**{uploaded_image_capt}**")



with tab3:
    st.subheader("🔗 Enter Image URL")
    image_url = st.text_input("Enter the image URL here:")

    if image_url:
        if is_valid_url(image_url):
            st.image(image_url, caption="Image from URL", use_container_width=True)

            if st.button("Generate Description for URL Image"):
                st.subheader("📝 Generated Description")
                url_image_capt = generate_caption_from_url(image_url, captioning_model, feature_extractor, tokenizer, max_length)
                st.success(f"**{url_image_capt}**")
        else:
            st.error("🚨 Please enter a valid image URL!")  # Error jika URL tidak valid

# Model Info & Footer
st.markdown("---")

st.markdown(
    "<p style='text-align: center;'>🔍 <b>Model Used:</b> Pretrained CNN (Feature Extraction) + LSTM (Description Generation)</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>📊 <b>Performance:</b> BLEU-1: 0.628 | BLEU-2: 0.404 | BLEU-3: 0.269 | BLEU-4: 0.168</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>📌 <b>Built with Streamlit & TensorFlow</b> | 🤖 <b>Powered by Deep Learning</b></p>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
    
    <p style='text-align: center;'>
        © 2025 <b>Personal Project - Ach. Arif Setiawan</b>
    </p>
    <p style='text-align: center;'>
        <a href="https://www.linkedin.com/in/ach-arif-setiawan" target="_blank" style="color: #0073b1; text-decoration: none; margin-right: 15px;">
            <i class="fa-brands fa-linkedin"></i>
        </a>
        <a href="https://github.com/Setyawant" target="_blank" style="color: #171515; text-decoration: none;">
            <i class="fa-brands fa-github"></i>
        </a>
    </p>
    """,
    unsafe_allow_html=True
)