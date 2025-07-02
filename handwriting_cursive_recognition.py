    # -*- coding: utf-8 -*-
import streamlit as st
from streamlit_cropper import st_cropper
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
import idx2numpy

# Configuration
st.set_page_config(page_title="Cursive Handwriting Recognition", layout="wide")

# Load EMNIST data with caching
@st.cache_resource
def load_emnist():
    train_images = idx2numpy.convert_from_file('emnist_data/emnist-byclass-train-images-idx3-ubyte')
    train_labels = idx2numpy.convert_from_file('emnist_data/emnist-byclass-train-labels-idx1-ubyte')
    test_images = idx2numpy.convert_from_file('emnist_data/emnist-byclass-test-images-idx3-ubyte')
    test_labels = idx2numpy.convert_from_file('emnist_data/emnist-byclass-test-labels-idx1-ubyte')

    # Normalize and reshape
    train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # Convert labels to one-hot encoding
    num_classes = 62  # 26 uppercase + 26 lowercase + 10 digits
    train_labels = keras.utils.to_categorical(train_labels, num_classes)
    test_labels = keras.utils.to_categorical(test_labels, num_classes)

    return train_images, train_labels, test_images, test_labels

# Enhanced model configuration
def build_optimized_model(input_shape=(28, 28, 1), num_classes=62):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# Image preprocessing pipeline
def preprocess_image(image):
    img = np.array(image.convert('L'))
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cleaned = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
    cleaned = cleaned.astype('float32') / 255.0
    return cleaned.reshape(1, 28, 28, 1)

# Character segmentation for cursive handwriting
def segment_cursive_characters(image_pil):
    image = np.array(image_pil.convert('L'))  # Grayscale
    st.image(image, caption="Grayscale", width=200)

    # Thresholding
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    st.image(thresh, caption="Binary (OTSU)", width=200)

    characters = []

    # Line segmentation using horizontal projection
    h_proj = np.sum(thresh, axis=1)
    lines = []
    in_line = False
    for y, val in enumerate(h_proj):
        if val > 0 and not in_line:
            start_y = y
            in_line = True
        elif val == 0 and in_line:
            end_y = y
            lines.append((start_y, end_y))
            in_line = False
    if in_line:
        lines.append((start_y, thresh.shape[0]))

    for i, (y0, y1) in enumerate(lines):
        line_img = thresh[y0:y1, :]
        st.image(line_img, caption=f"Line {i+1}", width=200)

    # Word segmentation using vertical projection
    v_proj = np.sum(line_img, axis=0)
    in_word = False
    words = []
    for x, val in enumerate(v_proj):
        if val > 0 and not in_word:
            start_x = x
            in_word = True
        elif val == 0 and in_word:
            end_x = x
            words.append((start_x, end_x))
            in_word = False
    if in_word:
        words.append((start_x, line_img.shape[1]))

    for j, (x0, x1) in enumerate(words):
        word_img = line_img[:, x0:x1]
        st.image(word_img, caption=f"Word {j+1}", width=200)

        # Find character contours inside word
        contours, _ = cv2.findContours(word_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        char_imgs = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 4 or h < 4 or w * h < 100:
                continue  # skip tiny noise

            char = word_img[y:y+h, x:x+w]

            # 🚫 Removed deskew / rotation here

            # Resize and normalize
            char = cv2.resize(char, (28, 28), interpolation=cv2.INTER_AREA)
            char = char.astype('float32') / 255.0
            char_imgs.append((x, char.reshape(1, 28, 28, 1)))

        # Sort characters left to right
        char_imgs.sort(key=lambda tup: tup[0])

        for idx, (_, char_img) in enumerate(char_imgs):
            st.image(char_img.reshape(28, 28), caption=f"Char {idx+1}", width=64)
            characters.append(char_img)


    return characters


# Main app
def main():
    st.title("Cursive Handwriting Recognition")
    st.markdown("""
    This system is designed to recognize cursive handwriting using a CNN model trained on EMNIST.
    """)

    # Load EMNIST data
    X_train, y_train, X_test, y_test = load_emnist()

    # Use new model file name to avoid loading old model with augmentation
    model_file = 'cursive_recognition_model_noDataGen.h5'

    if os.path.exists(model_file):
        st.info("Loading model...")
        model = keras.models.load_model(model_file)
    else:
        st.warning("Training new model without data augmentation...")
        model = build_optimized_model()

        # Split manually for validation
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42)

        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            min_delta=0.005,
            restore_best_weights=True
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )

        # Train progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        class ProgressCallback(callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                status_text.text(f"Epoch {epoch+1}/30 completed")
                progress_bar.progress((epoch+1) / 30)

        # Train the model directly
        history = model.fit(
            X_train_split, y_train_split,
            validation_data=(X_val_split, y_val_split),
            epochs=30,
            batch_size=128,
            callbacks=[early_stopping, reduce_lr, ProgressCallback()],
            verbose=1
        )

        # Save model
        model.save(model_file)

        # Show training history
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(history.history['accuracy'], label='Train')
        ax.plot(history.history['val_accuracy'], label='Validation')
        ax.set_title("Model Accuracy")
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Epoch")
        ax.legend()
        st.pyplot(fig)


    uploaded_file = st.file_uploader("Upload Handwritten Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=500)

        st.subheader("Select a character manually:")
        cropped_img = st_cropper(image, box_color='red', aspect_ratio=None)

        # Button 1: Automatic segmentation
        if st.button("Recognize Text Without Selecting"):
            with st.spinner("Processing..."):
                characters = segment_cursive_characters(image)

                # Show segmented characters
                st.subheader("Segmented Characters")
                char_cols = st.columns(len(characters))
                for idx, char in enumerate(characters):
                    with char_cols[idx]:
                        st.image(char.reshape(28, 28), caption=f"Char {idx+1}", width=64)

                predictions = []
                for char in characters:
                    pred = model.predict(char)
                    pred_class = np.argmax(pred)
                    predictions.append(pred_class)

                # Map classes to characters
                chars = list('0123456789') + \
                        [chr(i) for i in range(ord('A'), ord('Z')+1)] + \
                        [chr(i) for i in range(ord('a'), ord('z')+1)]

                recognized_text = ''.join([chars[p] for p in predictions])
                st.subheader("Recognition Results")
                st.code(recognized_text, language="text")

        # Button 2: Manual crop recognition
        if st.button("Recognize Cropped Character"):
            with st.spinner("Processing cropped area..."):
                processed = preprocess_image(cropped_img)
                prediction = model.predict(processed)
                pred_class = np.argmax(prediction)

                chars = list('0123456789') + \
                        [chr(i) for i in range(ord('A'), ord('Z')+1)] + \
                        [chr(i) for i in range(ord('a'), ord('z')+1)]

                st.success(f"Predicted Character: {chars[pred_class]}")
                st.image(processed.reshape(28, 28), caption="Cropped Character", width=128)


if __name__ == "__main__":
    main()









# import streamlit as st
# import easyocr
# import cv2
# import numpy as np

# # Initialize the EasyOCR reader
# reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have a compatible GPU

# def load_image(image_file):
#     # Convert the uploaded file to an OpenCV image
#     image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
#     return image

# def recognize_text(image):
#     # Use EasyOCR to recognize text in the image
#     results = reader.readtext(image)
#     text = " ".join([result[1] for result in results])
#     return text

# def calculate_accuracy(original_text, recognized_text):
#     # Calculate the accuracy of the recognized text
#     original_words = original_text.split()
#     recognized_words = recognized_text.split()
#     correct_count = sum(1 for o, r in zip(original_words, recognized_words) if o == r)
#     accuracy = (correct_count / len(original_words)) * 100 if original_words else 0
#     return accuracy

# # Streamlit UI
# st.title("Handwriting Cursive Text Recognition")
# st.write("Upload an image of handwritten text:")

# # File uploader
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Load and display the image
#     image = load_image(uploaded_file)
#     st.image(image, caption='Uploaded Image', use_column_width=True)

#     # Recognize text
#     recognized_text = recognize_text(image)
#     st.write("Recognized Text:")
#     st.write(recognized_text)

#     # Calculate and display accuracy
#     original_text = st.text_input("Enter the original text for accuracy calculation:")
#     if original_text:
#         accuracy = calculate_accuracy(original_text, recognized_text)
#         st.write(f"Accuracy: {accuracy:.2f}%")




# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# # Configure page
# st.set_page_config(
#     page_title="Handwriting Recognition with TrOCR",
#     page_icon="✍️",
#     layout="wide"
# )

# # Load model (cached)
# @st.cache_resource
# def load_model():
#     processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
#     model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
#     return processor, model

# def preprocess_image(uploaded_file):
#     # Convert to PIL Image
#     image = Image.open(uploaded_file).convert('RGB')
    
#     # Display original image
#     with st.expander("Original Image", expanded=True):
#         st.image(image, caption="Uploaded Handwriting Sample", use_column_width=True)
    
#     return image

# def main():
#     st.title("✍️ Handwritten Text Recognition with TrOCR")
#     st.markdown("""
#     Upload an image of handwritten text to recognize using Microsoft's Transformer-based OCR (TrOCR) model.
#     """)
    
#     # Initialize
#     if 'predictions' not in st.session_state:
#         st.session_state.predictions = []
    
#     # Model loading
#     with st.spinner("Loading AI model..."):
#         processor, model = load_model()
    
#     # File uploader
#     uploaded_file = st.file_uploader(
#         "Choose a handwriting image", 
#         type=["jpg", "jpeg", "png"],
#         accept_multiple_files=False
#     )
    
#     if uploaded_file is not None:
#         try:
#             # Process image
#             image = preprocess_image(uploaded_file)
            
#             # Predict text
#             with st.spinner("Recognizing handwriting..."):
#                 # Preprocess
#                 pixel_values = processor(image, return_tensors="pt").pixel_values
                
#                 # Generate predictions
#                 generated_ids = model.generate(pixel_values)
#                 recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
#                 # Store prediction
#                 st.session_state.predictions.append(recognized_text)
                
#             # Show results
#             st.subheader("Recognized Text")
#             st.code(recognized_text, language="text")
            
#             # Confidence estimation (approximate)
#             text_length = max(1, len(recognized_text))
#             confidence = min(95, 70 + text_length * 0.5)  # Simple approximation
            
#             # Metrics
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.metric("Estimated Confidence", f"{confidence:.1f}%")
#             with col2:
#                 st.metric("Text Length", f"{text_length} chars")
            
#             st.info("ℹ️ Note: Accuracy varies with handwriting quality. Expect 60-80% accuracy for cursive text.")
        
#         except Exception as e:
#             st.error(f"Error processing image: {str(e)}")
#             st.stop()

# if __name__ == "__main__":
#     main()
