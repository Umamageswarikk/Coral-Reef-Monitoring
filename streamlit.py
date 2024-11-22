

import os
import random
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from streamlit_option_menu import option_menu  # Import option_menu for the navbar 
from PIL import Image
import base64
from io import BytesIO



st.title("Monitoring coral reef ecosystem using deeplearning")

# Add option menu as the navigation bar at the top
selected = option_menu(
    None, 
    ["About", "Developer", "Dataset", "Metrics", "Prediction"],  # Added all options
    icons=["info", "person", "graph-up-arrow", "bar-chart", "images", "cloud-upload"],
    default_index=0,  # Default selected index, here it's "About"
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0!important", 
            "margin": "0 auto", 
            "width": "100%",  # Make the container 100% width to span full screen
            "height": "50px",  # Adjust height
            "display": "flex",  # Use flexbox to align items in a row
            "justify-content": "space-around"  # Distribute the items evenly across the container
        },
        "icon": {"color": "#fa6607", "font-size": "20px"},
        "nav-link": {
            "font-size": "14px",  # Smaller font size
            "text-align": "center", 
            "margin": "0px", 
            "--hover-color": "#eee", 
            "padding": "10px",  # Adjust padding
            "color": "orange",  # Set default color for the text
            "white-space": "nowrap"  # Prevent text wrapping
        },
        "nav-link-selected": {"background-color": "#fa6607", "color": "white"},
        "nav-link:hover": {  # Add hover effect for the nav link
            "color": "black"  # Change text color to black on hover
        }
    }
)



# Load paths to your image folders
healthy_folder = r'C:\MCA\5th trimester\coral-reef-bleaching-main\coral-reef-bleaching-main\cnn\archive (5)\healthy_corals'
bleached_folder = r'C:\MCA\5th trimester\coral-reef-bleaching-main\coral-reef-bleaching-main\cnn\archive (5)\bleached_corals'

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r'C:\MCA\5th trimester\coral-reef-bleaching-main\coral-reef-bleaching-main\cnn\cnn_model.h5')

model = load_model()

# Function to plot the dataset distribution (for Dataset Analysis page)
def plot_sample_analysis():
    healthy_images = len(os.listdir(healthy_folder))
    bleached_images = len(os.listdir(bleached_folder))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.pie([healthy_images, bleached_images], labels=['Healthy Coral', 'Bleached Coral'],
            autopct='%1.1f%%', startangle=90, colors=['cyan', 'pink'])
    ax1.set_title('Dataset Distribution')

    ax2.bar(['Healthy Coral', 'Bleached Coral'], [healthy_images, bleached_images], color=['cyan', 'pink'])
    ax2.set_title('Number of Images per Class')
    ax2.set_ylabel('Image Count')
    st.pyplot(fig)

# Function to evaluate the model (for Model Evaluation page)
def evaluate_model():
    pred = np.argmax(model.predict(all_images), axis=1)
    # Display classification report as table
    report_dict = classification_report(all_labels, pred, target_names=['Healthy Coral', 'Bleached Coral'], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.write("### Classification Report")
    st.dataframe(report_df.style.format(precision=2))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy Coral', 'Bleached Coral'],
                yticklabels=['Healthy Coral', 'Bleached Coral'])
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig)

# Function to display random samples from a folder (for Random Samples page)
def show_samples(folder, label, num_samples=20):
    samples = random.sample(os.listdir(folder), num_samples)
    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    for i, sample in enumerate(samples):
        img_path = os.path.join(folder, sample)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i // 5, i % 5].imshow(img)
        axes[i // 5, i % 5].set_title(label)
        axes[i // 5, i % 5].axis('off')
    st.pyplot(fig)


# Function to preprocess images for display and prediction
def preprocess_image(img_path, img_size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0  # Normalize pixel values
    return img

# Function to load and preprocess images for training and testing
def load_images_for_analysis():
    healthy_images = [preprocess_image(os.path.join(healthy_folder, img)) for img in os.listdir(healthy_folder)]
    bleached_images = [preprocess_image(os.path.join(bleached_folder, img)) for img in os.listdir(bleached_folder)]
    healthy_labels = [0] * len(healthy_images)
    bleached_labels = [1] * len(bleached_images)
    all_images = np.array(healthy_images + bleached_images)
    all_labels = np.array(healthy_labels + bleached_labels)
    return all_images, all_labels

all_images, all_labels = load_images_for_analysis()

# Prediction function
def predict_image(img_path):
    img = preprocess_image(img_path)
    img_expanded = np.expand_dims(img, axis=0)  # Expand dimensions to (1, height, width, channels)
    prediction = model.predict(img_expanded)
    class_idx = np.argmax(prediction)
    class_label = "Healthy Coral" if class_idx == 0 else "Bleached Coral"
    return class_label



# About Page
def show_about_page():
    
    st.title("Introduction")

    # Display Image
    st.markdown("### Visualizing AI-Powered Monitoring")
    
    image_path = r"C:\MCA\5th trimester\coral-reef-bleaching-main\coral-reef-bleaching-main\cnn\image.png"  # Replace with the path to your image file
    image = Image.open(image_path)
    st.image(image, caption="An AI-powered monitoring system for coral reefs using CNNs.", use_column_width=True)

    # Introduction Section
    st.write("""
    ## Introduction
    
    Coral reefs are critical ecosystems that support marine biodiversity and provide numerous environmental benefits. However, they are under threat from climate change, pollution, and human activity. Monitoring coral health is essential to preserve these ecosystems, but manual observation can be time-consuming and challenging.
    
    ### Coral Reef Bleaching Classification
    
    This application classifies coral images into two categories: **Healthy Coral** and **Bleached Coral**.
    
    The goal of this project is to support coral reef health monitoring by using CNNs to classify coral images whether they are healthy or bleached, providing actionable insights into reef conservation.
    """)




import streamlit as st
from PIL import Image
import base64
from io import BytesIO

def image_to_base64(image):
    """Convert image to base64 encoding."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()

def show_developer_page():
    st.markdown("<h2 style='text-align: center;'>Developers</h2>", unsafe_allow_html=True)

    # Define paths for each participant's image
    image_paths = [
        r"C:\MCA\5th trimester\coral-reef-bleaching-main\coral-reef-bleaching-main\cnn\priyanga.jpg",  # Replace with actual paths
        r"C:\MCA\3rd trimester\advanced python\image processing\input_imgs\image_0016.jpg",
        r"C:\MCA\5th trimester\coral-reef-bleaching-main\coral-reef-bleaching-main\cnn\image_0029.jpg"
    ]

    # Define participant details
    participants = [
        {
            "name": "Priyanga K",
            "institution": "Vellore Institute of Technology, Chennai",
            "phone": "6379314514",
            "email": "santhoshkumar150822@gmail.com",
            "linkedin": "https://www.linkedin.com/in/santhosh-kumar-150822-p",
            "github": "https://github.com/SanthoshKumar150822",
            "image_path": image_paths[0]
        },
        {
            "name": "Participant 2",
            "institution": "Vellore Institute of Technology, Chennai",
            "phone": "1234567890",
            "email": "participant2@example.com",
            "linkedin": "https://www.linkedin.com/in/participant2",
            "github": "https://github.com/participant2",
            "image_path": image_paths[1]
        },
        {
            "name": "Participant 3",
            "institution": "Vellore Institute of Technology, Chennai",
            "phone": "0987654321",
            "email": "participant3@example.com",
            "linkedin": "https://www.linkedin.com/in/participant3",
            "github": "https://github.com/participant3",
            "image_path": image_paths[2]
        }
    ]

    # Load CSS for circular image with "cover" scaling
    st.markdown(
        """
        <style>
        .circle-img {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            overflow: hidden;
            margin: 0 auto;
            box-shadow: 0 0 15px rgba(252, 176, 69, 0.8), 0 0 25px rgba(252, 176, 69, 0.8);
            animation: glow 1.5s infinite alternate;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .circle-img img {
            width: 100%;
            height: 100%;
            object-fit: cover; /* Ensures image covers the circle without distortion */
        }

        @keyframes glow {
            0% { box-shadow: 0 0 10px rgba(252, 176, 69, 0.8), 0 0 20px rgba(252, 176, 69, 0.8); }
            100% { box-shadow: 0 0 20px rgba(252, 176, 69, 1), 0 0 30px rgba(252, 176, 69, 1); }
        }

        .content {
            text-align: center;
        }

        h3, h4 {
            margin: 5px 0;
        }

        ul {
            list-style-type: none;
            padding: 0;
            text-align: left;
        }

        li {
            margin: 3px 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display participants in columns
    cols = st.columns(3)
    for col, participant in zip(cols, participants):
        with col:
            # Load image and convert to base64
            image = Image.open(participant["image_path"])
            image_base64 = image_to_base64(image)

            # Display circular glowing image
            st.markdown(
                f"""
                <div class="circle-img">
                    <img src="data:image/jpeg;base64,{image_base64}" alt="{participant['name']}" />
                </div>
                """,
                unsafe_allow_html=True
            )

            # Display participant details
            st.markdown(
                f"""
                <div class="content">
                    <h3>{participant['name']}</h3>
                    <p>{participant['institution']}</p>
                    <h4>Contact Details:</h4>
                    <ul>
                        <li><strong>Phone:</strong> {participant['phone']}</li>
                        <li><strong>Email:</strong> <a href="mailto:{participant['email']}">{participant['email']}</a></li>
                        <li><strong>LinkedIn:</strong> <a href="{participant['linkedin']}" target="_blank">LinkedIn</a></li>
                        <li><strong>GitHub:</strong> <a href="{participant['github']}" target="_blank">GitHub</a></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )


# Dataset Analysis Page
def show_dataset_analysis():
    st.title("Random Samples")
    st.subheader("Healthy Coral Samples")
    show_samples(healthy_folder, 'Healthy Coral')
    st.subheader("Bleached Coral Samples")
    show_samples(bleached_folder, 'Bleached Coral')
    st.title("Dataset Analysis")
    plot_sample_analysis()


# Model Evaluation Page
def show_model_evaluation():
    st.title("Model Evaluation")
    evaluate_model()




# Predict Image Page
def show_predict_image():
    st.title("Predict Image")
    uploaded_file = st.file_uploader("Upload an image of coral for prediction", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img_array = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb_resized = cv2.resize(img_rgb, (224, 224)) / 255.0

        # Show the uploaded image
        st.image(img_rgb, caption="Uploaded Coral Image", use_column_width=True)
        
        # Predict the label
        prediction = model.predict(np.expand_dims(img_rgb_resized, axis=0))
        class_label = "Healthy Coral" if np.argmax(prediction) == 0 else "Bleached Coral"
        st.write(f"Prediction: {class_label}")

# Main logic to render the selected page
if selected == "About":
    show_about_page()
elif selected == "Developer":
    show_developer_page()
elif selected == "Dataset":
    show_dataset_analysis()
elif selected == "Metrics":
    show_model_evaluation()
elif selected == "Prediction":
    show_predict_image()

