import streamlit as st
import PIL
from model_complex import ModelComplex

model = ModelComplex('weights/yolo.pt', 'weights/cp.ckpt') 

# Setting page layout
st.set_page_config(
    page_title="Object Detection and Monocular Depth Estimation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.sidebar.header("Model Config")
confidence = float(st.sidebar.slider("Minimum confidence", 25, 100, 40)) / 100

source_img = None 
try:
    source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp', 'png'))
    if source_img is not None:
        source_img = PIL.Image.open(source_img)
        st.image(source_img, caption="Uploaded Image", use_column_width=True)
        if st.sidebar.button('Run'):
            img1, img2, img3 = model([source_img], conf=confidence)[0]
            st.image(img1, caption='Object detection', use_column_width=True)
            st.image(img2, caption='Depth', use_column_width=True)
            st.image(img3, caption='Object with depth', use_column_width=True)
except Exception as ex:
    st.error(ex)

