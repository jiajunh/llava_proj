import streamlit as st

from utils import get_one_image

img_path = get_one_image(image_path="test_data")

st.set_page_config(page_title="Hello", page_icon=":material/waving_hand:")
st.title("Welcome to Streamlit!")
st.write(
    img_path
)