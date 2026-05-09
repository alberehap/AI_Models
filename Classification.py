import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(
    page_title="Image Classification App",
    layout="centered"
)

st.title("Image Classification App")
st.write("Upload an image and let AI classify it.")

classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224"
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

   
    with st.spinner("Classifying image..."):


        results = classifier(image)

    st.subheader("Classification Results")

    for result in results[:5]:

        label = result["label"]
        score = result["score"]

        st.write(f"### {label}")

        st.progress(float(score))

        st.write(f"Confidence: {score * 100:.2f}%")