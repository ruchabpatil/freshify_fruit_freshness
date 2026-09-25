
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
from pathlib import Path


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Freshify | Fruit Freshness",
    page_icon="🍓",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #fff5f8 0%,
            #fffafd 50%,
            #fdf2f8 100%
        );
    }

    /* Hide Streamlit default menu and footer */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Main content width */
    .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .brand {
        text-align: center;
        color: #d63384;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .tagline {
        text-align: center;
        color: #79566a;
        font-size: 17px;
        margin-bottom: 1.8rem;
    }

    /* Section cards */
    .custom-card {
        background-color: rgba(255, 255, 255, 0.88);
        padding: 1.5rem;
        border-radius: 22px;
        border: 1px solid #f8d7e5;
        box-shadow: 0 5px 20px rgba(214, 51, 132, 0.07);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* Result cards */
    .fresh-result {
        background: #ecfdf3;
        border: 1px solid #a7f3d0;
        border-radius: 18px;
        padding: 1.2rem;
        text-align: center;
        color: #166534;
        font-size: 25px;
        font-weight: 700;
    }

    .rotten-result {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        border-radius: 18px;
        padding: 1.2rem;
        text-align: center;
        color: #be123c;
        font-size: 25px;
        font-weight: 700;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background-color: #fff7fb;
        border: 2px dashed #f3a6c8;
        border-radius: 18px;
        padding: 1rem;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 14px;
        border: none;
        background: linear-gradient(
            90deg,
            #d63384,
            #ec4899
        );
        color: white;
        font-size: 17px;
        font-weight: 700;
        padding: 0.7rem;
        transition: 0.3s;
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            #be185d,
            #db2777
        );
        color: white;
    }

    /* Small text */
    .small-text {
        text-align: center;
        color: #8b7180;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="brand">🍓 Freshify</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="tagline">'
    'AI-Based Multi-Fruit Freshness Classification'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="custom-card">
        <h3 style="color:#d63384; text-align:center;">
            🌷 Welcome to Freshify
        </h3>
        <p style="text-align:center; color:#79566a;">
            Upload an image of a fruit and let our neural network
            estimate whether it appears fresh or rotten.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "multi_fruit_mlp_baseline.keras"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error("Unable to load the trained model.")
    st.exception(error)
    st.stop()


# --------------------------------------------------
# Image Upload
# --------------------------------------------------

st.subheader("📸 Upload Your Fruit Image")

uploaded_file = st.file_uploader(
    "Choose an image from your computer",
    type=None,
    help="Upload JPG, JPEG, PNG, WEBP, or JFIF images"
)

if uploaded_file is not None:

    try:
        # Open uploaded image
        image = Image.open(uploaded_file).convert("RGB")

        st.markdown(
            '<div class="custom-card">',
            unsafe_allow_html=True
        )

        st.subheader("🖼️ Image Preview")

        st.image(
            image,
            caption="Your Selected Fruit",
            use_container_width=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        # Prediction button
        predict_clicked = st.button(
            "🔍 Predict Freshness"
        )

        if predict_clicked:

            with st.spinner("Analyzing your fruit image..."):

                # Resize image
                resized_image = image.resize((32, 32))

                # Convert to NumPy array
                image_array = np.array(
                    resized_image,
                    dtype=np.float32
                )

                # Normalize pixel values
                image_array = image_array / 255.0

                # Flatten image
                image_flattened = image_array.reshape(1, 3072)

                # Model prediction
                prediction = model.predict(
                    image_flattened,
                    verbose=0
                )[0][0]

                prediction = float(prediction)

                # Interpret prediction
                if prediction >= 0.5:
                    predicted_class = "Rotten"
                else:
                    predicted_class = "Fresh"

                fresh_score = 1.0 - prediction
                rotten_score = prediction

            # --------------------------------------------------
            # Prediction Results
            # --------------------------------------------------

            st.subheader("🌸 Prediction Result")

            if predicted_class == "Fresh":

                st.markdown(
                    """
                    <div class="fresh-result">
                        🍏 Fresh
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="rotten-result">
                        🍂 Rotten
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")

            # Score columns
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="🍏 Fresh Score",
                    value=f"{fresh_score * 100:.2f}%"
                )

            with col2:
                st.metric(
                    label="🍂 Rotten Score",
                    value=f"{rotten_score * 100:.2f}%"
                )

            # Progress bars
            st.write("**Fresh Score**")
            st.progress(fresh_score)

            st.write("**Rotten Score**")
            st.progress(rotten_score)

            # Explanation
            st.info(
                "The scores represent the model's output for each class. "
                "They are not calibrated probabilities or a guarantee "
                "of food safety."
            )

    except Exception as error:

        st.error(
            "This file could not be processed as an image."
        )

        st.exception(error)

else:

    st.markdown(
        """
        <div class="custom-card">
            <p style="text-align:center; color:#8b7180;">
                🌷 Your fruit image will appear here after uploading.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.write("")

st.markdown(
    """
    <div class="small-text">
        Made with 💗 using Python, TensorFlow, and Streamlit
        <br>
        M.Tech AI & ML Project | LCA1
    </div>
    """,
    unsafe_allow_html=True
)