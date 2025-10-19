import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---------------------------
# Helper Functions
# ---------------------------
def load_image(uploaded_file):
    image = Image.open(uploaded_file)
    return np.array(image)

def save_image(image, filename="processed_image.png"):
    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return filename

def get_image_info(image, uploaded_file):
    h, w, c = image.shape
    size_kb = len(uploaded_file.getbuffer()) / 1024 if uploaded_file else 0
    return {
        "Dimensions": f"{w} x {h}",
        "Channels": c,
        "File Size": f"{size_kb:.2f} KB",
        "Format": uploaded_file.type if uploaded_file else "N/A"
    }

# ---------------------------
# Streamlit GUI Layout
# ---------------------------
st.set_page_config(page_title="VisionLab Toolkit", layout="wide")
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>✨ VisionLab: Interactive Image Processing Toolkit ✨</h1>", unsafe_allow_html=True)

# File Upload Section
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png", "bmp"], label_visibility="collapsed")

if uploaded_file:
    image = load_image(uploaded_file)

    # Tabs for Categories
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "ℹ️ Info", "🎨 Colors", "🔄 Transforms", "🧹 Filtering", 
        "⚡ Enhancement", "✂️ Edges", "💾 Compression"
    ])

    processed_image = image.copy()

    # ---- Info Tab ----
    with tab1:
        st.subheader("Image Information")
        st.image(image, caption="Original Image", use_container_width=True)
        st.json(get_image_info(image, uploaded_file))

    # ---- Color Conversion ----
    with tab2:
        choice = st.selectbox("Choose Conversion", [
            "RGB → Grayscale", "RGB → HSV", "RGB → YCbCr", "BGR ↔ RGB"
        ])
        if choice == "RGB → Grayscale":
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            st.image(processed_image, caption="Grayscale", channels="GRAY", use_container_width=True)
        elif choice == "RGB → HSV":
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            st.image(processed_image, caption="HSV", use_container_width=True)
        elif choice == "RGB → YCbCr":
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            st.image(processed_image, caption="YCbCr", use_container_width=True)
        elif choice == "BGR ↔ RGB":
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(processed_image, caption="BGR ↔ RGB", use_container_width=True)

    # ---- Transformations ----
    with tab3:
        choice = st.selectbox("Choose Transformation", ["Rotate", "Scale", "Translate"])
        if choice == "Rotate":
            angle = st.slider("Rotation Angle", -180, 180, 45)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
            processed_image = cv2.warpAffine(image, M, (w, h))
            st.image(processed_image, caption=f"Rotated {angle}°", use_container_width=True)
        elif choice == "Scale":
            scale = st.slider("Scaling Factor", 0.1, 2.0, 1.0)
            processed_image = cv2.resize(image, None, fx=scale, fy=scale)
            st.image(processed_image, caption=f"Scaled ×{scale}", use_container_width=True)
        elif choice == "Translate":
            tx = st.slider("Shift X", -100, 100, 20)
            ty = st.slider("Shift Y", -100, 100, 20)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            h, w = image.shape[:2]
            processed_image = cv2.warpAffine(image, M, (w, h))
            st.image(processed_image, caption=f"Translated ({tx},{ty})", use_container_width=True)

    # ---- Filtering ----
    with tab4:
        choice = st.selectbox("Choose Filter", ["Gaussian", "Median", "Mean", "Sobel", "Laplacian"])
        if choice == "Gaussian":
            k = st.slider("Kernel Size", 1, 15, 5, step=2)
            processed_image = cv2.GaussianBlur(image, (k, k), 0)
            st.image(processed_image, caption="Gaussian Blur", use_container_width=True)
        elif choice == "Median":
            k = st.slider("Kernel Size", 1, 15, 3, step=2)
            processed_image = cv2.medianBlur(image, k)
            st.image(processed_image, caption="Median Blur", use_container_width=True)
        elif choice == "Mean":
            k = st.slider("Kernel Size", 1, 15, 3, step=2)
            processed_image = cv2.blur(image, (k, k))
            st.image(processed_image, caption="Mean Blur", use_container_width=True)
        elif choice == "Sobel":
            processed_image = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)
            processed_image = cv2.convertScaleAbs(processed_image)  # ✅ Fix
            st.image(processed_image, caption="Sobel Filter", use_container_width=True)
        elif choice == "Laplacian":
            processed_image = cv2.Laplacian(image, cv2.CV_64F)
            processed_image = cv2.convertScaleAbs(processed_image)  # ✅ Fix
            st.image(processed_image, caption="Laplacian Filter", use_container_width=True)

    # ---- Enhancement ----
    with tab5:
        choice = st.selectbox("Choose Enhancement", ["Histogram Equalization", "Sharpening"])
        if choice == "Histogram Equalization":
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            processed_image = cv2.equalizeHist(gray)
            st.image(processed_image, caption="Histogram Equalized", channels="GRAY", use_container_width=True)
        elif choice == "Sharpening":
            kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
            processed_image = cv2.filter2D(image, -1, kernel)
            st.image(processed_image, caption="Sharpened Image", use_container_width=True)

    # ---- Edge Detection ----
    with tab6:
        choice = st.selectbox("Choose Edge Detection", ["Canny", "Sobel", "Laplacian"])
        if choice == "Canny":
            t1 = st.slider("Threshold1", 50, 300, 100)
            t2 = st.slider("Threshold2", 50, 300, 200)
            processed_image = cv2.Canny(image, t1, t2)
            st.image(processed_image, caption="Canny Edges", channels="GRAY", use_container_width=True)
        elif choice == "Sobel":
            processed_image = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
            processed_image = cv2.convertScaleAbs(processed_image)  # ✅ Fix
            st.image(processed_image, caption="Sobel Edge", use_container_width=True)
        elif choice == "Laplacian":
            processed_image = cv2.Laplacian(image, cv2.CV_64F)
            processed_image = cv2.convertScaleAbs(processed_image)  # ✅ Fix
            st.image(processed_image, caption="Laplacian Edge", use_container_width=True)

    # ---- Compression ----
    with tab7:
        fmt = st.selectbox("Save Format", ["JPG", "PNG", "BMP"])
        if st.button("💾 Save Image"):
            fname = f"output.{fmt.lower()}"
            save_image(processed_image, fname)
            st.success(f"Image saved as {fname}")
            with open(fname, "rb") as f:
                st.download_button("⬇️ Download Processed Image", f, file_name=fname)