
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
from pathlib import Path
from io import BytesIO
import pandas as pd
import zipfile

# ---------------- CONFIG ----------------

st.set_page_config(
    page_title="Newspaper Ad Detector",
    page_icon="📰",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "best.pt"

# ---------------- MODEL ----------------

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))

# ---------------- APP HEADER ----------------

st.title("📰 Newspaper Advertisement Detector")
st.write(
    "Automatically detect, extract and download "
    "advertisements from newspaper pages."
)

st.divider()

# ---------------- IMAGE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload a newspaper page",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Detection confidence",
    min_value=0.05,
    max_value=0.90,
    value=0.25,
    step=0.05
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    width, height = image.size

    st.subheader("Original Newspaper")
    st.image(image, use_container_width=True)

    if st.button("Detect Advertisements", type="primary"):

        with st.spinner("Detecting advertisements..."):

            model = load_model()

            results = model.predict(
                source=image,
                imgsz=1280,
                conf=confidence,
                iou=0.5,
                verbose=False
            )

            result = results[0]
            boxes = result.boxes

            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)

            crops = []

            for i, box in enumerate(boxes):

                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                score = float(box.conf[0])

                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(width, int(x2)), min(height, int(y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                # Draw bounding box
                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline="red",
                    width=max(3, width // 1000)
                )

                draw.text(
                    (x1, max(0, y1 - 20)),
                    f"Ad {i+1} ({score:.2f})",
                    fill="red"
                )

                # Crop from original resolution
                pad_x = int((x2 - x1) * 0.01)
                pad_y = int((y2 - y1) * 0.01)

                crop = image.crop((
                    max(0, x1 - pad_x),
                    max(0, y1 - pad_y),
                    min(width, x2 + pad_x),
                    min(height, y2 + pad_y)
                ))

                crops.append({
                    "name": f"advertisement_{len(crops)+1:03d}.jpg",
                    "image": crop,
                    "confidence": score
                })

            st.session_state["annotated"] = annotated
            st.session_state["crops"] = crops
            st.session_state["original_size"] = image.size

# ---------------- RESULTS ----------------

if "crops" in st.session_state:

    crops = st.session_state["crops"]
    annotated = st.session_state["annotated"]

    st.divider()
    st.subheader("Detection Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Advertisements detected", len(crops))

    with col2:
        st.metric(
            "Original image resolution",
            f"{st.session_state['original_size'][0]} × "
            f"{st.session_state['original_size'][1]}"
        )

    st.image(
        annotated,
        caption="Detected advertisements",
        use_container_width=True
    )

    st.subheader("Extracted Advertisements")

    if not crops:
        st.warning("No advertisements detected. Try lowering the confidence threshold.")
    else:
        cols = st.columns(2)

        for i, item in enumerate(crops):
            with cols[i % 2]:
                st.image(item["image"], use_container_width=True)
                st.write(f"**Advertisement {i+1}**")
                st.caption(f"Confidence: {item['confidence']:.2%}")

                buffer = BytesIO()
                item["image"].save(buffer, format="JPEG", quality=95)

                st.download_button(
                    "Download crop",
                    data=buffer.getvalue(),
                    file_name=item["name"],
                    mime="image/jpeg",
                    key=f"download_{i}"
                )

        # ZIP download
        zip_buffer = BytesIO()

        with zipfile.ZipFile(
            zip_buffer, "w", zipfile.ZIP_DEFLATED
        ) as zip_file:

            for item in crops:
                buffer = BytesIO()
                item["image"].save(buffer, format="JPEG", quality=95)
                zip_file.writestr(item["name"], buffer.getvalue())

        st.download_button(
            "Download all advertisements (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="newspaper_advertisements.zip",
            mime="application/zip",
            type="primary"
        )
