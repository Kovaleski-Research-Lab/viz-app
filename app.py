import streamlit as st
import os

st.set_page_config(page_title="Model Flipbooks", layout="wide")
st.title("Model Flipbook Comparisons")

# Get the absolute path to the directory containing this file (app.py).
BASE_DIR = os.path.dirname(__file__)

# Go two levels up (from viz-app/ to code/ to develop/) then into 'results'
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../results"))

def find_flipbook_pairs(root_dir):
    """
    Recursively walk through `root_dir` and find matching pairs of GIFs:
    One containing 'groundtruth' and another containing 'prediction'.
    Return a list of tuples: (groundtruth_path, prediction_path).
    """
    pairs = []
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for GIFs in {root_dir} ...")

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if "groundtruth" in filename.lower():
                groundtruth_file = os.path.join(dirpath, filename)
                # Heuristic to find the corresponding prediction file
                prediction_file = groundtruth_file.replace("groundtruth", "prediction")
                if os.path.exists(prediction_file):
                    pairs.append((groundtruth_file, prediction_file))

    return pairs

pairs = find_flipbook_pairs(RESULTS_DIR)

if not pairs:
    st.warning(f"No flipbook pairs found under {RESULTS_DIR}, womp womp.")
else:
    for i, (gt, pred) in enumerate(pairs):
        st.subheader(f"Flipbook Pair #{i+1}")
        col1, col2 = st.columns(2)

        with col1:
            st.caption(f"Ground Truth:\n{gt}")
            st.image(gt, use_container_width=True)

        with col2:
            st.caption(f"Prediction:\n{pred}")
            st.image(pred, use_container_width=True)

        st.divider()  # Horizontal line for clarity