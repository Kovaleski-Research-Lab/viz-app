import streamlit as st
import os
from pathlib import Path
import base64
import yaml
import re

st.set_page_config(page_title="Model Flipbooks", layout="wide")
#st.title("Model Flipbook Comparisons")

# Get the absolute path to the directory containing this file (app.py).
BASE_DIR = os.path.dirname(__file__)

# Two levels up from viz-app/ -> code/ -> develop/, then into 'results'
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../results"))

################################################################################
# Step 1: Data structure to store each flipbook pair + relevant metadata
################################################################################

def parse_metadata(pair_tuple, root_dir):
    """
    Given (groundtruth_path, prediction_path), parse the subdirectory structure
    to extract metadata:
      - model_type (e.g. lstm, convlstm, mlp, autoencoder, modelstm, ...)
      - sub_model_type (e.g. svd, random) if model_type == "modelstm"
      - time_series_type (one_to_many, many_to_many) if relevant
      - architecture (sequential, distributed)
      - model_id (extracted from folder name like 'model_15-v1')
    Returns a dict with these keys plus the actual file paths.
    """
    groundtruth, prediction = pair_tuple

    # Convert to pathlib for easier path slicing
    g_path = Path(groundtruth)
        
    # relative_parts: how the path looks *relative* to root_dir
    rel_parts = g_path.relative_to(root_dir).parts  # e.g. ("meep_meep", "lstm", "one_to_many", "sequential", "model_15_v1", "flipbooks", "sample_0_phase_groundtruth.gif")

    # Find if there's an eval_# subdirectory
    eval_dir = None
    for part in rel_parts:
        # This regex matches "eval_", followed by one or more digits
        if re.match(r"eval_\d+", part):
            eval_dir = part
            break
    

    # rel_parts[0] should be "meep_meep"
    # rel_parts[1] is typically model_type (unless it's "modelstm").
    # But if it's "modelstm", then rel_parts[2] might be "svd"/"random" before the time_series_type.

    model_type = None
    sub_model_type = None
    time_series_type = None
    architecture = None
    model_id = None
    train_static = None
    valid_static = None
    train_metrics = None
    valid_metrics = None
    loss_plot = None
    mse_evolution = None

    model_base_dir = "/".join(rel_parts[:-2])
    
    if "eval" in model_base_dir.split("/")[-1]:
        # everything except the last part
        loss_plot = os.path.join(root_dir, "/".join(model_base_dir.split("/")[:-1]), "loss_plots", "loss.pdf")
    else:
        loss_plot = os.path.join(root_dir, model_base_dir, "loss_plots", "loss.pdf")
    mse_evolution = os.path.join(root_dir, model_base_dir, "performance_metrics","mse_evolution.pdf")
    #st.write(loss_plot)
    
    dft_plots_dir = os.path.join(root_dir, model_base_dir, "dft_plots")
    for filename in os.listdir(dft_plots_dir):
        if "Training" in filename:
                train_static = os.path.join(dft_plots_dir, filename)
        elif "Validation" in filename:
            valid_static = os.path.join(dft_plots_dir, filename)
            
    train_metrics = os.path.join(root_dir, model_base_dir, "performance_metrics", "train_metrics.txt")
    valid_metrics = os.path.join(root_dir, model_base_dir, "performance_metrics", "valid_metrics.txt")
                
    # A quick helper to read safely from rel_parts by index
    def get_part(index):
        return rel_parts[index] if index < len(rel_parts) else None

    # By default:
    # index 0 = "meep_meep" (assuming your root is /develop/results/meep_meep)
    # index 1 = <model_type> or "modelstm"
    # index 2 = possibly "svd"/"random" if modelstm, or "one_to_many"/"many_to_many" otherwise
    # index 3 = time_series_type (if modelstm + subdir) or architecture
    # index 4 = architecture or model_??? 
    # index 5 = model_??? or 'flipbooks'
    # index 6 = 'flipbooks' or file

    # Let's parse
    model_type_candidate = get_part(1)  # e.g. "lstm", "convlstm", ...
    if model_type_candidate == "modelstm":
        # Then next part might be "svd"/"random"
        model_type = "modelstm"
        possible_sub_model_type = get_part(2)
        if possible_sub_model_type in ("svd", "random"):
            sub_model_type = possible_sub_model_type
            # Then time_series_type would be index 3
            time_series_type = get_part(3)  # one_to_many/many_to_many
            architecture = get_part(4)      # sequential/distributed
            model_id_dir = get_part(5)      # e.g. model_15_v1
        else:
            # If it's modelstm but no 'svd' or 'random' there,
            # maybe the structure differs. We can adapt if needed or skip.
            # For now, store them as None or handle your actual structure.
            time_series_type = get_part(2)
            architecture = get_part(3)
            model_id_dir = get_part(4)
    else:
        # Normal path
        model_type = model_type_candidate
        time_series_type = get_part(2)  # one_to_many / many_to_many
        architecture = get_part(3)      # sequential / distributed
        model_id_dir = get_part(4)      # e.g. model_15_v1

    # "model_15_v1" or "model_15-v1" etc.
    if model_id_dir and model_id_dir.startswith("model_"):
        model_id = model_id_dir.replace("model_", "")
    else:
        model_id = model_id_dir  # fallback if doesn't match

    return {
        "model_type": model_type,
        "sub_model_type": sub_model_type,
        "time_series_type": time_series_type,
        "architecture": architecture,
        "model_id": model_id,
        "eval_dir": eval_dir,
        "groundtruth": groundtruth,
        "prediction": prediction,
        "loss_plot": loss_plot,
        "train_static": train_static,
        "valid_static": valid_static,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "mse_evolution": mse_evolution,
    }


def find_flipbook_pairs(root_dir):
    """
    Walk through `root_dir` and find pairs of GIFs:
    One containing 'groundtruth' and a matching 'prediction'.
    Return a list of dictionaries with parsed metadata + file paths.
    """
    pairs_data = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if "groundtruth" in filename.lower():
                groundtruth_file = os.path.join(dirpath, filename)
                # Heuristic to find the corresponding prediction file
                prediction_file = groundtruth_file.replace("groundtruth", "prediction")

                if os.path.exists(prediction_file):
                    # Parse subdir info
                    meta = parse_metadata((groundtruth_file, prediction_file), root_dir)
                    pairs_data.append(meta)

    return pairs_data

################################################################################
# Step 2: Gather all pairs (with metadata) and build UI for filtering
################################################################################

all_pairs = find_flipbook_pairs(RESULTS_DIR)

if not all_pairs:
    st.warning(f"No flipbook pairs found under {RESULTS_DIR}.")
    st.stop()

# Collect possible filter options:
model_types = sorted(set(p["model_type"] for p in all_pairs if p["model_type"]))
sub_model_types = sorted(set(p["sub_model_type"] for p in all_pairs if p["sub_model_type"]))
time_series_types = sorted(set(p["time_series_type"] for p in all_pairs if p["time_series_type"]))
architectures = sorted(set(p["architecture"] for p in all_pairs if p["architecture"]))
model_ids = sorted(set(p["model_id"] for p in all_pairs if p["model_id"]))

# Sidebar filters
st.sidebar.title("Menu")
# add checkbox for showing/hiding performance metrics
show_params = st.sidebar.checkbox("Show Model Parameters", value=False)
show_metrics = st.sidebar.checkbox("Show Performance Metrics", value=False)
selected_plot = st.sidebar.selectbox(
    "Select content type to display for all models:",
    ["Flipbooks", "Loss Plot", "MSE Evolution", "Training Static Plot", "Validation Static Plot"],
    index=0
)
with st.sidebar.expander("Filters", expanded=True):
    selected_model_type = st.selectbox("Model Type", options=["All"] + model_types)
    selected_sub_model_type = None
    if selected_model_type == "modelstm" and len(sub_model_types) > 0:
        selected_sub_model_type = st.selectbox("Mode-LSTM Sub-Type", options=["All"] + sub_model_types)

    selected_time_series_type = st.selectbox("Time Series Type", options=["All"] + time_series_types)
    selected_architecture = st.selectbox("Architecture", options=["All"] + architectures)
    selected_model_id = st.selectbox("Specific Model ID", options=["All"] + model_ids)

################################################################################
# Step 3: Filter the data based on user selections
################################################################################
def pass_filter(item):
    # item is a dict with {model_type, sub_model_type, time_series_type, architecture, model_id, ...}
    if selected_model_type != "All" and item["model_type"] != selected_model_type:
        return False
    
    if selected_model_type == "modelstm" and selected_sub_model_type and selected_sub_model_type != "All":
        # only filter on sub_model_type if model_type == "modelstm"
        if item["sub_model_type"] != selected_sub_model_type:
            return False
    
    if selected_time_series_type != "All" and item["time_series_type"] != selected_time_series_type:
        return False
    
    if selected_architecture != "All" and item["architecture"] != selected_architecture:
        return False

    if selected_model_id != "All" and item["model_id"] != selected_model_id:
        return False
    
    return True

filtered_pairs = [p for p in all_pairs if pass_filter(p)]

if not filtered_pairs:
    st.warning("No results match your selected filters.")
    st.stop()
    
# Construct dynamic title based on filters
title_parts = ["Results"]
if selected_model_type != "All":
    title_parts.append(selected_model_type.upper())  # e.g., "LSTM", "MODELSTM"
    if selected_model_type == "modelstm" and selected_sub_model_type and selected_sub_model_type != "All":
        title_parts.append(selected_sub_model_type.capitalize())  # e.g., "SVD" or "Random"
if selected_time_series_type != "All":
    title_parts.append(selected_time_series_type.replace("_", "-").capitalize())  # e.g., "One-to-Many"
if selected_architecture != "All":
    title_parts.append(selected_architecture.capitalize())  # e.g., "Sequential" or "Distributed"

# Set dynamic page title
st.title(" - ".join(title_parts))

################################################################################
# Step 4: Group pairs by (model_id) so we can display them in “cards.”
################################################################################

from collections import defaultdict

from collections import defaultdict
models_dict = defaultdict(lambda: defaultdict(list))

for pdict in filtered_pairs:
    model_id = pdict["model_id"] or "unknown"
    eval_dir = pdict["eval_dir"] or "no_eval_subdir"
    models_dict[model_id][eval_dir].append(pdict)

################################################################################
# Step 5: Display results in a card-like layout
################################################################################

def read_params_file(params_path):
    """
    Reads and parses the params.yaml file.
    """
    if not os.path.exists(params_path):
        return None

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params

def read_text_file(text_path):
    """
    Reads and parses the text file.
    """
    if not os.path.exists(text_path):
        return None
    with open(text_path, "r") as f:
        text = f.read()
    return text

def display_model_params(params):
    """
    Displays a card with general and architecture-specific parameters, organized into columns.
    Dynamically shows additional sections for certain models.
    """
    if not params:
        st.warning("No params.yaml found for this model.")
        return

    # General model information
    general_info = {
        "Batch Size": params.get("trainer", {}).get("batch_size"),
        "Num Epochs": params.get("trainer", {}).get("num_epochs"),
        "Optimizer": params.get("model", {}).get("optimizer"),
        "Objective Function": params.get("model", {}).get("objective_function"),
        "LR Scheduler": params.get("model", {}).get("lr_scheduler"),
        "Learning Rate": params.get("model", {}).get("learning_rate"),
        "Sequence Length": params.get("model", {}).get("seq_len"),
        "Wavelength(s)": params.get("data", {}).get("wv_eval"),
    }

    # Display all information in a single card
    with st.expander("Model Parameters", expanded=show_params):
        st.write("### General Hyperparameters")
        display_info_in_columns(general_info)

        # Architecture-specific information
        model_arch = params.get("model", {}).get("arch")
        if model_arch == "convlstm":
            st.write("### ConvLSTM Specifics")
            convlstm_info = {
                "In Channels": params["model"]["convlstm"].get("in_channels"),
                "Kernel Size": params["model"]["convlstm"].get("kernel_size"),
                "Out Channels": params["model"]["convlstm"].get("out_channels"),
            }
            display_info_in_columns(convlstm_info)

        elif model_arch == "lstm":
            st.write("### LSTM Specifics")
            lstm_info = {
                "Hidden Dimensions": params["model"]["lstm"].get("h_dims"),
                "Input Dimensions": params["model"]["lstm"].get("i_dims"),
                "Num Layers": params["model"]["lstm"].get("num_layers"),
            }
            display_info_in_columns(lstm_info)

            # Additional LSTM-specific parameters
            extra_lstm_info = {
                "Dropout": params["model"]["lstm"].get("dropout"),
                "Bidirectional": params["model"]["lstm"].get("bidirectional"),
                "Activation Function": params["model"]["lstm"].get("activation_fn"),
            }
            if any(extra_lstm_info.values()):
                st.write("### Additional LSTM Parameters")
                display_info_in_columns(extra_lstm_info)

        elif model_arch == "mlp_real":
            st.write("### MLP (Real) Specifics")
            mlp_info = {
                "Activation": params["model"]["mlp_real"].get("activation"),
                "Layers": params["model"]["mlp_real"].get("layers"),
            }
            display_info_in_columns(mlp_info)

        elif model_arch == "mlp_imag":
            st.write("### MLP (Imaginary) Specifics")
            mlp_info = {
                "Activation": params["model"]["mlp_imag"].get("activation"),
                "Layers": params["model"]["mlp_imag"].get("layers"),
            }
            display_info_in_columns(mlp_info)

        elif model_arch == "autoencoder":
            st.write("### Autoencoder Specifics")
            autoencoder_info = {
                "Encoder Channels": params["model"]["autoencoder"].get("encoder_channels"),
                "Decoder Channels": params["model"]["autoencoder"].get("decoder_channels"),
                "Latent Dim": params["model"]["autoencoder"].get("latent_dim"),
            }
            display_info_in_columns(autoencoder_info)
            
        elif model_arch == "modelstm":
            method = params["model"]["modelstm"].get("method")
            if method == "svd":
                method_str = "Singular Value Decomposition"
            elif method == "random":
                method_str = "Fixed Random Encoding"
            st.write(f"### Modes - {method_str}")
            modelstm_info = {}
            display_info_in_columns(modelstm_info)

def parse_metrics(metrics_text):
    """
    Parses a metrics string into a dictionary.
    """
    metrics = {}
    # Regular expression to match "key: value" pairs
    matches = re.findall(r"(\w+): ([\d\.]+(?: \+\-/ [\d\.]+)?)", metrics_text)
    for key, value in matches:
        if key != 'SSIM' and key != 'MAE':
            metrics[key] = value
    return metrics

def display_performance_metrics(train_path, valid_path):
    """
    Displays the performance metrics for the model in a structured format.
    """
    train_metrics_text = read_text_file(train_path)
    valid_metrics_text = read_text_file(valid_path)

    with st.expander("Performance Metrics", expanded=show_metrics):
        st.write("### Resubstitution")
        train_metrics = parse_metrics(train_metrics_text)
        display_info_in_columns(train_metrics, num_columns=5)

        st.write("### Validation")
        valid_metrics = parse_metrics(valid_metrics_text)
        display_info_in_columns(valid_metrics, num_columns=5)


def display_info_in_columns(info_dict, num_columns=4):
    """
    Helper function to display key-value pairs in columns.
    """
    columns = st.columns(num_columns)
    items = list(info_dict.items())

    for idx, (key, value) in enumerate(items):
        col = columns[idx % num_columns]
        with col:
            st.write(f"**{key}:** {value if value is not None else 'N/A'}")
            
def display_pdf(file_path, caption=""):
    """
    Displays a PDF file in Streamlit using an embed tag.
    """
    if not os.path.exists(file_path):
        st.warning(f"{caption} not found.")
        return
    with st.container():
        with open(file_path, "rb") as f:
            st.write(f"### {caption}")
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf">'
            st.markdown(pdf_display, unsafe_allow_html=True)

# Check if the user has made a valid selection in the dropdown and filters
if selected_plot == "Flipbooks" and selected_model_type == "All" and selected_time_series_type == "All" and selected_architecture == "All" and selected_model_id == "All":
    st.warning("Please make a selection from the filtering menu to display results.")
else:
    # Iterate through models and display their cards
    for model_id, eval_dict in models_dict.items():
        st.markdown(f"### Model ID: {model_id}")
        if selected_plot == "Loss Plot":
            # Just show the loss plot once for this model, ignoring eval folders
            # Grab any subfolder’s items, or specifically the "no_eval_subdir" if present
            if "no_eval_subdir" in eval_dict:
                # Use the items from `no_eval_subdir`
                loss_plot_items = eval_dict["no_eval_subdir"]
            else:
                # Fallback: pick the first eval folder that exists
                any_eval_subdir = next(iter(eval_dict.keys()))
                loss_plot_items = eval_dict[any_eval_subdir]

            if not loss_plot_items:
                st.warning("No loss plot found.")
            else:
                loss_plot_path = loss_plot_items[0].get("loss_plot")
                display_pdf(loss_plot_path, caption="Loss Plot")

        else:
            for eval_subdir, items in eval_dict.items():
                if eval_subdir != "no_eval_subdir":
                    st.markdown(f"**Evaluation Folder:** {eval_subdir}")

                # Load and display params.yaml
                #st.write(f"* {items}")
                params_path = os.path.join(Path(items[0]["groundtruth"]).parent.parent, "params.yaml")
                params = read_params_file(params_path)
                if not params:
                    params_path = os.path.join(Path(items[0]["groundtruth"]).parent.parent.parent, "config.yaml")
                    params = read_params_file(params_path)
                display_model_params(params)

                # Display performance metrics
                train_path = items[0].get("train_metrics")
                valid_path = items[0].get("valid_metrics")
                display_performance_metrics(train_path, valid_path)

                if selected_plot == "Flipbooks":
                    channel = ['Magnitude', 'Phase']
                    cols = st.columns(4)  # Create 4 columns in a single row

                    # Display flipbook pairs
                    for i, meta in enumerate(items, start=1):
                        groundtruth = meta["groundtruth"]
                        prediction = meta["prediction"]

                        c = channel[i % 2]
                        if c == 'Phase':
                            idx = [0, 1]
                        else:
                            idx = [2, 3]

                        with cols[idx[0]]:
                            st.caption(f"Ground Truth {c}")
                            st.image(groundtruth, use_container_width=True)

                        with cols[idx[1]]:
                            st.caption(f"Predicted {c}")
                            st.image(prediction, use_container_width=True)

                elif selected_plot == "MSE Evolution":
                    mse_evolution_path = items[0].get("mse_evolution")
                    display_pdf(mse_evolution_path, caption="MSE Evolution")

                elif selected_plot == "Training Static Plot":
                    train_static_path = items[0].get("train_static")
                    display_pdf(train_static_path, caption="Training Static Plot")

                elif selected_plot == "Validation Static Plot":
                    valid_static_path = items[0].get("valid_static")
                    display_pdf(valid_static_path, caption="Validation Static Plot")

                st.divider()  # Divider for clarity