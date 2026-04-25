# Plant disease detection

Local tools for **plant leaf disease recognition** using a Keras model: a **Streamlit** web app for inference and a **Jupyter notebook** for exploration and training.

## Prerequisites

- **Python 3.10+** (3.11 works well with the bundled `.venv` layout)
- A trained Keras model file for the app (see below)

## Setup

From the project root (`plant-disease-detection`):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Streamlit app

1. **Model file**  
   Place your trained model as:

   `streamlit/trained_model.keras`

2. **Class names** (one of the following)  
   - `streamlit/class_names.json` — JSON array of label strings, **or**  
   - a `train/` folder at the project root with one subfolder per class (names used as labels).

3. **Start the app** (with the venv activated, from the project root):

   ```bash
   streamlit run streamlit/main.py
   ```

   Streamlit will print a local URL (usually `http://localhost:8501`). Open it in your browser to upload a leaf image and get a prediction.

   Optional: specify host/port, e.g. `streamlit run streamlit/main.py --server.port 8502`.

## Jupyter notebook

Training and analysis live in `plant_disease_detection.ipynb` at the project root. With the venv activated:

```bash
pip install jupyter   # if you do not already have it
jupyter lab
```

Then open the notebook from the UI.

## Project layout (high level)

| Path | Role |
|------|------|
| `streamlit/main.py` | Streamlit UI and inference |
| `streamlit/trained_model.keras` | Keras model (you provide) |
| `streamlit/class_names.json` | Optional: class list |
| `train/`, `valid/`, `test/` | Image data folders (if you use the notebook pipeline) |
| `requirements.txt` | Python dependencies |

## Notes

- Inference runs **on your machine**; images are not sent elsewhere unless you change deployment.
- The app is **decision-support only**, not a professional plant diagnosis or legal compliance claim.
