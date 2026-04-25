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

## Deploy on Streamlit Community Cloud

If build logs show **`Using Python 3.14.x`**, `pip` cannot install TensorFlow (`No matching distribution found` / *no matching Python ABI*). You **must** run the app on **Python 3.12** or **3.11** — a `runtime.txt` file in the repo is **not** what Community Cloud uses for the interpreter; the version is chosen in the **Streamlit Cloud UI** only.

1. In [Streamlit Community Cloud](https://share.streamlit.io/), open your app → **Settings** (⚙) (or use **Create app** / **Reboot** and expand **Advanced settings** on the deploy form).
2. Set **Python version** to **3.12** (recommended) or **3.11**. Save.
3. If the UI will not let you change Python, **delete the app** and create it again, selecting **3.12** in **Advanced settings** when you first deploy. ([Docs: upgrade Python on Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python).)
4. Add **`streamlit/trained_model.keras`** to the app by some means (e.g. Git LFS, release download, or hosted URL) — it is in `.gitignore` by default because of size.

## Notes

- Inference runs **on your machine**; images are not sent elsewhere unless you change deployment.
- The app is **decision-support only**, not a professional plant diagnosis or legal compliance claim.
