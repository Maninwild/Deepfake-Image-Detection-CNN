# Deepfake Image Detection Using a Custom CNN

This is a complete research prototype for classifying face images as **real** or
**fake**. It contains the full dataset the trained Keras model evaluation outputs
command-line tools and a Streamlit browser interface.

## Quick start on Windows

1. Install 64-bit Python 3.12 from python.org and enable **Add Python to PATH**.
2. Extract this ZIP to a normal folder.
3. Double-click `START_WINDOWS.bat`.
4. Wait while the isolated environment and packages are installed.
5. The script validates the project and opens the prediction interface in a browser.

The first run takes longer because TensorFlow must be downloaded and the dataset
must be converted into NumPy arrays. Later runs reuse the generated arrays.

## Manual setup

The GitHub version stores all 3,912 original images in `dataset_parts/` ZIP files.
`START_WINDOWS.bat` extracts them automatically. For manual setup first run
`py -3.12 code/extract_dataset.py` from the project folder.

Run these commands from the project folder:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python code\validate_dataset.py
python code\split_data.py
python code\preprocess_arrays.py
python code\evaluate.py
python -m streamlit run app.py
```

The terminal prints a local URL such as `http://localhost:8501`. Open it in your
browser if it does not open automatically.

## Predict one image from the terminal

```powershell
python code\predict.py "C:\path\to\image.jpg"
```

The output reports the predicted label confidence fake probability and real
probability. Label mapping is `real = 0` and `fake = 1`.

## Retrain the model

The supplied model has already completed 25 epochs. To intentionally discard its
training state and train a new model from epoch 1 run:

```powershell
python code\train_step.py 25 --restart
python code\evaluate.py
```

Training uses class weights Adam binary cross-entropy augmentation
`ReduceLROnPlateau` model checkpointing and early stopping. Intermediate training
is resumable through `outputs/checkpoint.keras`.

## Dataset

- 3,912 JPEG images
- 2,623 fake images
- 1,289 real images
- Every supplied image is 256 x 256 RGB
- No corrupt files or exact duplicate files were detected
- Deterministic stratified-style class split using seed 42

| Split | Real | Fake | Total |
|---|---:|---:|---:|
| Train | 902 | 1,836 | 2,738 |
| Validation | 193 | 393 | 586 |
| Test | 194 | 394 | 588 |

The training arrays resize images to 96 x 96 RGB. Pixel rescaling to `[0, 1]`
is performed inside the model. Training-only augmentation uses horizontal flip
small rotation zoom and contrast changes.

## Model

The custom CNN contains four convolution blocks with 32 64 128 and 256 filters.
Each block uses Conv2D BatchNormalization ReLU and MaxPooling. The classifier uses
GlobalAveragePooling Dense(128) Dropout(0.4) and a sigmoid output. The model has
422,881 parameters.

## Reproduced test results

The packaged model was loaded and evaluated again against all 588 generated test
arrays using TensorFlow 2.21.0 and Keras 3.15.1.

| Metric | Value |
|---|---:|
| Accuracy | 82.31% |
| Precision for fake | 94.48% |
| Recall for fake | 78.17% |
| F1-score for fake | 85.56% |
| ROC-AUC | 91.87% |

The reproduced confusion matrix is `[[176, 18], [86, 308]]` where rows are actual
classes and columns are predicted classes in the order real then fake.

## Project structure

```text
deepfake_cnn_project/
|-- START_WINDOWS.bat
|-- app.py
|-- requirements.txt
|-- README.md
|-- raw/
|   |-- real/
|   `-- fake/
|-- code/
|   |-- validate_dataset.py
|   |-- split_data.py
|   |-- preprocess_arrays.py
|   |-- train_step.py
|   |-- evaluate.py
|   `-- predict.py
`-- outputs/
    |-- best_model.keras
    |-- test_metrics.json
    |-- classification_report.txt
    |-- training_curves.png
    |-- confusion_matrix.png
    `-- roc_curve.png
```

The `split/` and `arrays/` directories are generated locally and are intentionally
not stored in the ZIP because they can be reproduced from `raw/`.

## Important limitation

The fake class mainly contains clean single-face generated portraits while the
real class contains more casual photographs including some group images and varied
backgrounds. The CNN may therefore learn composition or background differences in
addition to manipulation artifacts. This prototype should not be treated as a
forensic proof system. Face detection alignment and cross-dataset evaluation are
recommended future improvements.
