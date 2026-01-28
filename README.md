# SMS Spam Detector

A machine learning application that classifies SMS messages as **spam** or **ham** (legitimate) using K-Nearest Neighbours (KNN) with sentence embeddings and cosine distance.

## Features

- **Single Message Classification**: Enter an SMS message and get instant spam/ham prediction with confidence score
- **Batch Processing**: Upload a CSV file to classify multiple messages at once
- **Explainable Results**: View the K nearest training messages that influenced the prediction
- **Token Overlap Analysis**: See common words between your message and similar training examples
- **Model Performance Metrics**: View accuracy, precision, recall, and F1 score on the test set

## How It Works

1. Each SMS message is converted to a dense vector (embedding) using the `all-MiniLM-L6-v2` sentence transformer model
2. For classification, the K most similar messages from the training data are found using cosine distance
3. Prediction is based on distance-weighted voting of these neighbours
4. Confidence is calculated from the weighted vote share

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:

   **Windows:**
   ```bash
   .venv\Scripts\activate
   ```

   **macOS/Linux:**
   ```bash
   source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup

The application requires the SMS Spam Collection dataset. You have two options:

### Option 1: Download from UCI Repository

1. Download from: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
2. Convert to CSV format with columns `label` and `text`
3. Save as `data/sms_spam.csv`

### Option 2: Download from Kaggle

1. Download from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
2. Rename/format the file to have columns `label` and `text`
3. Save as `data/sms_spam.csv`

### Expected Format

The CSV file should have this structure:

```csv
label,text
ham,Go until jurong point, crazy.. Available only in bugis n great world la e buffet...
spam,Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005...
ham,Ok lar... Joking wif u oni...
spam,WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!
```

- `label`: Either "spam" or "ham"
- `text`: The SMS message content

## Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Project Structure

```
spam-detect/
├── app.py                  # Streamlit application entry point
├── src/
│   ├── __init__.py
│   ├── data.py            # Dataset loading and validation
│   ├── embedder.py        # Embedding model wrapper
│   ├── model.py           # KNN training, prediction, evaluation
│   └── explain.py         # Token overlap explanation utilities
├── data/
│   └── sms_spam.csv       # Dataset (you need to add this)
├── artefacts/             # Cached model artefacts (auto-generated)
├── requirements.txt       # Python dependencies
└── README.md
```

## Application Pages

### 1. Single Message

- Enter any SMS text in the text area
- Adjust the K slider (1-25) to control how many neighbours influence the prediction
- Click "Classify Message" to get:
  - Prediction (SPAM or HAM)
  - Confidence percentage
  - Neighbour vote breakdown
  - Table of nearest neighbours with similarity scores and common words

### 2. Batch Check

- Upload a CSV file with a `text` column
- Set the K value for classification
- Click "Classify All Messages" to process
- View results in a table
- Download results as CSV with predictions and confidence scores

### 3. Model Info

- Dataset statistics (size, class balance, message lengths)
- Model performance metrics on test set
- Embedding model details
- Algorithm explanation
- Known limitations

## Caching and Performance

The application uses several caching strategies:

- **Embedding model**: Cached using `st.cache_resource` (loaded once per session)
- **Training embeddings**: Computed once and saved to `artefacts/`
- **Model artefacts**: Saved to disk and reloaded on subsequent runs

On first run, the model training may take 1-2 minutes (depending on your hardware). Subsequent runs will load cached artefacts almost instantly.

If you change the dataset, the application will automatically detect the change (via file hash) and retrain.

## Technical Details

- **Embedding Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Classifier**: KNeighborsClassifier with cosine metric and distance weighting
- **Train/Test Split**: 80/20 stratified split
- **Text Preprocessing**: Minimal (strip whitespace, normalise spaces only)

## Limitations

- Primarily designed for English messages
- May not catch new spam patterns not in training data
- Each message is analysed independently (no sender context)
- Some legitimate messages with spam-like language may be misclassified

## Troubleshooting

**"Dataset file not found" error:**
- Ensure `data/sms_spam.csv` exists with correct columns

**Slow first run:**
- Initial embedding computation takes time; subsequent runs use cached artefacts

**Memory issues:**
- The sentence transformer model requires ~500MB of memory
- Batch processing large files may require more memory

**CUDA/GPU errors:**
- The app works on CPU by default
- If you have PyTorch GPU issues, try: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

## License

This project is for educational purposes. The SMS Spam Collection dataset has its own license from UCI.
