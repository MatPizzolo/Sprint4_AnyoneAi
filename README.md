# Sprint Project 04 - Best Buy Product Classification

Build a multimodal machine learning system to classify products using image and text embeddings from pre-trained deep learning models.

## Tech Stack

- **Python 3.9+** - Main programming language
- **TensorFlow** - Deep learning framework for image embeddings
- **Transformers (Hugging Face)** - Pre-trained models for text and image embeddings
- **Scikit-learn** - Classic ML models and preprocessing
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive experimentation
- **Pytest** - Testing framework
- **Black** - Code formatting
- **Docker** - Containerization (optional)

## Setup & Run

1. Install dependencies (use virtual environment recommended):
```bash
pip install -r requirements.txt
```

For Mac users with GPU support:
```bash
pip install -r requirements_mac.txt
```

For GPU-enabled systems:
```bash
pip install -r requirements_gpu.txt
```

2. Download the dataset and images:
   - Place `processed_products_with_images.csv` in `data/`
   - Download images from [this link](https://drive.google.com/file/d/14s2aDNTEWse86cWyLhvVIKmob6EbQrm_/view?usp=sharing)
   - Extract images to `data/images/`

3. Open and run the Jupyter notebook:
```bash
jupyter notebook "AnyoneAI - Sprint Project 04.ipynb"
```

4. Complete the TODO sections in the code (see `assignment.md` for details).

5. Run tests:
```bash
pytest tests/
```

Or without warnings:
```bash
pytest tests/ --disable-warnings
```

6. Format code:
```bash
black --line-length=88 .
```

### Docker Setup (Optional)

1. Build the container:
```bash
docker build -t anyoneai-project .
```

2. Run the container:
```bash
docker run -p 8888:8888 -v $(pwd):/app anyoneai-project
```

3. Access Jupyter at `http://127.0.0.1:8888/tree?token=your_token`

## Project Structure

```
├── data/                    # Dataset and images
│   ├── processed_products_with_images.csv
│   └── images/              # Product images (224x224)
├── src/                     # Source code
│   ├── vision_embeddings_tf.py    # Image embedding extraction
│   ├── nlp_models.py              # Text embedding extraction
│   ├── classifiers_classic_ml.py  # Classic ML models
│   ├── classifiers_mlp.py         # MLP models
│   └── utils.py                   # Utility functions
├── tests/                   # Unit tests
├── results/                 # Model evaluation results
├── Embeddings/              # Generated embeddings (not in repo)
├── AnyoneAI - Sprint Project 04.ipynb
├── assignment.md            # Detailed instructions
├── README.md
└── requirements.txt
```

## Key Concepts Covered

- **Multimodal Learning** - Combining image and text data for classification
- **Transfer Learning** - Using pre-trained models (ResNet50, ConvNextV2, MiniLM)
- **Embedding Extraction** - Generating numerical representations from images and text
- **Feature Engineering** - Merging and preprocessing multimodal embeddings
- **Classic ML Models** - Random Forest, Logistic Regression, SVM
- **Deep Learning** - Multi-layer Perceptron (MLP) with early fusion
- **Model Evaluation** - Classification metrics, confusion matrices
- **Dimensionality Reduction** - PCA visualization of embeddings

## Business Problem

Classify BestBuy products into predefined categories using both product images and text descriptions:
- **Input**: Product images (224x224 JPG) and text descriptions
- **Output**: Product category prediction
- **Approach**: Extract embeddings from pre-trained models, train ML classifiers

**Required Models**:
- **Image**: ResNet50 (Keras) + ConvNextV2 (Hugging Face)
- **Text**: MiniLM (Hugging Face) + optional BERT/OpenAI
- **Classifiers**: Random Forest, Logistic Regression, MLP

**Performance Targets**:
- Multimodal model: ≥85% accuracy, ≥80% F1-score
- Text-only model: ≥85% accuracy, ≥80% F1-score
- Image-only model: ≥75% accuracy, ≥70% F1-score

See `assignment.md` for complete project instructions and implementation details.
