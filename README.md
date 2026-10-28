Week 3 Assessment: Deep Learning + NLP
Objective

Build and compare neural-network and NLP models for customer feedback classification.

Task 1: Neural Network Fundamentals
Implemented ReLU, sigmoid, softmax, MSE and binary cross-entropy.
Built a 2-4-1 NumPy neural network for XOR.
Implemented backpropagation and gradient descent.
Verified gradients using PyTorch autograd.
Compared different learning rates and plotted loss.
Task 2: PyTorch Training Pipeline
Dataset: Scikit-learn breast cancer dataset.
Preprocessing: StandardScaler fitted on training data only.
Model: MLP with Linear, BatchNorm, ReLU and Dropout.
Optimizer: Adam.
Loss: BCEWithLogitsLoss.
Added early stopping and saved the model state_dict.
Compared results with logistic regression.
Task 3: Classical NLP
Created 360 synthetic labelled feedback texts.
Implemented text cleaning and TF-IDF features.
Trained logistic regression and reported accuracy, macro F1 and the confusion matrix.
Inspected important words and pretrained word embeddings.
Compared TF-IDF with averaged GloVe sentence vectors.
Task 4: Transformers and API
Inspected DistilBERT tokenizer outputs.
Demonstrated sentiment analysis and NER pipelines.
Fine-tuned DistilBERT for three-class classification.
Compared TF-IDF and DistilBERT using the same test split.
Created a FastAPI POST /predict endpoint.
Limitations

The NLP dataset is synthetic and small. High test scores on this dataset do not establish real-world performance.

Transformer training and embedding quality depend on the available hardware, downloaded models and random seed.

Results

Record actual accuracy, macro F1, training time and inference time from reports/comparison.csv after running the experiments.

How to Run
bash
python task1_neural_networks.py
python task2_pytorch_pipeline.py
python task3_classical_nlp.py
python task4_transformers.py
uvicorn app:app --reload