# Neuro-Inspired Machine Learning: HDC, SOM, Echo State Networks and Word Embeddings

From-scratch implementations of brain-inspired and classic learning methods: **hyperdimensional
computing** for language identification, **self-organizing maps**, a **backpropagation network
written in NumPy**, **echo state networks** (reservoir computing), plus k-NN image classification
and distributional word embeddings. Course: *Applied Artificial Intelligence* (D7041E), Luleå
University of Technology, 2025.

**Stack:** Python · NumPy · scikit-learn · gensim · Matplotlib · Jupyter

## Projects

| Folder | Topic | Highlights |
|---|---|---|
| [`lab1-knn-and-word-embeddings/`](lab1-knn-and-word-embeddings) | k-NN and word embeddings | k-NN on our own photo dataset (best k = 4 by cross-validation, **86.8 %** CV accuracy). Random Indexing vs. word2vec on the TOEFL synonym test |
| [`lab2-hdc-som-backprop/`](lab2-hdc-som-backprop) | HDC, SOM, backprop | Language identification with high-dimensional vectors (n-gram encoding, Europarl / Leipzig corpora); SOM trained on grids from 20×20 to 80×80 with learning-rate and neighbourhood-decay sweeps; multilayer perceptron with backpropagation implemented from scratch (best test error **1.9 %**) |
| [`lab3-echo-state-networks/`](lab3-echo-state-networks) | Reservoir computing | Echo state network for time-series prediction and classification; effect of spectral radius and input scaling (report as PDF) |

## Word-embedding comparison (TOEFL synonym test)

| Method | Best dimension | Accuracy |
|---|---|---|
| **Random Indexing** | 4000 | **70.3 %** |
| word2vec (gensim) | 100 | 67.4 % |

Random Indexing builds word vectors incrementally from sparse random index vectors, with no training
loop, yet it beat word2vec on this small benchmark.

## Run it

```bash
pip install numpy scipy scikit-learn gensim matplotlib tqdm jupyter
jupyter notebook
```

Large datasets (the TASA corpus, Europarl) are not included. The notebooks show where to download
them. The photo dataset for k-NN is included, resized to 512 px.

## Team

Labs 2 and 3 by **Abdalrahman Nasser** and **Dominic Addo**; full reports with code and outputs are
included as PDFs.
