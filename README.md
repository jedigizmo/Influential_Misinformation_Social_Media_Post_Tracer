# Influence Maximization for Misinformation Spread Using BERT and Independent Cascade Modeling

## 📌 Project Overview
This project investigates how misinformation spreads across a network of textual social media posts. Using BERT-based semantic similarity, we construct an influence graph where nodes are posts and edges represent potential influence pathways. Influence propagation is modeled using the **Independent Cascade (IC) Model**, and the goal is to identify the **top-k most influential misinformation posts**.

Only **fake news posts** are allowed to activate others, reflecting a constraint that harmful content is the main driver of misinformation cascades.

This project integrates techniques from:
- Natural Language Processing (NLP)
- Social Network Analysis
- Probabilistic Graph Models
- Influence Maximization
- Algorithmic optimization (Hill Climbing)

---

## 📂 Dataset
We use the **Fake and True News Dataset** (Fake.csv and True.csv), each containing:
- `title`
- `text`
- `subject`
- `date` (format: `December 31, 2017`)

For computational efficiency, **only 1,000 samples** from each dataset are used.
https://www.kaggle.com/datasets/islamic/fake-news-classification

---

## 🧠 Methodology

### 1. **Text Embedding using BERT**
We compute dense semantic embeddings using:
- Sentence-BERT (`all-MiniLM-L6-v2`)

Each post's `title + text` is encoded into a high-dimensional vector.

---

### 2. **Graph Construction**
Nodes represent individual news posts.

**Edge creation rule:**
- An edge from node *u → v* exists if:
  1. Semantic similarity(u, v) > threshold  
  2. Date(u) < Date(v)  
  3. Weight = `(similarity / 0.5) − 1`  
  4. No edge is added if dates are equal  
- Only **fake nodes** can activate others in the IC model

The resulting graph is a **directed, weighted influence network**.

---

### 3. **Independent Cascade Model (IC)**
The IC model simulates the spread of information:

- Active nodes get **one chance** to activate their neighbors.
- Activation probability = edge weight.
- Simulations are repeated (Monte Carlo) to estimate:
  - Expected cascade size
  - Activation likelihood
  - Spread depth and branching behavior

---

### 4. **Influence Maximization (Hill Climbing)**
The objective is:

\[
\max_{S : |S| = k} f(S)
\]

Where:
- \( S \) = seed set of misinformation nodes  
- \( k \) = number of seeds (default: 10)  
- \( f(S) \) = expected cascade size  

We use the **Hill Climbing** greedy algorithm, which is guaranteed to find a solution at least:

\[
(1 - 1/e) \approx 63\% \text{ of optimal}
\]

Outputs include:
- Ranked list of the 10 most influential fake posts
- Influence spreads for each seed
- Text content of the #1 most influential post

---

## 🔍 Visualizations
The notebook generates:
- Influence network visualization with node coloring:
  - 🔴 Seed nodes
  - 🟠 Activated nodes
  - 🔵 Inactive nodes
- Cascade size bar charts
- Subgraph visualization for the most influential seed

---

## 📦 Dependencies
pandas
numpy
torch
sentence-transformers
networkx
matplotlib
tqdm

Author
Gikonyo Njendu
United States Military Academy
MA461 – Graph Theory



