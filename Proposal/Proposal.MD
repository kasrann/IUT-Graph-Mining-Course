# Graph Mining Course Project Proposal

**Submission Date:** [18/11/2026]  
**Course:** Graph Mining [4051]  
**Instructor:** [Dr. Zeinab Maleki]  

## Student Information
- **Student Name(s):** [Mehdi Eftekhari , Kasra Navayineya]  
- **Student ID(s):** [40415424 , 40414774]  
- **Email(s):** [eftekhari.m@ec.iut.ac.ir , k.navayineya@ec.iut.ac.ir]  


## Project Title
Graph Neural Network-Based Anomaly Detection in Healthcare Time Series: A Comparative Study of GraphSAGE and Multiplex Visibility Graph Analytics

## Abstract
In the domain of healthcare monitoring, traditional time series anomaly detection methods often analyze variables in isolation and fail to capture complex interdependencies among signals. Building upon recent advances in graph-based time series analysis, this project proposes a **Graph Neural Network (GNN)** approach coupled with **Outlier Detection** to identify anomalous patterns in multivariate healthcare time series. By transforming time series into graph structures—using Horizontal Visibility Graphs (HVG) and statistical node features—we model healthcare entities as nodes and their behavioral similarities as edges. We will implement a **GraphSAGE-based Autoencoder** to learn low-dimensional node embeddings by aggregating neighborhood information, followed by unsupervised outlier detection (such as Local Outlier Factor or Isolation Forest). We will compare this deep learning approach against a classical graph-mining baseline: **Neighborhood Jaccard Similarity** (Okano et al., 2026). The expected outcome is a comparative evaluation demonstrating how GNN-based representation learning compares to structural heuristics in detecting healthcare anomalies.

## Problem and Motivation
Healthcare systems generate numerous concurrent time series (such as resource consumption or regional patient records) that interact dynamically, especially during public health crises like the COVID-19 pandemic. Identifying abnormal behavior across these series is vital for timely decision-making. The core research question is: **Can aggregating local neighborhood information via a GraphSAGE model combined with outlier detection outperform classical graph neighborhood heuristics in identifying anomalous time series?**

This problem is significant in graph mining because multivariate time series anomaly detection can be framed as a **node outlier detection task** on graphs. While traditional methods look only at individual signals, Graph Mining techniques allow us to leverage topological properties (such as degree distribution, PageRank centrality, and community structure) alongside statistical node features simultaneously.

We utilize a real-world multivariate healthcare time series dataset. This project is motivated by the future research directions of Okano et al. (2026), aiming to fuse structural visibility graphs with statistical features through a unified GNN and outlier detection pipeline rather than relying on manual heuristic thresholds.

## Objectives
Objective 1: Construct a graph representation from multivariate healthcare time series using Horizontal Visibility Graphs (HVG) for topological structure and extracted statistical descriptors as node features.

Objective 2: Implement a **GraphSAGE** model (Graph Autoencoder) to learn low-dimensional node embeddings and apply unsupervised outlier detection (**LOF / Isolation Forest**) to identify anomalous nodes.

Objective 3: Implement the classical graph-based baseline from Okano et al. (2026) based on $k$-Nearest Neighbor ($k$-NN) reference graphs and **Normalized Jaccard Similarity**.

Objective 4: Evaluate and compare both approaches using reconstruction performance (RMSE) and anomaly detection metrics (F1-Score and Precision@K).

## Related Work
Anomaly detection in time series has traditionally relied on univariate statistical or recurrent models, which often miss system-level anomalies that arise from interactions among multiple variables (Blázquez-García et al., 2021). To capture these relationships, Lacasa et al. (2008, 2015) introduced Visibility Graphs, mapping time series into complex networks to study their structural dynamics.

Recently, Okano et al. (2026) proposed an analytics framework for healthcare time series that constructs multiplex visibility graphs and feature similarity graphs, detecting anomalies by comparing node neighborhoods via the Jaccard index. Concurrently, Graph Neural Networks (GNNs) such as **GraphSAGE** (Hamilton et al., 2017) have shown strong performance in learning node embeddings by sampling and aggregating features from local neighborhoods. This project bridges these two directions by applying GraphSAGE representation learning and outlier detection to time-series-derived healthcare graphs.

## Proposed Methodology
### Dataset(s)
We will use a **Real-World Healthcare Time Series Dataset** (`data/real_healthcare_timeseries_dataset.csv`), containing daily healthcare and epidemiological records across multiple temporal periods.

- **Nodes**: Healthcare series / monitored entities (~187 active nodes).
- **Edges**: Similarity connections ($k$-NN graph) derived from Horizontal Visibility Graph Mutual Information and feature correlations.
- **Preprocessing**: We will filter out inactive series to reduce sparsity, segment the time series into temporal windows, and extract normalized statistical features for each node (selecting informative features via graph community detection and PageRank).

**Implementation Plan (Data Loading)**: We will use `pandas` and `numpy` to process the CSV data, `networkx` to build the graphs, and `PyTorch` / `PyTorch Geometric` (alongside `scikit-learn`) for GNN modeling and outlier detection.

### Techniques and Algorithms
1. **GraphSAGE + Outlier Detection (Main Model):** We will use GraphSAGE layers to aggregate feature information across each node's neighborhood in an unsupervised autoencoder setting.
- *Input*: Graph adjacency structure and extracted statistical node features.
- *Process*: Sample neighbors $\to$ Aggregate neighborhood features via GraphSAGE $\to$ Generate latent node embeddings and reconstruction errors.
- *Detection*: Apply **Local Outlier Factor (LOF)** or **Isolation Forest** on the learned embeddings and reconstruction scores to flag anomalous nodes.

2. **Neighborhood Jaccard Similarity (Baseline):**
We will implement the classical graph framework of Okano et al. (2026). This baseline constructs $k$-NN graphs for different time periods, builds a historical Reference Set of persistent neighbors, and identifies anomalies by computing the $Z$-score normalized Jaccard similarity between current and reference neighborhoods.

### Evaluation Plan
We will evaluate the models across different temporal windows (comparing stable baseline periods with crisis periods).

- **Metric 1 (Model & Reconstruction Quality):** Root Mean Square Error (RMSE) / Reconstruction Loss to evaluate how well the GraphSAGE model captures normal node behavior.
- **Metric 2 (Anomaly Detection & Ranking):** F1-Score, Recall, and Precision@K to measure detector agreement and evaluate how effectively anomalous nodes are ranked at the top of the list.

## Challenges and Resources
**Challenges:**

**Feature Redundancy and Noise:** Extracted time series features can be highly correlated. Mitigated by normalizing features and using graph-based feature selection (community detection and PageRank) before feeding them into the GNN.

**Unsupervised Evaluation:** Healthcare time series often lack exact point-level anomaly labels. Mitigated by comparing baseline periods against known pandemic shock periods and evaluating detector agreement and ranking metrics.

## References
1. Okano, E. Y., Aloise, D., & Nascimento, M. C. V. (2026). An analytics framework for graph-based anomaly detection in healthcare time series. *Healthcare Analytics*, 9, 100447. [link](https://doi.org/10.1016/j.health.2026.100447)

2. Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning on large graphs. *Advances in Neural Information Processing Systems*, 30. [link](https://arxiv.org/pdf/1706.02216)

3. Lacasa, L., Nicosia, V., & Latora, V. (2015). Network structure of multivariate time series. *Scientific Reports*, 5(1), 15508. [link](https://www.nature.com/articles/srep15508)

4. Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: Identifying density-based local outliers. *ACM SIGMOD Record*, 29(2), 93-104. [link](https://dl.acm.org/doi/10.1145/335191.335388)
