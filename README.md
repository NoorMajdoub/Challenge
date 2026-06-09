# ✨ GNN Challenge: Liar Node✨

##  The Problem


For most neural networks, classification tasks are made individually based on the embedding of each data point. Samples that belong to the same class tend to have similar embeddings and therefore lie close to each other in the embedding space.

**Graph Neural Networks (GNNs)** fundamentally alter this paradigm by performing **collective decision making**. Rather than relying solely on the representation (embedding) of an individual node, the classification decision is influenced by the structure of the graph and by aggregated information from neighboring nodes.



### 💡 The Core Question

**This raises an important question: what should a model do when there is a contradiction between the information provided by a node's embedding and the information coming from its neighborhood? aka adversary learning**

---
![Alt text](picture.png)

## 🎮 Challenge Overview

In this challenge, the goal is to implement mechanisms that balance these two sources of information that might contradict each other. Node features are manually corrupted, while graph connectivity encodes contextual relationships that may either reinforce or contradict a node's individual features.

We apply this concept to Cancer biology! A cell's environment can give misleading cues , a malignant cell might “look normal” in isolation, or neighboring immune cells might influence its signaling,the node’s own features (gene expression) might conflict with neighborhood information (similar cells in the microenvironment), mimicking real biological “contradictions.”

Participants must adapt GNN models using techniques such as **neighborhood sampling** and **aggregation** to learn when to trust the node embedding, when to trust the neighborhood, and how to effectively combine both in order to perform node classification and detect the cancerous cells.



###  Critical Constraint

**The provided node embeddings are fixed and cannot be modified or recomputed**; participants must rely solely on graph-based aggregation and sampling strategies to resolve conflicting signals.

---

## 💭 Hint

> **This challenge is not about inventing a new GNN, it is about choosing the right way to listen.**
> 
> You can’t tell which cells are sending corrupted signals , you must be clever about how you gather and trust information from the tumor microenvironment!

---

##  What Data You Will Use

The Cancer Single-cell Expression Map (CancerSCEM) dataset provides comprehensive single-cell RNA-seq data across multiple cancer types. Instead of raw sequencing reads, the dataset offers UMI count matrices in standard .tsv format for downstream analysis. Each dataset includes:
Gene Expression Matrices per cell  enabling gene descriptio/Cell-Type Annotations and Components /Functional Molecules /Cell-Cell Interaction Data .
You can read more about this data source on https://ngdc.cncb.ac.cn/cancerscem/index .
As for this task, we have altered samples from this data to present it for a  GNN training tasks, you will be given:

### 🔗 **A (Adjacency Matrix)**
- Encodes the connections and structure of your input graph.
- Most connections reflect real interactions between cells, but some edges may have been altered, introducing noise.
  
### 📈 **X (Node Embeddings)**
- Embeddings of nodes == Genes encodings
- ⚠️ **Some nodes have been corrupted to trick you!**
- You cannot modify these embeddings

### 🎭 **Mask (Corruption Indicator)**
- Gives you a hint about which nodes have been corrupted
- Available in `train.csv`
- **NOT available in `test.csv`** (this is what you need to test on)
- The corruption mask may be used only during training to modulate aggregation, weighting, or attention — not to filter nodes or labels.

### 📁 **Data Files**
- **`train.npz`** - Contains features AND corruption mask.
- **`test.csv`** - Contains features only (no labels, no mask)
- **`labels.csv`** - Contains labels for the full training data
- **`edges.csv`** - Describes the communication between the cells
Please review the baseline code to help you better navigate the data.
---
## 🚫 Constraints

###  You CANNOT
- Remove the noisy nodes based on the mask from the training data
- Alter the embedding of the nodes
- Use external data

###  You CAN:
- Use the mask to guide your aggregation strategy
- Implement creative neighborhood sampling
- Design custom aggregation functions
- Use attention mechanisms
- Combine multiple GNN layers strategically
- Look up additional resources to understand the behavior of the cancerous cells.
---

##  Your Mission ✨✨

Build a GNN that can navigate the noise and make accurate predictions by:

1. **Learning when to trust** individual node features
2. **Learning when to trust** neighborhood aggregation
3. **Build** robust GNNs can detect anomalies or corrupted signals — similar to spotting a misbehaving cell in a tumor.

## Resources

Convex Adversarial Collective Classification ;MohamadAli Torkamani, Daniel Lowd Proceedings of the 30th International Conference on Machine Learning, PMLR 28(1):642-650, 2013.

Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions Lukas Gosch, Simon Geisler, Daniel Sturm, Bertrand Charpentier, Daniel Zügner, Stephan Günnemann

The Reason Why Cancer is so Hard to Beat -Kurzgesagt – In a Nutshell

A Multimodal Graph Neural Network Framework of Cancer Molecular Subtype Classification

Cancer Single-cell Expression Map

Graph Neural Networks in Cancer and Oncology Research: Emerging and Future Trends Grigoriy Gogoshin 1,*, Andrei S Rodin 1,

---

