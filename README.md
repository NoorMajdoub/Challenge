# ✨ GNN Challenge: Liar Nodes✨

##  The Problem

For most neural networks, classification tasks are made individually based on the embedding of each data point. Samples that belong to the same class tend to have similar embeddings and therefore lie close to each other in the embedding space.

**Graph Neural Networks (GNNs)** fundamentally alter this paradigm by performing **collective decision making**. Rather than relying solely on the representation (embedding) of an individual node, the classification decision is influenced by the structure of the graph and by aggregated information from neighboring nodes.

### 💡 The Core Question

**This raises an important question: what should a model do when there is a contradiction between the information provided by a node's embedding and the information coming from its neighborhood?**

---
![Alt text](Challenge Picture!.png)

## 🎮 Challenge Overview

In this challenge, the goal is to implement mechanisms that balance these two sources of information. Node features are manually corrupted, while graph connectivity encodes contextual relationships that may either reinforce or contradict a node's individual features.

Participants must adapt GNN models using techniques such as **neighborhood sampling** and **aggregation** to learn when to trust the node embedding, when to trust the neighborhood, and how to effectively combine both in order to perform node classification on the given dataset.

###  Critical Constraint

**The provided node embeddings are fixed and cannot be modified or recomputed**; participants must rely solely on graph-based aggregation and sampling strategies to resolve conflicting signals.

---

## 💭 Hint

> **This challenge is not about inventing a new GNN, it is about choosing the right way to listen.**
> 
> You can't tell who is giving you the corrupted information and lying to you—you must be smart with your information collecting!

---

##  What Data You Will Use

As most regular GNN training tasks, you will be given:

### 🔗 **A (Adjacency Matrix)**
- **Absolute truth** ✓
- Contains information about the connections and structure of your input graph
- This is your reliable source

### 📈 **X (Node Embeddings)**
- Embeddings of nodes
- ⚠️ **Some nodes have been corrupted to trick you!**
- You cannot modify these embeddings

### 🎭 **Mask (Corruption Indicator)**
- Gives you a hint about which nodes have been corrupted
- Available in `train.csv`
- **NOT available in `test.csv`** (this is what you need to test on)

### 📁 **Data Files**
- **`train.csv`** - Contains features, labels, AND corruption mask
- **`test.csv`** - Contains features only (no labels, no mask)

---

##  Example Techniques

You can use the notebook in baseline as your starting point! It is a simple example solution  that I made!
We want the GNN to  learn when to rely on a node’s own features versus its neighbors’ right ?. An example of a solution is a trainable gate mechanism that dynamically balances these two signals, adapting to potential feature corruption without modifying the fixed embeddings. The model is trained on the provided graph and features, then predicts on test data using the same adaptive trust logic.
You can use the same GNN as your baseline model , same for the training loop , get creative with your aggregation mecanism!

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

---

##  Your Mission ✨✨

Build a GNN that can navigate the noise and make accurate predictions by:

1. **Learning when to trust** individual node features
2. **Learning when to trust** neighborhood aggregation
3. **Balancing both sources** intelligently to maximize classification accuracy

## Evaluation Metrics
Your submissions will be evaluated on the test set using three metrics:

1. **Accuracy** - Overall classification correctness
2. **F1-Score (Macro)** - Balanced performance across all classes
3. **Precision (Macro)** - Quality of positive predictions

## 🏆 Winning Criteria
The winner will be determined by the submission that achieves the best average rank across all three metrics.
###  Submission Process
1. Fork this repository
2. Add your predictions: `submissions/submission_<username>.csv`
3. Create Pull Request with title: `Submission: <username>`
4. Leaderboard updated within 24 hours

Good luck, and may the best aggregation strategy win! 

---
PS: the code used to create the challenge dataset is defined in challenge_data notebook , as a participants please ignore it.

