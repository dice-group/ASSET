## ASSET: A Semi-supervised Approach for Entity Typing in Knowledge Graphs
This repository contains the implementation and dataset of our paper *ASSET: A Sem-supervised Approach for Entity Typing in Knowledge Graphs*. 

## Summary
- In this work, we propose a novel approach for KG entity typing in knowledge graphs that leverages semi-supervised learning from massive unlabeled data.
- Our approach follows a teacher-student learning paradigm that allows combining a small amount of labeled data with a large amount of unlabeled data to boost the performance of downstream tasks: the teacher model annotates the unlabeled data with pseudo labels; then the student is trained on the pseudo-labeled data and a small amount of high-quality labeled data. 
- We conduct several experiments on two benchmarking datasets (FB15k-ET and YAGO43k-ET), our results demonstrate that our approach outperforms state-of-the-art baselines in entity-typing task. 

## Requirements
```
python 3.6
scikit-learn 0.24
tensorflow 2.0
scikit-multilearn 0.2.0 
pickle5 0.0.11
pykeen 1.5.0 
```
## Installation
You can install all requirements via ```pip install -r requirements.txt```

## Datasets

* In the `data` folder, you can download the benchmarking datasets `FB15k-ET` and `YAGO43K-ET`. 

* For ConnectE embedding, we use the source code from its [Github repository](https://github.com/Adam1679/ConnectE). For further details, we refer users to follow the installation instructions on the packages' websites. 

* If you want to experiment with another knowledge graph embedding model. We recommend [Pykeen](https://pykeen.github.io/) or [Graphvite](https://graphvite.io/) for generating embeddings for the datasets. 

* Furthermore, we provide the preprocessed files used in our experiments in the ```data/preprocessed files``` folder that can be used directly to evaluate the baselines and our approach.

## How to run:

- We provide the source code in Python in the folder ```src```. Users can download the code and use it in their favorite IDE, configure it with different models and datasets.

- As examples, we provide two jupyter noteboos with a description for FB15k-ET and YAGP43k-ET in the ```notebook``` folder. First, users should install the required libraries,  then locate the data files (e.g., the file of pre-trained embedding models and groud-truth labels.)


## Hyper-parameters
The following are our optimal values for the hyper-parameters used in the experiments: 

```
epochs = 100            # Maximum number of epochs
patience = 3            # After how many iterations to stop the training
batch_size = 128        # How many sequences in each batch during training
lr = 0.001             # Learning rate of Adam optimizer
Dropout= 0.20          # Dropout rate in the Deep Neural Model 
```

## Contact
If you have any feedback or suggestions, feel free to send me an email to hamada.zahera@upb.de 

## Cite
TBD
