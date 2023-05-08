# Optimizing NOTEARS objectives via topological swaps

This is an implementation of the following paper:

[notears]: https://arxiv.org/abs

## Summary

In this paper, we propose a bi-level algorithm that solve a class of non-convex optimization problems that has emerged in the context of learning directed acyclic graphs (DAGs). These problems (e.g. [NOTEARS][notears]) involve minimizing a given loss or score function, subject to a non-convex continuous constraint that penalizes the presence of cycles in a graph. Compared to the previous work, our algorithm: (1) Is guaranteed to find a local minimum or a KKT point under weaker conditions; (2) Achieves better score; (3) leads to siginificant improvement in structure recovery (e.g. SHD).

## Requirements

- Python 3.6+
- `numpy`
- `scipy`
- `python-igraph`
- `torch`: Only used for nonlinear 
- `scikit-learn`
- `scipy`

## Contents 

- `Topo_linear.py` - implementation of TOPO for linear models (also support Logistic losses).
- `Topo_utils` - implementation of support function for TOPO.
- `utils.py` - graph simulation, data simulation, and accuracy evaluation.

## Running TOPO

Use `requirements.txt` to install the dependencies (recommended to use virtualenv or conda).
The simplest way to try out DAGMA is to run a simple example:
```bash
$ git clone https://github.com/Duntrain/TOPO.git
$ cd TOPO/
$ pip3 install -r requirements.txt
$ python3 Topo_linear.py
```

The above runs the TOPO on a randomly generated 20-node Erdos-Renyi graph with 1000 samples. 
The output should look like the below:
```
{'fdr': 0.0, 'tpr': 1.0, 'fpr': 0.0, 'shd': 0, 'nnz': 20}
```


## Acknowledgments

We thank the authors of the [NOTEARS repo][notears-repo] for making their code available. Part of our code is based on their implementation, specially the `utils.py` file and some code from their implementation of nonlinear models.

[notears-repo]: https://github.com/xunzheng/

