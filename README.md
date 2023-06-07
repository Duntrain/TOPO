# Optimizing NOTEARS Objectives via Topological Swaps <img src="https://github.com/Duntrain/TOPO/assets/6846921/fa8812bc-b295-4200-9ce0-4c267aafc9ce" width="5%" height="5%">

This is an implementation of the following paper:

[1] C. Deng, K. Bello, B. Aragam, P. Ravikumar. (2023). [Optimizing NOTEARS Objectives via Topological Swaps][topo]. [ICML'23](https://icml.cc/Conferences/2023) 


[notears]: https://arxiv.org/abs/1803.01422
[dagma]: https://arxiv.org/abs/2209.08037
[topo]: https://arxiv.org/abs/2305.17277

If you find this code useful, please consider citing:

```
@inproceedings{deng2023optimizing,
  title={Optimizing NOTEARS Objectives via Topological Swaps},
  author={Chang, Deng and Kevin, Bello and Bryon, Aragam and Pradeep, Ravikumar},
  booktitle={Proceedings of the 40th International Conference on Machine Learning},
  year={2023}
}
```

## Summary

In this paper, we propose a bi-level algorithm that solve a class of non-convex optimization problems that has emerged in the context of learning directed acyclic graphs (DAGs). These problems (e.g. [NOTEARS][notears]) involve minimizing a given loss or score function, subject to a non-convex continuous constraint that penalizes the presence of cycles in a graph. Compared to the previous work, our algorithm: (1) Is guaranteed to find a local minimum or a KKT point under weaker conditions; (2) Achieves better score; (3) leads to siginificant improvement in structure recovery (e.g. SHD).

## Requirements

- Python 3.6+
- `numpy`
- `scipy`
- `python-igraph`
- `torch`: Only used for nonlinear 
- `scikit-learn`


## Contents 

- `Topo_linear.py` - implementation of TOPO for linear models (also support Logistic losses).
- `Topo_nonlinear.py` - implementation of TOPO for nonlinear models (__The code for nonlinear case is coming soon__)
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
## Extension

Users can take advantage of the framework of TOPO to test different structure equation model (SEM) by simply writing their own regression and score function (i.e. `regress(X,y)`, `score(X,W)`) in `Topo_linear.py`. We have implemented those functions for Linear and Logistic model.

## Acknowledgments

We thank the authors of the [NOTEARS repo][notears-repo] for making their code available. Part of our code is based on their implementation, specially the `utils.py` file and some code from their implementation of nonlinear models.

[notears-repo]: https://github.com/xunzheng/

