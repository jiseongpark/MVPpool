# Multi-View Node Pruning for Accurate Graph Representation

<img src = "https://github.com/MVPpool/Multiview-Graph-Pooling-with-Graph-Pruning/blob/main/figs/concept_figure.png" width="50%">

This repository is the official implementation of [Multi-View Node Pruning for Accurate Graph Representation](https://openreview.net/forum?id=HhUm1cnsTb). 


## Requirements

The following package specification is required for executing the implementation.

```setup
TensorFlow==2.3.1
Spektral==0.6.2
Numpy==1.18.5
Networkx==2.4
Scipy==1.4.1
Scikit-learn==0.22.2.post1
```

## Training & Evaluation

To train the models in the paper, run the given script:

```script
sh mvp_graph_classification.sh
```

This script will give a supervised learning experimental result for PROTEIN dataset.

You can also specify some hyperparameter or select dataset with the argument specification.

```eval
python mvp_main.py --viewpoints {# viewpoints} --hidden_dim {# hidden dimension} ...
```


## Results

Our model achieves the following performance on some [TU dataset](https://chrsmrrs.github.io/datasets/docs/datasets/) and [OGB dataset](https://ogb.stanford.edu/docs/graphprop/)

|                    |    PROTEINS     |        DD      |     COLLAB      |     molHIV     |
| ------------------ |---------------- | -------------- |---------------- | -------------- |
|      DiffPool      |     73.12%         |      75.34%       |     81.57%         |      71.62%       |
|   MVP + DiffPool   |     79.91%         |      76.16%       |     83.86%         |      76.34%       |
|      MincutPool    |     76.87%         |      78.56%       |     81.38%         |      72.19%       |
| MVP + MincutPool   |     81.34%         |      82.88%       |     83.92%         |      76.39%       |
|        GMT         |     77.59%         |      78.72%       |     80.08%         |      61.43%       |
|      MVP + GMT     |     79.46%         |      77.46%       |     81.04%         |      66.89%       |

As the table shows, our method significantly improves the existing pooling method with various datasets. 


## Contributing

> Licence part 
