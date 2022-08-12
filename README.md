# Completing Networks by Learning Local Connection Patterns

This repository will contain the official PyTorch implementation of:
<br>

**Completing Networks by Learning Local Connection Patterns**<br>
Zhang Zhang, Ruyi Tao<sup>\dagger</sup>, Yongzai Tao<sup>\dagger</sup>, Mingze Qi, Jiang Zhang<sup>\*</sup>
<br>
(<sup>\dagger</sup>: Equal Contribution)<br>
(<sup>\*</sup>: Corresponding author) <br>
[Download PDF](https://arxiv.org/pdf/2204.11852)<br>

<img src="./img/fig22.png" width="500px" alt="">

<br>

### Abstract: 

Network completion is a harder problem than link prediction because it does not only try to infer missing links but also nodes. Different methods have been proposed to solve this problem, but few of them employed structural information - the similarity of local connection patterns. In this paper, we propose a model named C-GIN to capture the local structural patterns from the observed part of a network based on the Graph Auto-Encoder framework equipped with Graph Isomorphism Network model and generalize these patterns to complete the whole graph. Experiments and analysis on synthetic and real-world networks from different domains show that competitive performance can be achieved by C-GIN with less information being needed, and higher accuracy compared with baseline prediction models in most cases can be obtained. We further proposed a metric ”Reachable Clustering Coefficient(CC)” based on network structure. And experiments show that our model perform better on a network with higher Reachable CC.

### Requirements

- Python 3.8
- Pytorch 1.1


### Run Experiment
You can replicate the experiment for Boolean Network by simply running the file train_bn.py
```
python main.py
```


### Cite
If you use this code in your own work, please cite our paper:
```

```
