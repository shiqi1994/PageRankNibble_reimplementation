# PageRankNibble_reimplementation

This is the student project in Rome-Moscow school 2020.

## Requirements
Following packages can be installed using pip or conda.
```
networkx 2.5
numpy 1.19.2
pydot 1.3.0
pygraphviz 1.3
matplotlib 3.3.2
scipy 3.3.2
scikit-learn 0.23.2
```

## Usage
```
python main_case1.py
```
Note that

* PageRankNibble algorithem is reimplemented in the function ``PageRankNibble_undirected.py``.

* ``main_case1.py`` and ``main_case2.py`` use synthetic data.

* ``main_case3.py`` use UCI ML Wine recognition datasets. https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data

* To apply different k in k-NN algorithem, please modify the variable ``n_neighbors``, and get accuracy under different k. 

* Also you can choose a start point by modify the variable ``Seed``.



## Results
### Experiment result on synthetic data (1/2)
![13.png](https://i.loli.net/2020/12/15/EP5azRejNcLH13M.png)
![24.png](https://i.loli.net/2020/12/15/4PyKEpQAU9dnOXj.png)

### Experiment result on synthetic data (2/2)
![1133.png](https://i.loli.net/2020/12/15/UNsupdWgeRPylJj.png)
![2244.png](https://i.loli.net/2020/12/15/VFj8UesSMhLkNwl.png)

### Experiment result on UCI ML Wine recognition datasets
![111222.png](https://i.loli.net/2020/12/15/tTH61jv2bDJRXEA.png)
