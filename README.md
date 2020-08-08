# Graph2GO
## Description
This is a graph-based representation learning method for predicting protein functions. We use both network information and node attributes to improve the performance. Protein-protein interaction (PPIs) networks and sequence similarity networks are used to construct graphs, which are used to propagate node attribtues, according to the definition of graph convolutional networks.

We use amino acid sequence (CT encoding), subcellular location (bag-of-words encoding) and protein domains (bag-of-words encoding) as the node attributes (initial feature representation).

The auto-encoder part of our model is improved based on the implementation by T. N. Kifp. You can find the source code [here](https://github.com/tkipf/gae). 

![VGAE model](https://github.com/yanzhanglab/Graph2GO/blob/master/model_up.PNG)
![model architecture](https://github.com/yanzhanglab/Graph2GO/blob/master/model_down.PNG)

## citing
If you found Graph2GO is useful for your research, please consider citing our [work](https://academic.oup.com/gigascience/article/9/8/giaa081/5885490):
```
@article{10.1093/gigascience/giaa081,
    author = {Fan, Kunjie and Guan, Yuanfang and Zhang, Yan},
    title = "{Graph2GO: a multi-modal attributed network embedding method for inferring protein functions}",
    journal = {GigaScience},
    volume = {9},
    number = {8},
    year = {2020},
    month = {08},
    issn = {2047-217X},
    doi = {10.1093/gigascience/giaa081},
    url = {https://doi.org/10.1093/gigascience/giaa081}
}
```

## Usage
### Requirements
- Python 3.6
- TensorFlow
- Keras
- networkx
- scipy
- numpy
- pickle
- scikit-learn
- pandas

### Data
You can download the data of all six species from here <a href="https://www.dropbox.com/s/ilrudy0j7wb7b8s/data.zip?dl=0" target="_blank">data</a>. Please Download the datasets and put the data folder in the same path as thee src folder.

### Steps
#### Step1: decompress data files
> unzip data.zip

#### Step2: run the model
> cd src/Graph2GO     
> python main.py    
> **Note there are several parameters can be tuned: --ppi_attributes, --simi_attributes, --species, --thr_ppi, --thr_evalue, etc. Please refer to the main.py file for detailed description of all parameters**
