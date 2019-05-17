# Graph2GO
## Description
This is a graph-based representation learning method for prediction protein functions. Protein-protein interactions (PPIs), amino acid sequence, subcellular location and protein domains information are used to train the model.
The auto-encoder part of our model is improved based on the implementation by T. N. Kifp. You can find the source code here [Graph Auto-Encoders](https://github.com/tkipf/gae). Thanks for T. N. Kifp.

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

### Steps
#### Step1: decompress data files
> cd data
> unzip data.zip

#### Step2: run the model
> cd ../src/Graph2GO
> python main.py
