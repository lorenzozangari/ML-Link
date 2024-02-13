# Link Prediction on Multilayer Networks through Learning of Within-Layer and Across-Layer Node-Pair  Structural Features and Node Embedding Similarity


## Overview

Implementation of the *ML-Link* framework presented in the following research paper:

Lorenzo Zangari, Domenico Mandaglio, Andrea Tagarelli (2024). *Link Prediction on Multilayer Networks through Learning of Within-Layer and Across-Layer Node-Pair Structural Features and Node Embedding Similarity*. ACM Web Conference 2024.

>Link prediction has traditionally been studied in the context of simple graphs, although real-world networks are inherently complex as they are often comprised of  multiple interconnected components, or layers. Predicting links in such network systems, or \textit{multilayer networks}, require to consider both the internal structure of a target layer as well as the structure of the other layers in a network, in addition to layer-specific node-attributes when available. This problem poses several challenges, even for graph neural network based approaches despite their successful and wide application to a variety of graph learning problems. In this work, we aim to fill a lack of  multilayer graph representation learning  methods designed for link prediction. Our proposal is a novel neural-network-based learning framework for link prediction on (attributed) multilayer networks, whose key idea is to combine (i) pairwise similarities of multilayer node embeddings learned by a graph neural network model, and (ii) structural features learned from both within-layer and across-layer link information based on overlapping multilayer neighborhoods. Extensive experimental results have shown  that our framework consistently outperforms both single-layer and multilayer methods for link prediction on popular real-world multilayer networks, 
> with an average percentage increase in AUC up to 38%.

Please cite the above paper in any research publication you may produce using this code or data/analysis derived from it.

## Input data
The original real-world networks we used in our paper are contained within the *data/nets* folder, while the
preprocessed data can be found in the *data/prep_nets* folder.

### Data loading
To use your own data, first create a sub-folder named *\<dataset\>* inside the *data/nets/* folder, where *\<dataset\>* is the name of the input network.

The *\<dataset\>* folder must include the files *meta_info.txt* and *net.edges*, which are mandatory, and *features.pt* which is optional. These files are described as follows:

1. The file *meta_info.txt* contains information about the input network, such as:
    * *N*, the number of entities.
    * *L*, the number of layers.
    * *E*, whether the multilayer graph is directed ('DIRECTED') or undirected ('UNDIRECTED').
   
   The file must be formatted with two rows: the first one lists the aforementioned column names (i.e., *N, L, E*), while the second row contains the corresponding column values. For example, given an undirected multiplex graph with 20 entities, 3 layers, 
   the *meta_info.txt* is defined as follows:

    ```
    N  L     E
    20 3  UNDIRECTED
    ```

2. The file *net.edges* contains edge information. 
The required format is *\<*layer src-node dst-node*\>*, where *src-node* and *dst-node* are the source and the destination nodes, respectively, while *layer* denotes the layer to which the edge belongs. Note that node identifiers must be numeric, progressive and starting from 0, while layers identifiers must start with 1 and be progressive. Furthermore, it is required that the *net.edges* file does not contain duplicate edges. In the case of an undirected graph, the *net.edges* file must not specify edges in both directions.  To see an example of a *net.edges* file, refer to one of the networks within the *data/nets/* folder.


3. The file *features.pt* (optional) contains node feature information. It must be a PyTorch tensor with shape (*N***L*, *F*), where *N* is the number of entities, *L* is the number of layers, and *F* is the size of the input features.  Each of the *L* blocks of size *(N,F)* corresponds to the feature matrix of a layer.



The files described in points 1 and 2 are mandatory. If the node attribute matrix is not provided, the default is to use the identity matrix for each layer. The folder structure should appear as follows:

```
ML-Link
│   README.md    
└───data
    │
    └───nets
    │   │   
    │   └───dataset
    │         meta_info.txt
    │         net.edges
    │         features.pt
    │          ...
...
```
### Data preprocessing
To preprocess the data and create the train/test/validation split, run  the *preprocess.py* script as follows:
```python
python preprocess.py --dataset 'ckm' --fold 10 --src './data/nets/' --dst './data/prep_nets'
```

which splits the network (--dataset) *ckm*, located in the source dir (--src) *'../data/nets/'*, in 10 folds (--fold), and save the preprocessed data in the destination directory (--dst) *'../data/prep_nets'*. 

The set of input arguments of the *preprocess.py* script are listed as follows:
```
 --dataset DATASET  
  Input network.

 --dst DST          
  Destination folder where the preprocessed network is saved.

 --src SRC          
  Folder where the input network is stored.

 --seed SEED        
  Random seed.
  
 --fold FOLD         
  Number of folds for K-fold cross validation.
```

The default values of these parameters are shown in the *preprocess.py* script.

Assuming the preprocessed network is named *\<dataset\>*, the directory structure following the execution of the preprocess.py script with the previously outlined parameter configuration will look as follows:
```
ML-Link
│   README.md    
└───data
    │
    └───nets
    │   │   
    │   └───dataset
    │       ...
    └───prep_nets
    │   │   
    │   └───dataset
    │       ...
...
```

### Synthetic networks generation
To generate the synthetic networks used in the efficiency analysis reported in our paper, first move inside the *./input_data* directory,  then run the *ws_generator.py* script. You can use the following arguments:
```
 --layers LAYERS  
   Number of layers.

 --nodes NODES    
   Number of nodes per layer.

 --beta BETA      
   Rewiring probability. 

 --seed SEED      
   Random seed.
```

The default values of these parameters are shown in the *ws_generator.py* script.

To generate a network with 3 layers, 500 entities and rewiring probability 0.1, run the 
*ws_generator.py* script with the following arguments:

```python
python ws_generator.py --nodes 500 --layers 3  --beta 0.1 --seed 72
```

This will create a folder in *./data/nets/* named *rn_500_3_0.1*. This folder will contain the *meta_info.txt* and the *net.edges* files (as described above), and a file for each layer containing the edges of that layer, that is, *l1.edges* for the first layer.

After the network was generated and saved into the *./data/nets/rn_500_3_0.1* folder, you must preprocess it as described in the **Data preprocessing** section, so that it can be used for training the model.


## Usage
If you are using your own data, you must preprocess it before training the model (**Data preoprocessing** section).

To train the model, run the *train.py* script with the following commands:

```python
python train.py --dataset 'ckm' --runs 10 --epochs 100 --omn 'oan;maan' 
                --prep_dir './data/prep_nets/' --gpu 0 --save_dir './artifacts'
```

This will train the model on the *ckm* network (specified by *--dataset*). The preprocessed data of this network (indicated by *--prep_dir*) is located in './data/prep_nets/'. 
The training will be conducted on the GPU device 0 (*--gpu*), and it will run for 100 epochs (*--epochs*) 
for each of the 10 fold (*--fold*), employing OAN and MAAN (*--omn*) as overlapping multilayer neighborhoods. 
Note that if you do not want to use overlapping multilayer neighborhoods, you can provide either the empty string or the string 'none' to the *--omn* argument.

The AP and AUC values yielded by the model are saved into the *./artifacts/results* folder (*--save_dir*). See the **Input arguments** section for additional details.


## Input arguments
The input arguments for the ML-Link framework are listed as follows:
```
 --dataset DATASET     
  Name of the input network, whose name identifies a particular subfolder of 'data/nets'.

 --gpu GPU             
  Which GPU to use (-1 for CPU).

 --seed SEED
  Random seed.

 --runs RUNS 
  Number of runs (should match the number of folds).

 --edge_dim EDGE_DIM   
   Hidden dimension of the edge MLP.

 --node_dim NODE_DIM   
   Hidden dimension of node MLP.

 --phi_dim PHI_DIM     
   Hidden dimension of context MLP.

 --hidden_dim HIDDEN_DIM
   Hidden dimension of the GNN layers.

 --num_hidden NUM_HIDDEN
   Numbers of hidden layers of the GNN.

 --epochs EPOCHS       
   Number of epochs.

 --n_heads N_HEADS     
    Number of attention heads of GNN.

 --heads_mode HEADS_MODE
   Concatenate ('concat') or averaging ('avg') the multiple  attention heads.

 --predictor PREDICTOR
   MLP decoder to use ("mlp").

 --omn OMN
   Types of overlapping multilayer neighborhoods. 
   Supported types are oan ('oan') and maan ('maan'). 
   Each overlapping multilayer neighborhood must be followed by a semicolon, e.g., 'oan;maan'

 --dropout DROPOUT     
   Dropout rate.

 --attn_dropout ATTN_DROPOUT
   Attention dropout rate for attention based GNN models.

 --lr LR 
   Learning rate

 --weight-decay WEIGHT_DECAY
    Weight decay (L2 loss).

 --psi PSI             
   Impact of overlapping multilayer neighborhoods.

 --no_gnn              
   Whether to use only the NN-NPN component.

 --no_struct           
   Whether to use only the GNN-NE component.

 --root ROOT           
   Root directory of input data.

--save_dir SAVE_DIR   
   Folder where the performance scores are saved.

--prep_dir PREP_DIR   
   Folder storing the preprocessed data.

 --ck_dir CK_DIR       
   Folder where the checkpoint model is saved.
```

Modify the file *params.py* to change the default values of the parameters.
## Requirements and environment

The required libraries are listed in the file *requirements.txt*.

CUDA version: 11.7

GPU: GeForce RTX 3090

CPU: Intel(R) Xeon(R) GOLD 628R CPU @ 2.70GHz

## Hyper-parameters selection

Below are the best hyper-parameters found on each dataset used in the evaluation reported in the submitted paper:

- Cs-Aarhus:
  - hidden_dim = 128; attn_dropout = 0.3;
- CKM:
  - hidden_dim = 64; attn_dropout = 0.3;
- Elegans:
  - hidden_dim = 128, attn_dropout_0.7; n_heads = 4; heads_mode = 'concat'
- Lazega:
  - hidden_dim = 64; attn_dropout = 0.7;
- DkPol:
  - hidden_dim = 32; attn_dropout = 0.3;
- ArXiv:
  - hidden_dim = 256; attn_dropout = 0.3;
- Synthetic networks:
  - hidden_dim = 256; attn_dropout=0.7; epochs=10/20 for the class of synthetic networks generated with rewiring probability 0.1/0.5.

Other hyper-parameters are shared by all datasets. You can find their default value inside the *params.py* file.

