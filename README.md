# EG (Expanded Graph)
This repository contains the code of paper:  
 >****** Not published yet   
 
This repository is forked from https://github.com/deepopo/CENALP

##Branches
The branches are organized as follows:
- Master and EG: GPU implementation of the Expanded Graph method that apply the default SVD (not the truncated version)
- EG-CPU: CPU implementation of EG 
- EG-Truncated: GPU implementation of EG with the truncated version of SVD
- EG-Truncated-CPU: CPU implementation of EG-Truncated
- EG-Mini_Truncated: EG-Mini according to the paper
- EG-Mini-CPU: CPU implementation of EG-Mini_Truncated
- CENALP: https://github.com/deepopo/CENALP


Before executing , you should install the following packages:  
``pip install sklearn``  
``pip install networkx``  
``pip install gensim``  
``pip install tqdm``  
``pip install tensorly``  
``pip install fbpca``  
The detailed version are ``python==3.7.2`` and ``networkx==2.4``, ``sklearn==0.22.1``, ``gensim==3.4.0``, ``tqdm==4.31.1``, ``tensorly==0.6.0``, ``fbpca==1.0``, but they are not mandatory unless the code doesn't work.  
## Basic usage  
### Data  
See folders /graph/ and /alignment/ for the full used datasets  

### Example  
In order to run *EG-FST*, you can execute *demo.py* directly or execute the following command in ./src/:  
``python demo.py --filename bigtoy --embedding_method_class MatrixFactorization --embedding_method_kind sum_power_tran --alpha 5 --layer 3 --align_train_prop 0.5``  
In order to run *EG-Lab*, you can execute *demo.py* directly or execute the following command in ./src/:  
``python demo.py --filename bigtoy --embedding_method_class MatrixFactorization --embedding_method_kind laplacian --alpha 5 --layer 3 --align_train_prop 0.5``  
You can check out the other options:  
``python demo.py --help``  

### Evaluate
We use precision and recall to evaluate both link prediction and network alignment in this repository.

## Reference  
If you are interested in this research, please cite this paper:  
*****
