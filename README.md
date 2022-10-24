# Local Differential Privacy for Federated Learning: LDPFL

This repository contains the source code of the paper titled "Local Differential Privacy for Federated Learning". 
For more details, you can read the paper at https://doi.org/10.1007/978-3-031-17140-6_10

If you find that this work is related to your work and it is useful, please cite our work as:
 
```
 @inproceedings{mahawaga2022local,
  title={Local Differential Privacy for Federated Learning},
  author={Mahawaga Arachchige, Pathum Chamikara and Liu, Dongxi and Camtepe, Seyit and Nepal, Surya and Grobler, Marthie and Bertok, Peter and Khalil, Ibrahim},
  booktitle={European Symposium on Research in Computer Security},
  pages={195--216},
  year={2022},
  organization={Springer}
}
```

## The datasets
Download the SVHN dataset from http://ufldl.stanford.edu/housenumbers/

## Requirements 
- Keras==2.4.3
- numpy==1.23.4
- tensorflow==2.10.0
- torch==1.12.1
- matplotlib==3.3.2

Note: The python version used for the experiments: Python 3.9.4

## Files
- LDPFL.ipynb: The main program that demonstrates LDPFL on the SVHN dataset. This file includes the main flow of LDPFL.
- Randomizer.py: Defines the data randomization model.
- FL.py : Defines the federated learning setup.
- Helper.py : Defines the local model and provides the functionality required for plotting the outputs related to the local model performance.

## Usage
1. Install the requirements. 
2. Download the dataset and save it inside a folder named "data".
3. Make sure cuda is enabled. 
4. Run LDPFL.ipynb.


