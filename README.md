# Local Differential Privacy for Federated Learning: LDPFL

This repository contains the source code of of the paper titled "Local Differential Privacy for Federated Learning".

If you find that this work is related to your work and it is useful please cite our work as:
 
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

## The abstract of the paper

Advanced adversarial attacks such as membership inference and model memorization can make federated learning (FL) vulnerable and potentially leak sensitive private data. Local differentially private (LDP) approaches are gaining more popularity due to stronger privacy notions and native support for data distribution compared to other differentially private (DP) solutions. However, DP approaches assume that the FL server (that aggregates the models) is honest (run the FL protocol honestly) or semi-honest (run the FL protocol honestly while also trying to learn as much information as possible). These assumptions make such approaches unrealistic and unreliable for real-world settings. Besides, in real-world industrial environments (e.g., healthcare), the distributed entities (e.g., hospitals) are already composed of locally running machine learning models (this setting is also referred to as the cross-silo setting). Existing approaches do not provide a scalable mechanism for privacy-preserving FL to be utilized under such settings, potentially with untrusted parties. This paper proposes a new local differentially private FL protocol (named LDPFL) for industrial settings. LDPFL can run in industrial settings with untrusted entities while enforcing stronger privacy guarantees than existing approaches. LDPFL shows high FL model performance (up to 98%) under small privacy budgets (e.g., ε =0.5) in comparison to existing methods.

For more details, you can read the paper at https://doi.org/10.1007/978-3-031-17140-6_10

## The datasets
Download the SVHN dataset from http://ufldl.stanford.edu/housenumbers/

## Requirements 
- Keras==2.4.3
- numpy==1.23.4
- tensorflow==2.10.0
- torch==1.12.1
- matplotlib==3.3.2

## Files
> LDPFL.ipynb: The main program that demonstrates LDPFL on the SVHN dataset. This file includes the main flow of LDPFL.

> Randomizer.py: Defines the data randomization model.

> FL.py : Defines the federated learning functionality.

> Helper.py : Defines the local model and provides the functionality required for plotting the outputs related to the local model performance.

## Usage
1. Install the requirements 
2. Download the dataset and save it inside a folder named "data"
3. Make sure cuda is active 
4. Run LDPFL.ipynb


