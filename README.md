## Introduction

This repository contains the implementations of Stinger and its competitors. Stringer is a novel data poisoning based WF defense, which enables effective defense against WF attacks with low bandwidth overhead and only maintains one generator for all websites. Its overall achitecture is depected as follows.


<p align="center">
<img src="https://obsidian-tencent-1259097531.cos.ap-nanjing.myqcloud.com/Snipaste_2025-02-22_21-12-14.png" width="650px"/>
</p>



## Installation and Usage Hints

Stinger is created, trained and evuluated on Ubuntu 16.04 using Python 3.7.



## WF Defense  Reproducibility


### Stinger

**Create a conda environment:**

```
conda create -n wf python=3.7
conda activate wf
```

**Install the required packages:**

```
# install tensorflow-gpu
pip install tensorflow-gpu==1.15.5
# config cuda && cudnn 
conda install cudatoolkit=10.0
conda install cudnn=7.6.0
# install pytorch
conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch
# install other libs
pip install -r requirements.txt
```

#### Pre-train poison model to generate particular packet sequence

Stinger’s key idea is to guide WF classifiers to over-fit on a particular crafted packet sequence by manipulating training samples (misleading) and disable the overfitted WF classifier by changing that sequence (deceiving)


The screenshot of pretraning poison model:


<p align="center">
<img src="https://obsidian-tencent-1259097531.cos.ap-nanjing.myqcloud.com/Frame%202.png"/>
</p>


### The usage of the competitors are introduced as follows.


#### run_defender.py

Different WF defense algorithms are run to generate the perturbation sequence. Supports the use of different datasets and parameter sets

```
python run_defender.py stinger -d DF
python run_defender.py stinger -d AWF
```



#### ui/pages/plot.sh

Polymerization of multiple experimental results， the results are shown as follows:

<figure style="text-align: center;">
  <img src="https://obsidian-tencent-1259097531.cos.ap-nanjing.myqcloud.com/Snipaste_2025-02-10_22-23-41.png" alt="defense performance" style="max-width: 100%;">
</figure>


#### ui/pages/table.sh

Stinger’s SDR cross various surrogate model:


<figure style="text-align: center;">
  <img src="https://obsidian-tencent-1259097531.cos.ap-nanjing.myqcloud.com/Snipaste_2025-02-10_22-24-50.png" alt="SDR" style="max-width: 100%;">
</figure>



### Tamaraw[1]

#### Introduction:
Tamaraw forces the client and server to send packets at fixed intervals to mask timing patterns. 

#### Usage:
```
python run_defender.py tamaraw -d DF -i 0.024 -o 0.08
```


### WTF-PAD[2]
#### Introduction:
WTF-PAD  randomizes Tor traces through adaptive padding. 

#### Usage:
```
python run_defender.py wtf-pad -d DF -r 1
```

### FRONT[3]
#### Introduction:
FRONT randomizes the number and timing of dummy packets based on the Rayleigh distribution.

#### Usage:
```
python run_defender.py front -d DF -c 25 -s 100
```

### Mockingbird[4]
#### Introduction:
Mockingbird  uses gradient descent to approach an original traces to another target. 

#### Usage:
```
python run_defender.py mockingbird -d AWF -a 15
```

### ALERT[5]
#### Introduction:
ALERT generates adversarial perturbations without knowing traffic traces, thus can effectively resist adversarial training-aided WF attacks. Its key idea is to produce universal perturbations that vary among users.

#### Usage:
```
python run_defender.py alert -d AWF -min 0.89  -max 0.91 
```


## WF Attacks Reproducibility

### CUMUL[6]
#### Introduction
CUMUL belongs to statistical-based WFs. It extracts 104 statistical features and uses Support Vector Machine (SVM) with a radial basis kernel to fingerprint websites.
#### Usage
```
python run_attack.py -de $defense -a CUMUL -d $dataset
```

### DF[7]

#### Introduction
DF uses deep CNN to fingerprint websites and achieves state-of-the-art performance among deep architecture-based solutions.
#### Usage
```
python run_attack.py -de $defense -a DF -d $dataset
```


### AWF[8]
#### Introduction
AWF uses three deep learning-based classifiers on a larger dataset, demonstrating the effectiveness of deep learning-based WF attacks. We selectively implement the SDAE- and CNN-based solutions.
#### Usage
```
python run_attack.py -de $defense -a AWF-SDAE -d $dataset
python run_attack.py -de $defense -a AWF-CNN -d $dataset
```


### Var-CNN[9]
#### Introduction
Var-CNN uses the Resnet-18 network and achieves good performance on limited training data. According to Sanjit et al.,
#### Usage
```
python run_attack.py -de $defense -a Var-CNN -d $dataset
```


### GANDaLF[10]
#### Introduction
GANDaLF exploits semi-supervised learning GANs for WFs. 
#### Usage
```
python run_attack.py -de $defense -a GANDaLF -d $dataset
```

## Disclaimer

This project is still under development and may be missing at the moment. In addition, some paths may require you to modify.

## Reference


[1] X. Cai, R. Nithyanand, T. Wang, R. Johnson, and I. Goldberg, “A systematic approach to developing and evaluating website fingerprinting defenses,” in Proc. ACM SIGSAC Conf. Comput. and Commun. Secur., 2014, pp. 227–238.

[2] M. Juarez, M. Imani, M. Perry, C. Diaz, and M. Wright, “Toward an efficient website fingerprinting defense,” in Proc. Eur. Symp. on Res. in Comput. Secur., 2016, pp. 27–46.

[3] J. Gong and T. Wang, “Zero-delay lightweight defenses against website fingerprinting,” in Proc. USENIX Secur. Symp., 2020, pp. 717–734.

[4] M. S. Rahman, M. Imani, N. Mathews, and M. Wright, “Mockingbird: Defending against deep-learning-based website fingerprinting attacks with adversarial traces,” IEEE Trans. Inf. Forensics Secur., vol. 16, pp. 1594–1609, 2021.

[5] L. Qiao, B. Wu, H. Li, C. Gao, W. Yuan, and X. Luo, “Trace-agnostic and adversarial training-resilient website fingerprinting defense,” in IEEE Conf. on Comput. Commun. (INFOCOM), 2024, pp. 211–220.

[6] A. Panchenko, F. Lanze, J. Pennekamp, T. Engel, A. Zinnen, M. Henze, and K. Wehrle, “Website fingerprinting at internet scale,” in Proc. Netw. Distrib. Syst. Secur. Symp., 2016.

[7] P. Sirinam, M. Imani, M. Juarez, and M. Wright, “Deep fingerprinting: Undermining website fingerprinting defenses with deep learning,” in Proc. ACM SIGSAC Conf. Comput. and Commun. Secur., 2018, pp. 1928–1943.

[8] V. Rimmer, D. Preuveneers, M. Ju´arez, T. van Goethem, and W. Joosen, “Automated website fingerprinting through deep learning,” in Proc. Netw. Distrib. Syst. Secur. Symp., 2018.

[9] S. Bhat, D. Lu, A. Kwon, and S. Devadas, “Var-cnn: A data-efficient website fingerprinting attack based on deep learning,” Proc. Privacy. Enhancing Technol., vol. 1, no. 4, pp. 292–310, 2019.

[10] S. E. Oh, N. Mathews, M. S. Rahman, M. Wright, and N. Hopper, “Gandalf: GAN for data-limited fingerprinting,” Proc. Privacy. Enhancing Technol., vol.2021, no. 2, pp. 305–322, 2021.