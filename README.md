
## Dataset

// TODO

## Run Program

environment: python3.7 + GPU (nvidia driver、cuda 10.0、cudnn 7.6.0) 


```bash
# create virtual env
conda create -n tf1 python=3.7
conda activate tf1
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

## Config the Dataset

A record is automatically added to the mysql database when running a defense or attack method, so change the configuration in `config.ini` before running it.

## Web UI

There is a simple UI interface to show the result after attack/defense and how long it takes to run, the script is in `ui/web_ui.sh`, run or stop by executing the command in the project root directory:

```bash
cd {projectRoot}  
bash ui/web_ui.sh start   
bash ui/web_ui.sh stop    
bash ui/web_ui.sh restart 
```



## Experiment Result


### Impact of training sets

We train WF attacks on raw traces (DF and AWF) and evaluate their accuracy on Stinger-defended traces. Stinger’s SDR over undefended training sets can then be determined according to equation. The evaluations are presented in the below figure:

<figure style="text-align: center;">
  <img src="https://obsidian-tencent-1259097531.cos.ap-nanjing.myqcloud.com/Snipaste_2025-02-10_22-23-41.png" alt="defense performance" style="max-width: 100%;">
  <figcaption style="font-style: italic; margin-top: 0.5em;">Stinger’s defense performance on undefended training datasets.
Stinger and Stinger-Plus differ in bandwidth overheads: Stinger employs the default overhead (l=128), while Stinger-Plus uses doubled overhead (l=256).</figcaption>
</figure>

Stinger achieves comparable defense performance to its competitors on undefended training datasets. On DF, the competitors’ average against six selected attacks is 61.20%, 43.37%, 70.14%, 53.82%, 45.77%, and 57.42%, respectively. In comparison, Stinger achieves SDRs of 88.04%, 49.34%, 86.37%, 69.33%, 34.29% and 77.66%, which are slightly higher than the competitors’ average. On AWF, Stinger similarly outperforms the competing defenses.


### Impact of surrogate models


Stinger’s performance is inevitably influenced by the choice
of surrogate model, as it generates P sequences by exploiting
the surrogate model’s vulnerabilities and uses these sequences
to compromise target models. To evaluate the impact of the
surrogate model, we implement various versions of Stinger, each using a different surrogate model. For each version, we evaluate its SDR against different WF classifiers. Consequently, we obtain confusion matrices between the surrogate models and the WF classifiers, which are shown in below figure:


<figure style="text-align: center;">
  <img src="https://obsidian-tencent-1259097531.cos.ap-nanjing.myqcloud.com/Snipaste_2025-02-10_22-24-50.png" alt="SDR" style="max-width: 100%;">
  <figcaption style="font-style: italic; margin-top: 0.5em;">
  Stinger’s SDR cross various surrogate model.
  </figcaption>
</figure>

Concept drift is observed when the surrogate model differs from the target classifier. In machine learning, concept drift refers to a decrease in model performance caused by changes in data or the surrogate model. As shown in the figure, Stinger’s SDR decreases when the surrogate
model differs from the real WF classifier, indicating there is performance discount when using the surrogate model to represent the WF target classifier. Although concept drift occurs, the reduction in performance is not very large, averaging 6.56% on DF set and 8.68% on AWF set. This demonstrates Stinger’s effectiveness even when the real WF
classifier is inaccessible. Moreover, forcing the P model to defend against various types of surrogate classifiers is a promising approach to mitigate concept drift. However, Stinger cannot achieve this goal as it relies on neural surrogate models to compute gradients. This limitation inspires us to explore gradient-independent defenses to improve SDR. Thus exploring the surrogate model gradient-independent method (e.g., reinforcement learning models) is a promising way to
further improve Stinger’s practicality against unknown WFs.
