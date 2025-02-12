
## Appendix

Stinger works relying on assumptions: (i) attackers’ training sets are defended by Stinger and (ii) surrogate and target models share vulnerabilities. This section studies the impacts of the assumptions via measuring Stinger’s SDR on raw traces, and evaluating Stinger’s SDR cross various surrogate models.

*A.Impact of training sets*

Stinger works relying on assumptions: (i) attackers’ training sets are defended by Stinger and (ii) surrogate and target models share vulnerabilities. This section studies the impacts of the assumptions via measuring Stinger’s SDR on raw traces, and evaluating Stinger’s SDR cross various surrogate models.

One key assumption for applying Stinger is that adversaries train WF classifiers on the Stinger-defended traces. As a result, the WF classifiers over-fit on P sequences and further fail to fingerprint victims’ websites. This assumption requires deploying Stinger on all Tor clients and middle nodes, thereby preventing adversaries from accessing the raw traces. In practice, however, adversaries can still access raw traces by (i) refusing to update their Tor clients to the Stinger-defended version or (ii) disabling Stinger once their client updates. Hence, we evaluate Stinger’s performance when adversaries can collect raw traces. The experimental setups are as follows. We train WF attacks on raw traces (DF and AWF) and evaluate their accuracy on Stinger-defended traces. Stinger’s SDR over undefended training sets can then be determined according to Eq. 14. The evaluations are presented in Fig. 10.

<figure style="text-align: center;">
  <img src="https://obsidian-tencent-1259097531.cos.ap-nanjing.myqcloud.com/Snipaste_2025-02-10_22-23-41.png" alt="defense performance" style="max-width: 100%;">
</figure>

**Fig.10 Stinger’s defense performance on undefended training datasets. Stinger and Stinger-Plus differ in bandwidth overheads: Stinger employs the default overhead (l=128), while Stinger-Plus uses doubled overhead (l=256).**


***Finding:*** <u>Stinger achieves comparable defense performance to its competitors on undefended training datasets. </u> On DF, the competitors’ average against six selected attacks is 61.20%, 43.37%, 70.14%, 53.82%, 45.77%, and 57.42%, respectively. In comparison, Stinger achieves SDRs of 88.04%, 49.34%, 86.37%, 69.33%, 34.29% and 77.66%, which are slightly higher than the competitors’ average. On AWF, Stinger similarly outperforms the competing defenses. This demonstrates that Stinger can work on undefended training sets and even slightly outperform existing defenses, which can be explained by adversarial examples. Adversarial examples are commonly observed in machine learning [42]– [45] due to model-related reasons [46]–[48] and data-related reasons [49]–[51]. The latter type is model-independent, such as perturbations in data distribution, data subspace, and data features. Obviously, Stinger generates model-related adversarial examples, as adversaries learn overfitted/poisoned models on Stinger-defended traces. Additionally, finding from Fig. 10 demonstrates that Stinger can also generated data-related adversarial examples since it can defend against WF attacks on undefended traces. In particular, like its camouflage-based counterparts, Stinger inserts dummy packets into raw traces during the deceiving stage, partially obscuring their distinguishing features. This also explains why Stinger-Plus achieves a significant improvement over Stinger. Although Stinger shows comparable performance, we still recommend that the Tor community deploy Stinger across all Tor clients and middle nodes to enhance SDR using fewer overheads.


*B.Impact of surrogate models*

Stinger’s performance is inevitably influenced by the choice of surrogate model, as it generates P sequences by exploiting the surrogate model’s vulnerabilities and uses these sequences to compromise target models. To evaluate the impact of the surrogate model, we implement various versions of Stinger, each using a different surrogate model. For each version, we evaluate its SDR against different WF classifiers. Consequently, we obtain confusion matrices between the surrogate models and the WF classifiers, which are shown in Fig. 11.



<figure style="text-align: center;">
  <img src="https://obsidian-tencent-1259097531.cos.ap-nanjing.myqcloud.com/Snipaste_2025-02-10_22-24-50.png" alt="SDR" style="max-width: 100%;">
</figure>


**Fig. 11. Stinger’s SDR cross various surrogate model.**

***Finding:*** <u>Concept drift is observed when the surrogate model differs from the target classifier.</u> In machine learning, concept drift refers to a decrease in model performance caused by changes in data or the surrogate model. As shown in Fig. 11, Stinger’s SDR decreases when the surrogate model differs from the real WF classifier, indicating there is performance discount when using the surrogate model to represent the WF target classifier. Although concept drift occurs, the reduction in performance is not very large, averaging 6.56% on DF set and 8.68% on AWF set. This demonstrates Stinger’s effectiveness even when the real WF classifier is inaccessible. Moreover, forcing the P model to defend against various types of surrogate classifiers is a promising approach to mitigate concept drift. However, Stinger cannot achieve this goal as it relies on neural surrogate models to compute gradients. This limitation inspires us to explore gradient-independent defenses to improve SDR. Thus exploring the surrogate model gradient-independent method (e.g., reinforcement learning models) is a promising way to further improve Stinger’s practicality against unknown WFs.


**Reference：**

[42] S. Han, C. Lin, C. Shen, Q. Wang, and X. Guan, “Interpreting adversarial examples in deep learning: A review,” ACM Comput. Surv., vol. 55, no. 14, 2023. 

[43] A. Serban, E. Poll, and J. Visser, “Adversarial examples on object recognition: A comprehensive survey,” ACM Comput. Surv., vol. 53, no. 3, pp. 1–38, 2020. 

[44] R. R. Wiyatno, A. Xu, O. Dia, and A. De Berker, “Adversarial examples in modern machine learning: A review,” arXiv preprint arXiv:1911.05268, 2019.

[45] J. Zhang and C. Li, “Adversarial examples: Opportunities and challenges,” IEEE trans. on neural netw. and learn. syst., vol. 31, no. 7, pp. 2578–2593, 2019. 

[46] A. Ross and F. Doshi-Velez, “Improving the adversarial robustness and interpretability of deep neural networks by regularizing their input gradients,” in Proc. the AAAI conf. on artif. intell., vol. 32, no. 1, 2018. 

[47] K. Nar, O. Ocal, S. S. Sastry, and K. Ramchandran, “Cross-entropy loss and low-rank features have responsibility for adversarial examples,” arXiv preprint arXiv:1901.08360, 2019. 

[48] A. Ross and F. Doshi-Velez, “Improving the adversarial robustness and interpretability of deep neural networks by regularizing their input gradients,” in Proc. the AAAI conf. on artif. intell., vol. 32, no. 1, 2018. 

[49] C.-J. Simon-Gabriel, Y. Ollivier, L. Bottou, B. Scholkopf, and D. Lopez- ¨ Paz, “First-order adversarial vulnerability of neural networks and input dimension,” in Proc. Int. conf. on mach. learn. (ICML), 2019, pp. 5809– 5817. 

[50] K. Grosse, P. Manoharan, N. Papernot, M. Backes, and P. McDaniel, “On the (statistical) detection of adversarial examples,” arXiv preprint arXiv:1702.06280, 2017.

[51] D. Diochnos, S. Mahloujifar, and M. Mahmoody, “Adversarial risk and robustness: General definitions and implications for the uniform distribution,” Proc. Advances in Neural Inf. Process. Syst., 2018.

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

## Config file

A record is automatically added to the mysql database when running a defense or attack method, so change the configuration in `config.ini` before running it.

## Web UI

There is a simple UI interface to show the result after attack/defense and how long it takes to run, the script is in `ui/web_ui.sh`, run or stop by executing the command in the project root directory:

```bash
cd {projectRoot}  
bash ui/web_ui.sh start   
bash ui/web_ui.sh stop    
bash ui/web_ui.sh restart 
```



