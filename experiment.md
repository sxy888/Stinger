# WELCOME TO THE EXTRA STINGER EVALUATION

## Performance against Adaptive attacks


### Setups
Stinger operates under strong assumptions that adversaries are unaware of its existence and train their website fingerprinting classifiers solely on Stinger-defended traces. However, this approach may be vulnerable to adaptive attacks where adversaries recognize Stinger’s deployment and subsequently modify their classifiers to
counteract the data poisoning effects. To address this concern, we evaluate Stinger’s robustness under such adaptive attack scenarios. We consider an adversary capable of collecting a mixed dataset comprising raw traces, Stinger -defended traces, and traces defended by other methods. The WF classifier is then trained on this composite dataset. Specifically, we increase the percentage of Stinger -defended traces from 0% to 90% in 10% increments, with a corresponding inverse decrease in the proportion of raw traces from 90% to 0%. When the adversary is aware of Stinger’s deployment, a common adaptive attack involves collecting Stinger -defended traces from multiple users and training the WF classifier on such data to compromise the defense. To simulate this scenario, we generate Stinger -defended traces using 10 different secrets, each representing a distinct user. DF is selected as the base model for adaptive attack due to its strongest performance. The current findings are presented in Table A. We commit to completing this extended analysis post-submission.


The overall SDR (denoted by SDRo) of a specific defense is defined as 1−ACCo, where ACCo is the WF’s generalized accuracy on the defended and unseen traces. Here defended means the traces are transformed by that specific defense, and unseen indicates that the traces are not present in the training set. Furthermore, the overall SDR is composed of two parts: training SDR (denoted by SDRt) and inferencing SDR (denoted by SDRi). Here training SDR is derived from the limitation of WF classifiers, which commonly cannot achieve perfect accuracy (100%) during the training phase. In particular, SDRt is defined as 1 − ACCt
, where ACCt represents the validating accuracy during the WF
classifier training. Besides, inferencing SDR refers to the accuracy degradation of the trained classifier resulting from adversarial manipulations during the inference stage, such as secret modifications introduced by Stinger. The inferencing SDR can be derived as 

$$SDR_i = SDR_o - SDR_t$$

where SDRo and SDRt
is computed on the unseen and the seen Stinger-defended traces.

### Proportion of traces defended by other mechanisms is 10%


**Table A: Stinger’s SDR (%) against DF-based adaptive attack （Other defenses 10%）**

<img src="https://obsidian-tencent-1259097531.cos.ap-nanjing.myqcloud.com/20250810213743.png"/>


As shown in the above table, the training SDR demonstrates a near-linear correlation with the increasing ratio of Stinger -defended traces. This relationship indicates that as more mixed Stinger -defended traces are introduced, the WF classifier experiences greater difficulty in model convergence, leading to progressively degraded validation accuracy. Concurrently, the inferencing SDR significantly decreases as Stinger-defended traces constitute a larger proportion of the training data, suggesting a gradual weakening of the Stinger’s secret based defensive efficacy. This is because WF classifiers are more robust to secret changes when the they are exposed to higher volumes of defended traces. The interplay of these
two competing factors produces a characteristic non-monotonic trend in the overall SDR, that is, an initial decline is followed by a period of relative stability, culminating in a subsequent increase. Our experimental results demonstrate that adaptive attacks against
Stinger remain fundamentally limited in effectiveness, achieving only approximately 50%, which is significantly smaller than these of the existing WF classifiers trained on raw traces (≥ 90%) and may fail to provide much meaningful information for adversaries.

### Proportion of traces defended by other mechanisms is 20%

**Table B: Stinger’s SDR (%) against DF-based adaptive attack （Other defenses 20%）**
| dataset      | metric| 0    |10% | 20%| 30% |40%| 50%| 60% |70%|80% |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| DF | $SDR_t$ | 12.73 |15.35 |15.95 |16.45 |28.34 |--.-- |--.-- |--.-- |--.-- |
| DF | $SDR_i$ | --.-- |--.-- |--.-- |--.-- |--.-- |--.-- |--.-- |--.-- |--.-- |
| DF | $SDR_o$ | --.-- |--.-- |--.-- |--.-- |--.-- |--.-- |--.-- |--.-- |--.-- |
| AWF| $SDR_t$ | --.-- |--.-- |--.-- |--.-- |--.-- |--.-- |--.-- |--.-- |--.-- |
| AWF| $SDR_i$ | --.-- |--.-- |--.-- |--.-- |--.-- |--.-- |--.-- |--.-- |--.-- |
| AWF| $SDR_t$ | --.-- |--.-- |--.-- |--.-- |--.-- |--.-- |--.-- |--.-- |--.-- |

