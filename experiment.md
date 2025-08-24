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
| DF      | $SDR_t$ | 12.73 | 15.35 | 15.95 | 16.45 | 28.34 | 30.89 | 34.73 | 38.78 | 48.78 |
| DF      | $SDR_i$ | 49.70 | 38.76 | 37.34 | 38.13 | 22.87 | 26.24 | 22.34 | 16.83 | 9.82 |
| DF      | $SDR_o$ | 62.43 | 54.11 | 53.29 | 54.58 | 51.21 | 57.13 | 57.07 | 55.61 | 58.60 |
| AWF     | $SDR_t$ | 12.06 | 14.75 | 17.37 | 19.43 | 20.16 | 23.82 | 25.47 | 35.83 | 49.36 |
| AWF     | $SDR_i$ | 43.50 | 33.92 | 31.62 | 30.42 | 32.81 | 27.90 | 30.42 | 23.73 | 9.87 |
| AWF     | $SDR_o$ | 55.56 | 48.67 | 48.99 | 49.85 | 52.97 | 51.72 | 55.89 | 59.56 | 59.23 |

### Proportion of traces defended by other mechanisms is 30%

**Table B: Stinger’s SDR (%) against DF-based adaptive attack （Other defenses 30%）**
| dataset      | metric| 0    |10% | 20%| 30% |40%| 50%| 60% |70%|
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| DF      | $SDR_t$ | 23.57 | 22.17 | 25.87 | 28.61 | 32.92 | 34.31 | 39.73 | 43.78 |
| DF      | $SDR_i$ | 34.72 | 28.10 | 25.65 | 21.54 | 25.00 | 19.22 | 16.08 | 12.36 |
| DF      | $SDR_o$ | 58.29 | 50.27 | 51.52 | 50.15 | 57.92 | 53.53 | 55.81 | 56.14 |
| AWF     | $SDR_t$ | 13.68 | 15.91 | 16.74 | 22.80 | 29.85 | 34.84 | 39.29 | 41.17 |
| AWF     | $SDR_i$ | 42.45 | 41.17 | 40.95 | 34.28 | 28.54 | 24.29 | 18.54 | 18.26 |
| AWF     | $SDR_o$ | 56.13 | 57.08 | 57.69 | 57.08 | 58.39 | 59.13 | 57.83 | 59.43 |



### Proportion of traces defended by other mechanisms is 40%

**Table C: Stinger’s SDR (%) against DF-based adaptive attack （Other defenses 40%）**
| dataset | metric | 0%    | 10%   | 20%   | 30%   | 40%   | 50%   | 60%   |
|---------|--------|-------|-------|-------|-------|-------|-------|-------|
| DF      | $SDR_t$ | 24.26 | 25.89 | 27.12 | 37.87 | 40.51 | 44.35 | 48.00 |
| DF      | $SDR_i$ | 36.89 | 37.58 | 29.94 | 13.55 | 12.42 | 15.46 | 6.63 |
| DF      | $SDR_o$ | 61.15 | 63.47 | 57.06 | 51.42 | 52.93 | 59.81 | 54.63 |
| AWF     | $SDR_t$ | 25.73 | 27.91 | 30.70 | 38.85 | 42.49 | 46.35 | 48.25 |
| AWF     | $SDR_i$ | 32.27 | 31.08 | 29.15 | 22.37 | 18.30 | 14.39 | 12.13 |
| AWF     | $SDR_o$ | 58.00 | 58.99 | 59.85 | 61.22 | 60.79 | 60.74 | 60.38 |



### Proportion of traces defended by other mechanisms is 50%

**Table D: Stinger’s SDR (%) against DF-based adaptive attack （Other defenses 50%）**
| dataset      | metric| 0    |10% | 20%| 30% |40%| 50%| 
|-------|-------|-------|-------|-------|-------|-------|-------|
| DF | $SDR_t$ | --.-- |--.-- |--.-- |--.-- |--.-- |--.-- |--.-- |
| DF | $SDR_i$ | --.-- |--.-- |--.-- |--.-- |--.-- |--.-- |--.-- |
| DF | $SDR_o$ | --.-- |--.-- |--.-- |--.-- |--.-- |--.-- |--.-- |
| AWF| $SDR_t$ | --.-- |--.-- |--.-- |--.-- |--.-- |--.-- |--.-- |
| AWF| $SDR_i$ | --.-- |--.-- |--.-- |--.-- |--.-- |--.-- |--.-- |
| AWF| $SDR_t$ | --.-- |--.-- |--.-- |--.-- |--.-- |--.-- |--.-- |
