---
date: 2025-03-07T22:55:45+08:00
draft: false
title: 一种新的KL散度估计
tags:
    - Machine Learning
    - Math
categories:
    - Machine Learning
image: cover.png
math : true
---

最近一直在研究LLM中的强化学习，其中KL散度作为一个关键的方法，通常用于作为正则，要求优化分布距离参考分布不能太远。[John Schulman的博客](http://joschu.net/blog/kl-approx.html)里讨论K2和K3，作为两种能保证KL估计在所有采样点处均非负的估计子。

然而这两个估计子都不够鲁棒，K2自不用说，方差特别大。广泛被大家采用的K3其实也有很大的问题，原因是估计中存在$\frac{p(x)}{q(x)}$项，当$q(x)$很小时这个值会非常大。这导致我们的优化过程中会不时出现很大的spike，容易带崩训练。

因此我们需要构造一个不会引起spike的KL估计子，这个估计子中不能包含$p(x)/q(x)$项。同时，我们也需要这个KL估计子是非负的，否则模型将可以很容易地hack这个KL。

直接上结论，我提出一个K4估计子

$$
K4(x; p, q)=\log \left(p^2(x) - 2p(x)q(x) + 2q^2(x)\right) - 2\log q(x)  \tag{1}
$$

这个算子的性能全方位优于已有算子，表现为具有更低的偏差，更低的方差，保证非负，保证没有spike。我用John的代码测试了K4：

```python
import torch.distributions as dis
import pandas as pd

p = dis.Normal(loc=0.5, scale=1.2)
q = dis.Normal(loc=1.3, scale=2.5)
x = q.sample(sample_shape=(10_000_000,))
truekl = dis.kl_divergence(p, q)
print("true", truekl)
logr = p.log_prob(x) - q.log_prob(x)
k1 = -logr
k2 = logr ** 2 / 2
k3 = (logr.exp() - 1) - logr

px = p.log_prob(x).exp()
qx = q.log_prob(x).exp()
k4 = (px**2 - 2*px*qx+2*qx**2).log() - 2 * q.log_prob(x)

results = {}
kl_names = ["k1", "k2", "k3", "k4"]
kl_estimators = [k1, k2, k3, k4]
for k, kl_name in zip(kl_estimators, kl_names):
    bias = (k.mean() - truekl) / truekl
    std = k.std() / truekl
    min = k.min()
    max = k.max()
    results[kl_name] = {
        "bias": bias.item(),
        "std": std.item(),
        "min": min.item(),
        "max": max.item()
    }
    
pd.DataFrame(results).T
```
结果：
||bias|std|min|max|
|-|-|-|-|-|
|**k1**|1.893368|6.842822|-8.004973e-01|49.598541|
|**k2**|10.049346|40.413708|2.842171e-14|1230.007690|
|**k3**|1.892912|5.611935|0.000000e+00|48.598541|
|**k4**|0.164644|0.717409|-2.384186e-07|0.918155|

<details>

K4在式$(1)$中的形式等价于

$$
\log\left\{\left(\frac{p(x)}{q(x)}-1\right)^2+1\right\}
$$

这显然是大于0且在$\frac{p(x)}{q(x)}=1$处取得最小值0。

</details>

### Citation

```
@misc{ZhangBlogKL,
  author       = {Zhang, Han},
  title        = {Yet another new {KL} estimator},
  year         = {2025},
  howpublished = {Blog post},
  url          = {https://fingertap.github.io/p/一种新的KL散度估计},
  urldate      = {2025-03-07}
}
```