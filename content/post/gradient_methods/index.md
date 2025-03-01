---
title : 梯度下降收敛分析
date : '2019-07-28T23:34:55+08:00'
tags :
    - Machine Learning
    - Math
math : true
categories:
    - Machine Learning
image: cover.png
---

这篇里记录一下梯度下降在一般条件下的收敛分析。关键的思想有  

1. 利用泰勒公式+拉格朗日余项对函数做二阶展开。  
2. Smooth和Convexity分别对梯度、海塞引入上下界。  
3. 收敛$\Leftrightarrow f$是Lipschitz smooth（stochastic情形下也需要假设次梯度方差有界，这等于是说smooth，甚至比smooth更强因为smoothness等价于梯度平方有上界）。  
4. 收敛速度取决于$\|\nabla f(x)\|$的下界，即有多convex。

注：用$\|x-x^\star\|^2$和$f(x)-f(x^\star)$来推导结果是一致的，前者用完全平方公式展开，后者用泰勒展开。

---

## 表格结果

假设Lipschitz平滑常数为$M$，强凸常数为$m$，初始距离最优值距离$\|x_0-x^\star\|\leq r$，初始函数值差距$f(x_0)-f(x^\star)\leq R$，随机情形下假设$\mathbb{E}[\|\tilde g_{\theta}|\theta\|^2]\leq B^2$，有以下结果：

| Methods | Non-smooth | Smooth+Non-convex | Smooth+Convex | Smooth+Strong Convexity |
| :---: | :---: | :---: | :---: | :---: |
| Gradient Descent | May Divergent | Converge to local optima | $O(\frac{Mr^2}{K})$ | $O\left(\left(1-\frac{m}{M}\right)^KR\right)$ |
| Stochastic Gradient Descent | May Divergent | Almost surely converge to Critical points | $O(\frac{Br}{\sqrt{K}})$ | $O(\frac{B^2}{mK})$ |


对于一般非凸非光滑问题的收敛速度的界我们没有好的结果，因为这至少是NP难问题。

---

## 基础

考虑可导函数$f:\mathbb{R}^d\rightarrow\mathbb{R}$，在任一点处展开有：

$$
f(x) = f(y) + (x-y)^T\nabla f(y) + (x-y)^T\nabla^2f(c)(x-y),
\tag{1}
$$

其中$c$是$x$和$y$线段上一点。

### Convexity、Strong Convexity、Lipschitz Smooth

我们说他是**凸函数**，意味着$\forall c\in\mathbf{dom}f, \nabla^2f(c)\succeq0$，即$\forall x, y\in \mathbf{dom}f$

$$
f(x) \geq f(y) + (x-y)^T\nabla f(y).
$$

Lipschitz smooth和Strong convexity类似，是对$\nabla^2f(c)$引入了上下界。假设$f(x)$是$M$-Lipschitz smooth，以及$m$-strongly convex，有$\forall c\in\mathbf{dom}f$

$$
mI\preceq \nabla^2f(c)\preceq MI,
$$

其中$I$是单位矩阵。注意这样一来Convex可以看做是0-strongly convex。Lipschitz smooth直觉理解就是没有折点（例如$|x|$在$x=0$处），Strong convexity直觉理解就是没有盆地（一片区域的函数值相等）。如果我们考虑最优值$x^\star$，对于强凸和smooth我们分别有

$$
2m\left\{f(x)-f(y)\right\}\leq\left\|\nabla f(x)\right\|^2\leq 2m\left\{f(x)-f(y)\right\}
$$

（代入式(1)，右边对$y$求极值）

### Machine Learning Loss、Gradient Descent、Stochastic Gradient Descent

对于一族有监督统计学习模型$M_\theta$（模型的参数为$\theta\in\Theta$），设输入样本-标签对$(x, y)\in\mathcal{D}$满足$x\in\mathcal{X}, y\in\mathcal{Y}$，模型的决策函数（泛函）为$f_\theta:\mathcal{X}\rightarrow\mathcal{Y}$为连续映射。给定连续的损失函数$l:\mathcal{Y}\times\mathcal{Y}\rightarrow\mathbb{R}$，以及定义在$\mathcal{D}$上的概率度量$P$（密度函数为p），一个模型的好坏由期望损失给出：

$$
L(M_\theta)\triangleq \mathbb{E}_{(x, y)\in\mathcal{D}}[l(f_\theta(x), y)]
$$

如果我们拿不到输入空间$\mathcal{D}$（需要掌握所有可能的数据生成的方式，但是我们如果有这个生成方式还训练什么模型呢？直接查表不好吗？）和概率度量$P$，则这个期望只能由我们已有的数据集$D\subset\mathcal{D}$来近似，此时在$D$上的损失叫做经验损失：

$$
\hat{L}(M_\theta)\triangleq \frac{1}{|D|}\sum_{(x, y)\in D}l(f_\theta(x), y),
$$

这里假设了每个样本都是服从$P$，从$\mathcal{D}$中独立抽样出来的。则梯度下降\(GD\)的更新策略：

$$
\begin{align}
\theta_{k+1} &= \theta_k - \alpha \nabla l (\theta_k)\\
&=\theta_k - \alpha\cdot\frac{1}{D}\sum_{(x, y)\in D}\frac{\partial l(f_{\theta_k}(x), y)}{\partial \theta_k}.
\end{align}
$$

当$|D|$很大时计算$\nabla l(\theta_k)$的开支较大，随机梯度下降（SGD）对$D$进行采样，然后用带有随机性的梯度代替梯度下降中的$\nabla l(\theta_k)$。记采样得到的mini-batch为$D_k\subset D$，随机梯度定义为

$$
\nabla l_{D_k}(\theta_k) = \frac{1}{|D_k|}\sum_{(x, y)\in D_k}\frac{\partial l(f_{\theta_k}(x), y)}{\partial \theta_k}.
$$

由于任意一个样本$(x, y)$的损失的期望都是$L(M_\theta)$，可以很简单地证明随机梯度和经验梯度的期望都是期望损失在$\theta$处的梯度。

有时候随机梯度下降的随机性并不只是来自于对数据的采样（比如为了使目标变平滑，对数据加入随机白噪声），此时有可能使得随机梯度并不落在可行区域内（比如$\theta_k=[0.1, 0.9]$，而梯度为$-0.2, 1.1$，而我们希望$\|\theta\|_\infty\leq 1$），这时需要做一步正交投影操作，将更新后的$\theta_{k+1}$投影到可行区域$\Theta$内，方式是用最小二乘法在$\Theta$找一个距离$\theta_{k+1}$最近的点$\tilde\theta_{k+1}$，它满足$\forall \theta\in \Theta, \|\theta-\theta_{k+1}\|\geq\|\theta-\tilde\theta_{k+1}\|$，即投影后距离会缩小。

### Relationship between Gradient and Stochastic Gradient、Subgradient

$f$在$x$处的次梯度（Subgradient）$g_x\in\mathbb{R}^d$是所有满足一阶条件的向量：$\forall x, y\in \mathbb{R}^d,$

$$
f(y)\geq f(x) + g_x^T(y-x),
$$

所有次梯度的集合叫做Subdifferential，记作$\partial f(x)$。当$f$在$x$处可导时，$\partial f(x) = \{\nabla f(x)\}$。也即，如果函数是光滑的，就不用考虑次梯度，对于非光滑的问题一般会用次梯度下降来分析收敛性等。

我们称一个向量$\tilde g_x$为$f$在$x$处的带噪无偏次梯度（Noisy Unbiased Subgradient）若$\mathbb{E}[\tilde g_x]=g_x\in\partial f(x)$。**则随机梯度下降可以看做在梯度**$\nabla l(\theta_k)$**中引入一个零均值的加性噪声，而这个噪声为此次mini-batch的泛函，记作**$v(D_k)$。

---

## 收敛性分析

### Gradient Descent

**结论**：_梯度下降在无Smooth假设时可能不收敛，有M-smooth假设时收敛。达到精度_$\epsilon>0$_，凸时收敛速度为_$o(\frac{MR^2}{\epsilon})$_，强凸时收敛速度为_$o(\log_{1-m/M}\frac{\epsilon}{f(x_0)-f(x^\star)})$。

对$l(\theta)$没有任何假设的情况下，设$\theta^\star$使$l(\theta)$取得最小值，有：

$$
l(\theta_{k+1})-l(\theta^\star) 
=\left\{l(\theta_k)-\alpha\nabla l(\theta_k)^T\nabla l(\theta_k)+\frac{\alpha^2}{2}\nabla l(\theta_k)^T\nabla^2 l(c)\nabla l(\theta_k)\right\}-l(\theta^\star),
$$

这里$c$是$\theta_k$和$\theta_{k+1}$线段上一点。如果$\nabla^2l(c)\rightarrow\infty$，则$\forall \alpha > 0, l(\theta_{k+1})-l(\theta^\star) >l(\theta_k)-l(\theta^\star)$，因此第$k$步迭代并没有降低损失。我们可以构造一个函数，使得从某个起点$\theta_0$开始，每一步梯度下降都是发散的。因此我们需要限制$\nabla^2l(\theta)\preceq MI$，即$l(\theta)$是$M$-Lipschitz smooth的。代入smooth条件有

$$
l(\theta_{k+1})-l(\theta^\star)
\leq\left\{\frac{\alpha^2M}{2}-\alpha\right\}\|\nabla l(\theta_k)\|^2+l(\theta_k)-l(\theta^\star),
$$

等式右侧对$\alpha$求最小，得$\alpha=1/M$时

$$
l(\theta_{k+1})-l(\theta^\star)
\leq -\frac{1}{2M}\left\|\nabla l(\theta_k)\right\|^2+l(\theta_k)-l(\theta^\star).
$$

因此当$0 < \alpha < 2/M$时，我们都有$l(\theta_{k+1}) - l(\theta^\star) < l(\theta_k) - l(\theta^\star)$。

**要给出收敛速度需要对**$\|\nabla l(\theta_k)\|$**给出下界**。

#### Convex情形

$$
\begin{align}
&l(\theta^\star)\geq l(\theta_k)+\nabla l(\theta_k)^T(\theta^\star-\theta_k)\\
\Rightarrow~~& l(\theta_k)-l(\theta^\star)\leq\left\|\nabla l(\theta_k)^T(\theta^\star-\theta_k)\right\|\\
&~~~~~~~~~~~~~~~~~~~~~\leq\left\|\nabla l(\theta_k)\right\|\left\|\theta_k-\theta^\star\right\|\\
&~~~~~~~~~~~~~~~~~~~~~\leq\left\|\nabla l(\theta_k)\right\|\left\|\theta_0-\theta^\star\right\|\\
\Leftrightarrow~~&\|\nabla l(\theta_k)\|\geq\frac{l(\theta_k)-l(\theta^\star)}{\left\|\theta_0-\theta^\star\right\|}
\end{align}
$$

记$\eta_k=l(\theta_k)-l(\theta^\star)$，并假设$\|\theta_0-\theta^\star\|\leq R$，有

$$
\begin{align}
\eta_{k+1} &\leq \eta_k - \frac{1}{2M}\|\nabla l(\theta_k)\|^2\\
&\leq\eta_k-\frac{\eta_k^2}{2MR^2}
\end{align}
$$

化简方式是两边除以$\eta_k\eta_{k+1}$，整理得

$$
\begin{align}
&\frac{1}{\eta_{k+1}}-\frac{1}{\eta_k}\geq\frac{1}{2MR^2}\frac{\eta_k}{\eta_{k+1}}\geq\frac{1}{2MR^2}\\
\Rightarrow~~&\sum_{i=0}^k\frac{1}{\eta_{i+1}}-\frac{1}{\eta_i}=\frac{1}{\eta_{k+1}}-\frac{1}{\eta_0}\geq\frac{k+1}{2MR^2},
\end{align}
$$

即

$$
l(\theta_k)-l(\theta^\star)=\eta_k\leq\frac{2MR^2}{k},
$$

因此收敛速度是$O(1/k)$级别的次线性收敛。

#### $m$-strongly convex情形

根据强凸定义（参考Lipschitz smooth的几种定义相互的推导）：

$$
\|\nabla l(\theta_k)\|^2 \geq 2m\left(l(\theta_k)-l(\theta^\star)\right)
$$

代入下界有

$$
l(\theta_k)-l(\theta^\star)=\eta_k\leq\left(1-\frac{m}{M}\right)^k\eta_0
$$

### Stochastic Gradient Descent

随机情形下，假设$l(\theta)$为凸，记$\theta$处的带噪次梯度为$\tilde g_{\theta}$，并且$\exists B>0, \forall \theta\in\Theta, \mathbb{E}[\left\|\tilde g_\theta\right\|^2|\theta]\leq B^2$，且假设参数空间有界，即$\forall \theta \in \Theta, \|\theta\|\leq r$。考虑一般的随机梯度下降，即更新后可能落在可行域外，通过投影得到新的参数估计：

$$
\theta_{k+1} = Proj_\Theta(\theta_k-\alpha \tilde g_{\theta_k})，
$$

因为投影后的向量距离$\Theta$中的任意向量更近，有

$$
\begin{align}
\|\theta_{k+1}-\theta^\star\|^2 &\leq \|\theta_k-\alpha\tilde g_{\theta_k} -\theta^\star\|^2 \\
&=\|\theta_k-\theta^\star\|^2 + \alpha^2\|\tilde g_{\theta_k}\|^2-2\alpha\tilde g_{\theta_k}^T(\theta_k-\theta^\star)
\end{align}
$$

注意到这个式子中有两个随机变量$\theta_k$和$\tilde g_{\theta_k}$，后者依赖前者。因此我们这里对$\tilde g_{\theta_k}$在给定$\theta_k$的情况下求期望：

$$
\begin{align}
\mathbb{E}_{\tilde g_{\theta_k}}[\|\theta_{k+1}-\theta^\star\|^2|\theta_k] &\leq \|\theta_k-\theta^\star\|^2+ \alpha^2\mathbb{E}[\|\tilde g_{\theta_k}\|^2|\theta_k] - 2\alpha (\theta_k-\theta^\star)^T g_{\theta_k}\\
&\leq \|\theta_k-\theta^\star\|^2+ \alpha^2B^2 - 2\alpha (\theta_k-\theta^\star)^T g_{\theta_k}\\
\end{align}
$$

#### Convex情形

利用函数的凸性，有

$$
\mathbb{E}_{\tilde g_{\theta_k}}[\|\theta_{k+1}-\theta^\star\|^2|\theta_k] \leq \|\theta_k-\theta^\star\|^2+ \alpha^2B^2 - 2\alpha \{l(\theta_k)-l(\theta^\star)\}
$$

再对$\theta_k$求期望，则之后所有的期望都是对$\tilde g_{\theta_k}$和$\theta_k$求的联合期望，因此下标之后省略。记$\gamma_k = \mathbb{E}[\|\theta_{k}-\theta^\star\|^2]$，整理有

$$
\begin{align}
& \mathbb{E}[l(\theta_k)] - l(\theta^\star) \leq \frac{1}{2\alpha} (\gamma_k-\gamma_{k+1}) + \frac{\alpha B^2}{2}\\
\Rightarrow~~& \sum_{i=0}^k\left\{\mathbb{E}[l(\theta_k)]-l(\theta^\star)\right\} \leq \frac{1}{2\alpha}(\gamma_0-\gamma_{k+1}) + \frac{(k+1)\alpha B^2}{2}\\
&~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\leq \frac{r^2}{2\alpha} + \frac{(k+1)\alpha B^2}{2}
\end{align}
$$

利用$\min$函数的凹性，并记$k$次迭代中最好的参数为$\theta_{best}$

$$
\begin{align}
\sum_{i=0}^k\left\{\mathbb{E}[l(\theta_k)]-l(\theta^\star)\right\} &\geq (k+1)\left\{\min_{i=0,\dots,k}\mathbb{E}[l(\theta_i)]-l(\theta^\star)\right\} \\
&\geq (k+1)\left\{\mathbb{E}[\min_{i=0,\dots,k}l(\theta_i)] - l(\theta^\star)\right\} \\
& = (k+1)\left\{\mathbb{E}[l(\theta_{best})]-l(\theta^\star)\right\}
\end{align}
$$

最后不等式右边对$\alpha$求最小，整理得

$$
\mathbb{E}[l(\theta_{best})] - l(\theta^\star) \leq \frac{Br}{\sqrt{k}}
$$

#### Strong Convex情形

强凸情形下用常数步长$\alpha$先把$k$个式子加起来再取最优的$\alpha^\star$并不能达到最优的收敛界。为了使得上界更紧，我们允许步长可变，即每一步有一个步长$\alpha_k$，然后对每个式子都取一个精心构造的步长，最后达到$O(1/K)$的收敛。利用强凸有

$$
\mathbb{E}_{\tilde g_{\theta_k}}[\|\theta_{k+1}-\theta^\star\|^2]\leq(1-\alpha_k m)\|\theta_k-\theta^\star\|^2 + \alpha_k^2B^2-2\alpha\{l(\theta_k)-l(\theta^\star)\}
$$

我们这里不能对$\alpha_k$直接取最优，因为最优值依赖于$\|\theta_k-\theta^\star\|^2$，而我们不知道$\theta^\star$的具体值。这里**非常精妙**地构造了一个步长$\alpha_k=1/km$，我还没弄懂怎么想到这样取的。Anyway，代入步长，对$\theta_k$求期望并记$\gamma_k=\mathbb{E}[\|\theta_k-\theta^\star\|^2],\eta_k=\mathbb{E}[l(\theta_k)-l(\theta^\star)]$有

$$
\begin{align}
\eta_k \leq \frac{B^2}{2km} + \frac{(k-1)m\gamma_k}{2} - \frac{km\gamma_{k+1}}{2}
\end{align}
$$

注意到右边最后两项可以被telescope消掉。两边乘以$k$并取telescope sum有

$$
\begin{align}
&k\cdot\eta_k \leq \frac{B^2}{2m} + \frac{k(k-1)m\gamma_k}{2}-\frac{k^2m\gamma_{k+1}}{2}\\
&~~~~~~~~~\leq\frac{B^2}{2m}+\frac{k(k-1)m\gamma_k}{2}-\frac{(k+1)km\gamma_{k+1}}{2}\\
\Rightarrow~~&\sum_{i=1}^{k}i\cdot\eta_i \leq \frac{B^2k}{2m}+ 0 -\frac{(k+1)km\gamma_{k+1}}{2}
\end{align}
$$

再利用之前的技巧$\eta_i\geq\min_{j=1,\dots,k}\mathbb{E}[l(\theta_j)]-l(\theta^\star)\geq\mathbb{E}[l(\theta_{best})]-l(\theta^\star)$，有

$$
\begin{align}
&\frac{(k+1)k}{2}\cdot\left\{\mathbb{E}[l(\theta_{best})]-l(\theta^\star)\right\} \leq \frac{B^2k}{2m}-\frac{(k+1)km\gamma_{k+1}}{2}\\
\Leftrightarrow~~&\mathbb{E}[l(\theta_{best})]-l(\theta^\star)\leq \frac{B^2}{mk}
\end{align}
$$
