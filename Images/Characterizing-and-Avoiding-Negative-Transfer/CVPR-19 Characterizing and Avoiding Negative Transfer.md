# Characterizing and Avoiding Negative Transfer

## 摘要

​		当标记数据在特定目标任务中稀缺时，迁移学习通常可以利用相关源任务中的数据提供有效的解决方案。然而，当从一个不太相关的来源转移知识时，它可能会对目标域产生负面影响，这种现象被称为负转移。尽管负迁移现象普遍存在，但人们对它的描述通常不太正式，缺乏严格的定义、仔细的分析和系统的处理。本文提出了负迁移的形式化定义，并分析了负迁移的三个重要方面。在此基础上，提出了一种通过过滤不相关源数据来规避负迁移的新技术。该技术基于对抗网络，具有很强的通用性，可广泛应用于迁移学习算法。通过在4个不同难度的基准数据集上的实验，在6种最先进的深度传输方法上评估了所提出的方法。实验结果表明，该方法能够持续地提高所有基线方法的性能，并在很大程度上避免了负迁移，即使源数据是退化的。

## 介绍

​		迁移学习通过利用来自一个或多个源任务的有标记的数据来解决目标域数据稀缺的问题。这种方法已经在各种设置下进行了研究，并在广泛的应用中被证明是有效的。然而，迁移学习的成功并不总是有保证的。如果源域和目标域不够相似，那么从这种弱相关源进行传输可能会阻碍目标中的性能，这种现象称为负迁移。负迁移的概念在迁移学习学界得到了广泛认可。

​	   尽管有这些经验观察，但很少有研究工作发表了对负迁移的分析或预测，以下问题仍然存在:第一，虽然负迁移的概念相当直观，但不清楚应该如何准确地定义负迁移。例如，我们应该如何在测试时测量它?我们应该与什么类型的基线进行比较?其次，导致负迁移的因素是什么，如何利用这些因素来确定负迁移的发生也是未知的。尽管源域和目标域之间的差异当然是至关重要的，但我们确实知道负迁移的发生必须有多大，也不知道它是否是唯一的因素。第三，也是最重要的，给定有限或没有标记的目标数据，如何检测或避免负迁移。

​		在本文中，作者首先推导了负迁移的一个形式化定义，该定义在实践中具有普遍性和可处理性。这个定义进一步揭示了负迁移的三个潜在因素，让人们洞察到它何时会发生。在这些理论观察的激励下，作者开发了一种基于对抗网络的新型和高度通用的技术来对抗负迁移。在本文的方法中，一个同时估计边际分布和联合分布的鉴别器被用作一个门，通过降低源和目标之间的偏差来过滤潜在有害的源数据。

## 重新思考负迁移

​		本文中作者定义$P_{S}(X,Y)$  和$P_{T}(X,Y)$ 分别表示源域和目标域的联合分布。源域标记数据集为$S=\{x_{s}^{i},y_s^i\}_{i=1}^{n_s}$来自源域的联合分布$P_{S}(X,Y)$ ；有标记的目标域数据为$\mathcal T_l=\{x_{l}^{j},y_l^j\}_{j=1}^{n_l}$来自目标域的联合分布$P_{T}(X,Y)$；目标域未标记的数据为$\mathcal T_u=\{x_u^k\}_{k=1}^{n_u}$来自目标域的边缘分布$P_{T}(X)$，定义目标域数据的集合为$\mathcal T=(\mathcal T_l,\mathcal T_u)$。 

​		迁移学习的目标就是设计一个算法$A$，其输入为源域和目标域的数据$S,T$。并生成一个较单独使用目标域数据而言效果更好的模型（假设）$h=A(S,T)$。为了便于模型比较，作者使用标准期望风险，其定义为
$$
R_{P_T}(h):=\mathbb{E}_{x, y \sim P_T}[\ell(h(x), y)]
$$
该公式仅为一个示例，其中$l$为具体的分类损失，通常假设$n_s \gg n_l$。

​		负迁移的概念缺乏一个严格的定义。关于负迁移的一个被广泛接受的描述是“从源域转移知识会对目标域的学习产生负面影响”。这种描述虽然直观，但隐藏了许多负迁移的关键因素，其中作者强调以下三点:

1.  **负迁移的定义应该与算法相关**，在研究负迁移时应该只关注一种特定的算法，并在有源域数据和没有源域数据的情况下比较其性能。因此，作者定义在任意算法$A$下的负迁移条件（negative transfer condition, NTC）
$$
R_{P_T}(A(\mathcal{S}, \mathcal{T}))>R_{P_T}(A(\emptyset, \mathcal{T}))
$$
​		

​	  为了方便起见，作者定义负迁移差，它是负迁移的量化度量
$$
R_{P_T}(A(\mathcal{S}, \mathcal{T}))-R_{P_T}(A(\emptyset, \mathcal{T})),
$$
​	   当负迁移差为正时标志负迁移存在，反之亦然。

2. **联合分布的差异是负迁移的根源**

3. **目标域标记数据的多少也影响着负迁移**，仅使用未标记的目标数据将导致弱随机模型，NTC不太可能得到满足。当有标记的目标数据时，使用半监督学习方法可以获得更好的只针对目标的基准，因此负迁移相对更容易发生。

   另一方面，标记目标数据的数量直接影响到发现联合分布之间共享规律的可行性和可靠性。迁移学习算法的关键部分是发现源域的联合分布$P_{S}(X,Y)$和目标域联合分布$P_{T}(X,Y)$之间的相似性。当标记目标数据不可用时$(n_l=0)$，只能依靠边缘分布$P_S(X)$和$P_T(X)$的相似度，。相反，如果有相当数量的样本$(x_l,y_l)\sim P_T(X,Y)$和$(x_s,y_s)\sim P_S(X,Y)$，则问题是可以处理的。因此，一个理想的迁移学习算法可能能够利用标记的目标数据来减轻不相关源信息的负面影响。考虑到这些，作者接下来讨论如何系统地避免负迁移的问题。

## 提出方法

​		成功实现迁移和避免负面影像的关键是发现和利用$P_{S}(X,Y)$和$P_{T}(X,Y)$之间共享的底层结构。本文中提出的方法在基于对抗的领域自适应（DANN）方法上提出改进，DANN也就是我的毕设方向，该方法认为存在一片特征空间被源域与目标域共享，从而利用一个特征提取器F共同映射源域与目标域的特征，从而使得源域$P(F(X_S))$和目标域$P(F(X_T))$的特征无法被判别器D区分。与我的毕设不同的是，该方法是利用生成对抗网络来映射与判别特征而非梯度反转层，其公式如下：
$$
\begin{aligned}
\underset{F, C}{\operatorname{argmin}} \underset{D}{\operatorname{argmax}} & \mathcal{L}_{\mathrm{CLF}}(F, C)-\mu \mathcal{L}_{\mathrm{ADV}}(F, D), \\
\mathcal{L}_{\mathrm{CLF}}(F, C) &=\mathbb{E}_{x_l, y_l \sim \mathcal{T}_L}\left[\ell_{\mathrm{CLF}}\left(C\left(F\left(x_l\right)\right), y_l\right)\right] \\
&+\mathbb{E}_{x_s, y_s \sim \mathcal{S}}\left[\ell_{\mathrm{CLF}}\left(C\left(F\left(x_s\right)\right), y_s\right)\right], \\
\mathcal{L}_{\mathrm{ADV}}(F, D) &=\mathbb{E}_{x_u \sim P_T(X)}\left[\log D\left(F\left(x_u\right)\right)\right] \\
&+\mathbb{E}_{x_s \sim P_S(X)}\left[\log \left(1-D\left(F\left(x_s\right)\right)\right)\right] .
\end{aligned}
$$
其中$\mathcal{L}_{\mathrm{CLF}}$为有监督的标签分类损失，第一行为目标域有标记数据的标签分类损失，第二行为源域数据的标签分类损失，最小化标签分类器C的参数能够有效减小分类损失；$\mathcal{L}_{\mathrm{ADV}}$是一个标准的GAN损失，D为一个域判别器，用于区分提取的特征$F(x_u)$和$F(x_s)$是属于源域还是目标域，域得出一个分数，其数值在0到1之间，$D(F(x_u))$为真实的目标域测试集（即未被标记的目标域数据）的特征在通过域判别器后的得分，我们希望它尽可能地趋向1，$D(F(x_s))$为源域特征在通过域判别器后的得分，我们希望它尽可能地趋向0。GAN的基本训练思路就是，第一步：固定特征提取器F的参数，训练最优的域判别器，目的就是使得它能够更好地区分属于源域和目标域的特征，即最大化域判别器D的参数，使损失$\mathcal{L}_{\mathrm{ADV}}$最小；第二部：固定域判别器D的参数，调整特征提取器F的参数，特征提取器的目的就是尽可能地让源域与目标域的特征无法被域判别器区分，表现在公式上就是让$\mathcal{L}_{\mathrm{ADV}}$的损失变大，即使特征提取器F的参数变小。$\mu$是一个超参数用于平衡两个方程。

​		作者发现，DANN隐式地做了如下假设：

$$
P_S(Y\mid x_s)=P_T(Y\mid x_t)=P(Y\mid F(x_s))=P(Y\mid F(x_t))
$$

即假设源域和目标域同分布，则每个源域都能给迁移提供有用的信息，但根据上文，有些源域样本可能没法提供任何有用的信息。举一个反例假设对于任意$x_t$，$P_S(Y\mid x_s)\neq P_T(Y\mid x_t)$。由于$P(Y\mid F(x_s))=P(Y\mid F(x_t))$是GAN的结果，则目标域中存在一个$x^{\prime}\in \mathcal X_t $使得$F(x^{\prime})=F(x_s)$。因此$P(Y\mid F(x^{\prime}))=P(Y\mid F(x_s))$，同时$P(Y\mid F(x_s))$用于拟合$P_S(Y\mid x_s)$的分布，因此得出如下等式：
$$
P(Y\mid F(x^{\prime}))=P(Y\mid F(x_s))=P(Y\mid x_s)\neq P(Y\mid x^{\prime})
$$
在这种情况下，源域提供与目标域无关的信息，影响分类结果，造成负迁移。



**<font size=5>判别门</font>**

​		DANN的局限来自上文提出的非必要的假设即所有源域样本都能够为迁移学习提供有用的信息，为了剔除这一弊端，作者提出一种想法，对源域样本进行重新加权，例：标准的有监督学习公式可以被改写为
$$
\begin{aligned}
\mathcal{L}_{\mathrm{SUP}} &=\mathbb{E}_{x, y \sim P_T(X, Y)}\left[\ell_{\mathrm{CLF}}(C(F(x)), y)\right] \\
&=\mathbb{E}_{x, y \sim P_S(X, Y)}\left[\frac{P_T(x, y)}{P_S(x, y)} \ell_{\mathrm{CLF}}(C(F(x)), y)\right]
\end{aligned}
$$
密度比$\frac {P_T(x,y)}{P_S(x,y)}$变为源域的重要性权重 ，因此，问题简化为经典的密度比估计问题。

​		在本文中，尝试对GAN使用密度比估计的方法，域判别器接收接收输入x以及其对应的标签y，并尝试分类这组配对是来自源域还是目标域。由GAN中的公式推导得：域判别器的最优点为：$D(x,y)=\frac {P_T(x,y)}{P_T(x,y)+P_S(x,y)}$，等价于$\frac {P_T(x,y)}{P_S(x,y)}=\frac {D(x,y)}{1-D(x,y)}$。在本文中，$D(x,y)$转化为特征提取器提取后的参数$D(F(x),y)$，加入权重参数后，标签分类损失转化为：

转化前：
$$
\begin{aligned}
\mathcal{L}_{\mathrm{CLF}}(F, C) &=\mathbb{E}_{x_l, y_l \sim \mathcal{T}_L}\left[\ell_{\mathrm{CLF}}\left(C\left(F\left(x_l\right)\right), y_l\right)\right] \\
&+\mathbb{E}_{x_s, y_s \sim \mathcal{S}}\left[\ell_{\mathrm{CLF}}\left(C\left(F\left(x_s\right)\right), y_s\right)\right], \\
\end{aligned}
$$
转化后：
$$
\begin{aligned}
&\mathcal{L}_{\mathrm{CLF}}^{\text {gate }}(C, F)=\mathbb{E}_{x_l, y_l \sim \mathcal{T}_L}\left[\ell_{\mathrm{CLF}}\left(C\left(F\left(x_l\right)\right), y_l\right)\right] \\
&+\lambda \mathbb{E}_{x_s, y_s \sim \mathcal{S}}\left[\omega\left(x_s, y_s\right) \ell_{\mathrm{CLF}}\left(C\left(F\left(x_s\right)\right), y_s\right)\right], \\
&\omega\left(x_s, y_s\right)=\mathrm{SG}\left(\frac{D\left(x_s, y_s\right)}{1-D\left(x_s, y_s\right)}\right)
\end{aligned}
$$
其中SG(.)表示停止梯度，$\lambda$作为另一个超参数用于缩放密度比，由于密度比函数类似门函数，作者将该方法称作判别门。

​		同时，作者希望在对抗学习的过程中引入公式用于匹配源域和目标域的联合分布，其公式如下：

转化前：
$$
\begin{aligned}
\mathcal{L}_{\mathrm{ADV}}(F, D) &=\mathbb{E}_{x_u \sim P_T(X)}\left[\log D\left(F\left(x_u\right)\right)\right] \\
&+\mathbb{E}_{x_s \sim P_S(X)}\left[\log \left(1-D\left(F\left(x_s\right)\right)\right)\right] .
\end{aligned}
$$
转化后：
$$
\begin{aligned}
&\mathcal{L}_{\mathrm{ADV}}^{\text {aug }}(F, D)=\mathbb{E}_{x_u \sim P_T(X)}\left[\log D\left(F\left(x_u\right), \mathrm{nil}\right)\right] \\
&\quad+\mathbb{E}_{x_s \sim P_S(X)}\left[\log \left(1-D\left(F\left(x_s\right), \mathrm{ni} 1\right)\right)\right] \\
&\quad+\mathbb{E}_{x_l, y_l \sim \mathcal{T}_L}\left[\log D\left(F\left(x_l\right), y_l\right)\right] \\
&\quad+\mathbb{E}_{x_s, y_s \sim \mathcal{S}}\left[\log \left(1-D\left(F\left(x_s\right), y_s\right)\right)\right],
\end{aligned}
$$
其中nil表示一个加标签，它不提供任何的标签信息，它的引入使得D既可以作为边缘分布判别器，也可以作为联合分布判别器；与此同时，特征提取器F能够同时收到边缘分布判别器和联合分布判别器的梯度。它的一大优势是，在目标域有标记样本较少的情况下，它可以使用未标记的目标域样本。从理论上讲，联合分布匹配包含了边缘分布匹配，因为匹配的联合分布意味着匹配的边缘分布。然而，在实际操作中，标记的目标数据$\mathcal T_l$通常是有限的，使得联合匹配目标本身是不够的。

​		结合门控分类模型和增强对抗学习模型，作者提出了他们的迁移学习方法：
$$
\begin{aligned}
\underset{F, C}{\operatorname{argmin}} \underset{D}{\operatorname{argmax}} & \mathcal{L}_{\mathrm{CLF}}^{\mathrm{gate}}(F, C)-\mu \mathcal{L}_{\mathrm{ADV}}^{\mathrm{aug}}(F, D), \\
\end{aligned}
$$
模型结构如图所示：

![](https://github.com/RipC-me/TL-learning/blob/main/Images/Characterizing-and-Avoiding-Negative-Transfer/Model.png?raw=true)

尽管本文的模型主要基于DANN方法，但该方法具有高度的通用性，可以直接应用于其他对抗迁移学习方法上。

## 实验

### 数据集

 		作者在四个数据集（Digits,Office-31,Office-Home,VisDA）进行实验，在上文中提到的三种因素的影响下分析负迁移，并用六种迁移学习方法评估文中提出的鉴别门。

**Digits**  包含常用的数据集MNIST,USPS,SVHN，作者只考虑较难的情况SVHN→MNIST

**Office-31**  包含三个领域的图片A(Amazon),W(Webcam),D(DSLR)，作者评估了三个任务中的所有方法:W→D、A→D和D→A

**Office-Home ** 是一个更具挑战的数据集，包含15500张图片和65个种类来自多个搜索引擎和网络目录。它包含四个领域:艺术图像(Ar)，剪辑艺术(Cl)，产品图像(Pr)和现实世界图像(Rw)。作者想要测试更有趣和实用的迁移学习任务，包括从合成到现实世界的适应，因此作者考虑三个迁移任务:Ar→Rw, Cl→Rw和Pr→Rw。

**VisDA**  是另一个具有挑战性的合成真实数据集。作者使用训练集作为合成源，测试集作为真实世界的目标(合成→真实)。具体而言，训练集包含由渲染3D模型生成的152K张合成图像，测试集包含来自Youtube Bounding Box数据集的作物的72K张真实图像，均包含12个类别。

### 实验建立

控制三个因素：算法因素、分布差异因素、目标域标记数据数量的因素

**分布差异因素**：由于现有的基准数据集通常包含彼此相似的域，作者希望改变它们的分布，以便更好地观察负迁移的效应，实验中，引入两个摄动率$\epsilon_x$和$\epsilon_y$来控制两个域的边缘分布和条件分布。对每个源域数据绘制概率为x的伯努利变量，若返回1，则在乳香中加入一系列噪声，这样足以导致神经网络的分类错误。

**目标域标记数据数量的因素**：使用所有标记的源数据进行训练。对于目标数据，我们首先将50%作为训练集，其余50%用于测试。此外，我们使用所有的目标训练数据作为未标记的目标数据，并使用其中的L%作为标记的目标数据。

**算法因素**：作者评估了包括DANN、ADDA、PADA、GTA等在内的六种深度方法，对所有深度方法使用相同的特征提取器和分类体系结构。超参数设置如下：$\lambda=1$,$\mu$逐渐从0增长到1。为了测试是否发生负迁移，作者测量负迁移差(NTG)，即纯目标基线的准确性与原始方法的准确性之间的差距。例如，对于DANN，只有目标的基线是$\mathrm DANN_T$，它将标记的目标数据视为“源”数据，并照常使用未标记的数据。NTG为正表明负迁移的发生，反之亦然。

### 负迁移研究

**分布差异因素**

作者研究了在不同摄动率（$\epsilon_x$,$\epsilon_y$）和目标标记数据L%下不同方法的负迁移效应。DANN在不同摄动率$\epsilon$和L%的情况下，在Office-31数据集上的表现如表1所示

![](https://github.com/RipC-me/TL-learning/blob/main/Images/Characterizing-and-Avoiding-Negative-Transfer/Table1.png?raw=true)

由表可知，随着摄动率的增加，负迁移差也在增加。在某些情况下，如L% = 10%，我们甚至可以观察到NTG符号的变化。对于更细粒度的研究，我们通过图3(a)中$\epsilon$从0.0逐渐增加到1.0来研究更广泛的分布发散谱。虽然$\epsilon$在较小时DANN优于$\mathrm DANN_T$，但随着$\epsilon_x$的增加，DANN的性能迅速下降并低于$\mathrm DANN_T$的表现，表明发生了负迁移。

![](https://github.com/RipC-me/TL-learning/blob/main/Images/Characterizing-and-Avoiding-Negative-Transfer/Figure3.png?raw=true)

另一方面，通过固定$\epsilon_y=0$，并使用已知特别相似的两个域W和D，我们在表3中协变量移位的假设下研究负迁移，并观察到负迁移即使在高$\epsilon_x$和低L%的情况下也不会发生。这些实验结果证实了分布差异是负迁移的一个重要因素。

![](https://github.com/RipC-me/TL-learning/blob/main/Images/Characterizing-and-Avoiding-Negative-Transfer/Tabel3.png?raw=true)

**目标域标记数据数量的因素**

在表1中，固定一个$\epsilon$，能够发现负迁移差随着L%的增加而增加，在一种极端的情况下（L%=0%）时，即使两个域相聚很远（$\epsilon=0.9$），它们的负迁移差依旧为负。在图3(b)中，我们固定$\epsilon=0.2$，并绘制L%增加时的性能曲线。可以看到，随着目标域标记数据的增加，DANN表现逐渐被$\mathrm DANN_T$超越，这表明负迁移时相对的，它依赖于目标域标记数据的多少。

**算法因素**

作者经过实验发现，即使使用相同的数据，有些迁移学习方法更容易产生负迁移，对抗网络较好的源域和目标域匹配能力导致了更加严重的负迁移。

### 门控模型分析

作者将带门控模型的方法与各自在表2基准上的方法进行了比较，实验发现，即使使用有限数量的有标记的目标域数据，本文的方法也一致提高了所有深度学习方法的性能，更重要的是，本文的方法在很大程度上消除了关联度较低的源域数据的负面影像，避免了负迁移（$\mathrm DANN_{gate}$的平均NTG为负，而DANN的平均NTG为正）。具体来说，该方法在更困难的任务中取得了更高的精确度，例如在Office-Home和VisDA中，这主要是因为这些任务中的源域往往包含更多不相关的样本。这一发现也与表1和图3(a)的结果一致。

![](https://github.com/RipC-me/TL-learning/blob/main/Images/Characterizing-and-Avoiding-Negative-Transfer/table2.png?raw=true)

还可以发现，在源域恶化（图3(a)中$\epsilon=1.0$的情况下），门控模型$\mathrm DANN_{gate}$与$\mathrm DANN_{T}$结果箱单；另一方面，当源域与目标域密切相关时，门控模型与DANN结果相似，说明鉴别门能够控制最大迁移与负迁移之间的权重。

## 总结

在本文中，作者分析了负迁移的问题，并提出了一种新的鉴别门技术来避免它。作者表明负迁移与特定算法、分布差异和目标标记数据量直接相关。实验证明了这些结论和本文方法的有效性。本文的方法一致地提高了一些基本方法的性能，并在很大程度上避免了负迁移。
