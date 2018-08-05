# MachineLearningFoundations
Notebooks for Machine Learning Foundations by @hsuantien

---

## Coursera Links

- [机器学习基石上 (Machine Learning Foundations)-Mathematical Foundations](https://www.coursera.org/learn/ntumlone-mathematicalfoundations)
- [机器学习基石下 (Machine Learning Foundations)-Algorithmic Foundations](https://www.coursera.org/learn/ntumlone-algorithmicfoundations)

by [Hsuan-Tien Lin](https://www.csie.ntu.edu.tw/~htlin/)

## 前言介绍

《机器学习基石》是国立台湾大学资讯工程系的 **林轩田** 老师开设的课程（**中文授课**）。

该课程旨在从基础的角度介绍机器学习，包括机器学习的**哲学**、关键**理论**和核心**技术**。

从基础角度出发，既能保证学生能够了解机器学习的基本概念，同时对学生基础的要求最少，也能够保证课程不会太枯燥。

（如果从理论角度出发，需要深入掌握各种机器学习理论，花费大量时间，但却不实用；
而如果从技术角度出发，快速介绍多种机器学习方法，但无法清晰理解，难以选择和应用。）


## Lecture 1: The Learning Problem

—— 简介机器学习的概念和基本的数据符号表示

### 哲学思考：什么是`机器学习`？

人：
- 观察 -> 学习 -> 技巧
- Observation -> Learning -> Skill

机器：
- 数据 -> 机器学习 -> 技巧
- Data -> Machine Learning -> Skill

### 下一个问题：什么是`技巧`？

- 改进一些表现
- Improve some performance measure

### 那么，`为什么`要使用机器学习？

因为用传统的编程方式定义、解决某些问题非常难；但使用机器学习的方法可以让这个问题变得很简单。

- 构建复杂的系统

例子：
- 识别什么是树

其他的例子：
- 控制火星探测车（不了解火星的情况）
- 语音识别（难以定义这个问题的解决方法）
- 高频股市交易（人类无法实现）

### 机器学习的三个关键

1. 存在潜在的模式（exists some underlying patten）
  - 这样才存在改进的空间（反例：预测下一个随机数）

2. 无法用简单的编程实现
  - 否则没有必要使用机器学习（反例：识别图中是否存在环路）

3. 必须有能够反映这个问题的数据
  - 否则无法学习（反例：预测核能是否会毁灭人类）


如果有以上三点，那么用机器学习 **有可能** 可以解决这个问题；

### 机器学习的应用举例

略

### 机器学习的组成

数据输入（Input）：
<img src="http://latex.codecogs.com/svg.latex?\mathbf{x}\in\mathcal{X}"/>

结果输出（Output）：
<img src="http://latex.codecogs.com/svg.latex?\mathrm{y}\in\mathcal{Y}"/>

目标函数（Target function）：
<img src="http://latex.codecogs.com/svg.latex?f:\mathcal{X}\to\mathcal{Y}"/>

数据集（Data）：
<img src="http://latex.codecogs.com/svg.latex?\mathcal{D}=\left\{(\mathbf{x}_1,\mathrm{y}_1),(\mathbf{x}_2,\mathrm{y}_2),\cdots,(\mathbf{x}_N,\mathrm{y}_N)\right\}"/>

机器学习算法（Learning algorithm)：
<img src="http://latex.codecogs.com/svg.latex?\mathcal{A}"/>

函数集合（Hypothesis set）：
<img src="http://latex.codecogs.com/svg.latex?\mathcal{H}=\left\{h_k\right\};\;(g\in\mathcal{H})"/>

假设函数（Hypothesis <=> Skill）：
<img src="http://latex.codecogs.com/svg.latex?g:\mathcal{X}\to\mathcal{Y};\;(g\approx{f})"/>


机器学习的过程，就是：
- 在**符合**`目标函数`的**数据**上；
- 运用用`机器学习算法`；
- 从`函数集合`中；
- 得到`假设函数`的过程。

机器学习模型是由`机器学习算法`和`函数集合`组成。

### 机器学习（Machine Learning）与其他领域的区别

1. 机器学习 v.s. 数据挖掘（Data Mining）
  - 机器学习和数据挖掘可以互相帮助
  - 机器学习有时包含在数据挖掘中，数据挖掘的范围更加广泛
2. 机器学习 v.s. 人工智能（Artificial Intelligence）
  - 机器学习是实现人工智能的一种方法
  - 机器学习也是一种人工智能，人工智能的范围更加广泛
3. 机器学习 v.s. 统计学（Statistics）
  - 统计学是实现机器学习的一种方法
  - 机器学习更重视计算结构，而统计学更加重视数学的严谨性（当然也损失了很多）

---

## Lecture 2: Learning Answer Yes/No

—— 介绍感知机（线性分类器）的概念和数学表示
—— 介绍感知机的算法（PLA）和数学表示
—— 数学推导、证明 PLA 的可实现性

### 感知机（Perceptron）

考虑一个简单的分类问题，是否给一个顾客办理信用卡。

假设每个顾客有一系列的特征（Feature），比如年薪、花费、债务等：
<img src="http://latex.codecogs.com/svg.latex?\mathbf{x}=(\mathrm{x}_1,\mathrm{x}_2,\cdots,\mathrm{x}_d)"/>

计算特征的加权求和作为分数：

<img src="http://latex.codecogs.com/svg.latex?\sum_{i=1}^d\mathrm{w}_i\mathrm{x}_i"/>

如果客户的得分高于某个分数（threshold），则办理信用卡；若低于某个分数，则不办理信用卡。
因此有：

<img src="http://latex.codecogs.com/svg.latex?h(\mathbf{x})=\textrm{sign}\left\(\left\(\sum_{i=1}^d\mathrm{w}_i\mathrm{x}_i\right\)-\textrm{threshold}\right\)"/>

这就是**感知机**。

</br>

简化一下这个公式：

<img src="http://latex.codecogs.com/svg.latex?h(\mathbf{x})=\textrm{sign}\left\(\left\(\sum_{i=1}^d\mathrm{w}_i\mathrm{x}_i\right\)+\begin{matrix}\underbrace{-\textrm{threshold}}\\\mathrm{w}_0\end{matrix}\cdot\begin{matrix}\underbrace{+1}\\\mathrm{x}_0\end{matrix}\right\)"/>

<img src="http://latex.codecogs.com/svg.latex?h(\mathbf{x})=\textrm{sign}\left\(\sum_{i=0}^d\mathrm{w}_i\mathrm{x}_i\right\)=\textrm{sign}(\mathbf{w}^T\mathbf{x})"/>

每一种`权重`向量（ <img src="http://latex.codecogs.com/svg.latex?\mathbf{w}"/> ）就是一个假设函数 <img src="http://latex.codecogs.com/svg.latex?h"/>（Hypothesis）。

在二维空间中（ <img src="http://latex.codecogs.com/svg.latex?\mathbb{R}^2"/> ），每一种 <img src="http://latex.codecogs.com/svg.latex?h"/> 可以用一条直线表示，在这个直线上的值为0，直线将平面分为 +1 和 -1 两个部分。因此，感知机也叫**线性分类器（Linear/binary classifiers）**

### Perceptron Learning Algorithm (PLA)

—— A fault confessed is half redressed.

那么，如何选出最好的`假设函数`呢？

我们希望得到的`假设函数`近似等于`目标函数`：
<img src="http://latex.codecogs.com/svg.latex?g\approx{f}"/>

我们并不知道`目标函数`，但我们有符合`目标函数`的`数据`，因此，至少在这些数据中，这两个函数应该是近似的：

<img src="http://latex.codecogs.com/svg.latex?g\approx{f}\;\mathrm{on}\;\mathcal{D}\;\Rightarrow\;g(\mathbf{x}_n)\approx{f(\mathbf{x}_n)\approx{}\mathrm{y}_n}"/>

不过，因为`目标函数`所属的`函数集合` <img src="http://latex.codecogs.com/svg.latex?\mathcal{H}\;(g\in\mathcal{H})"/> 可以是无限大的，从中找到我们想要的`目标函数`非常难。

因此，可以先从`函数集合`中随意拿出一个函数 <img src="http://latex.codecogs.com/svg.latex?g_0"/>（可以用权重的向量 <img src="http://latex.codecogs.com/svg.latex?\mathbf{w}_0"/> 表示），
然后，在数据中优化这个函数的表现，这就是PLA (Cyclic PLA) 的思路。

在一个循环 *t* = 0,1,2,3,... 中：
>
> - 找到当前函数判断错误的数据： <img src="http://latex.codecogs.com/svg.latex?\textrm{sign}(\mathbf{w}_t^T\mathbf{x}_{n(t)})\ne\mathrm{y}_{n(t)}"/>
>
>
> - 使用这个数据修正函数（向量求和）： <img src="http://latex.codecogs.com/svg.latex?\mathbf{w}_{t+1}\gets\mathbf{w}_t+\mathrm{y}_{n(t)}\mathbf{x}_{n(t)}"/>
>
>
> - 直到每个数据都不出现错误时，循环停止，得到权重向量： <img src="http://latex.codecogs.com/svg.latex?\mathbf{w}_{\textrm{PLA}}\;\textrm{as}\;g"/>
>

但是，这个算法还有一些问题：
- 算法中的循环不一定会**停止**
- 算法能够保证在已有的数据中是正确的，但未必在**未知数据**中也是正确的

### Guarantee of PLA

- 那么，什么情况下PLA的循环会停止？

数据是线性可分的（Linear Separable）

- 当数据是线性可分的时候，PLA的循环就一定会停止吗？

当数据线性可分时，存在一条线（ <img src="http://latex.codecogs.com/svg.latex?\mathbf{w}_f"/> ）可以完美区分这个数据集，每一个数据都可以被这条线区分在正确的部分，因此有：

<img src="http://latex.codecogs.com/svg.latex?\mathrm{y}_{n(t)}\mathbf{w}^T_f\mathbf{x}_{n(t)}\geq\mathop{\min}_n\,\mathrm{y}_n\mathbf{w}^T_f\mathbf{x}_n>0"/>

（任意一个数据点的向量表示与分割线法向量的夹角小于90度，向量内积等于向量的长度与夹角cos值的乘积）

我们使用 **向量内积** 的方式来查看这个完美的分割线和我们 _T_ 循环中分割线的相似程度。

如果两个向量越相似，他们的向量内积越大。
此外，还需要考虑两个向量的模/长度，如果向量变长，內积也会变大，因此使用单位向量进行内积。
所以，以下公式可以衡量这两个向量的相似程度：

<img src="http://latex.codecogs.com/svg.latex?\frac{\mathbf{w}^T_f}{||\mathbf{w}_f||}\,\frac{\mathbf{w}_T}{||\mathbf{w}_T||}\;(\mathbf{w}_0=\mathbf{0})"/>

对于**分子**部分，有：

<img src="http://latex.codecogs.com/svg.latex?\mathbf{w}^T_f\mathbf{w}_T}=\mathbf{w}^T_f(\mathbf{w}_{T-1}}+\mathrm{y}_{n(T-1)}\mathbf{x}_{n(T-1)})"/>

<img src="http://latex.codecogs.com/svg.latex?\mathbf{w}^T_f\mathbf{w}_T}\geq\mathbf{w}^T_f\mathbf{w}_{T-1}}+\mathop{\min}_n\,\mathrm{y}_n\mathbf{w}^T_f\mathbf{x}_n"/>

迭代后有：

<img src="http://latex.codecogs.com/svg.latex?\mathbf{w}^T_f\mathbf{w}_T\geq\mathbf{w}^T_f\mathbf{w}_0+T\mathop{\min}_n\,\mathrm{y}_n\mathbf{w}^T_f\mathbf{x}_n=T\mathop{\min}_n\,\mathrm{y}_n\mathbf{w}^T_f\mathbf{x}_n"/>

对于**分母**部分，有：

<img src="http://latex.codecogs.com/svg.latex?||\mathbf{w}_T||^2=||\mathbf{w}_{T-1}}+\mathrm{y}_{n(T-1)}\mathbf{x}_{n(T-1)}||^2"/>

<img src="http://latex.codecogs.com/svg.latex?||\mathbf{w}_T||^2=||\mathbf{w}_{T-1}||^2+2\,\mathrm{y}_{n(T-1)}\mathbf{w}_{T-1}\mathbf{x}_{n(T-1)}+||\mathrm{y}_{n(T-1)}\mathbf{x}_{n(T-1)}||^2"/>

因为只有在某个数据出现错误时，才会使用这个数据更新向量，所以有：

<img src="http://latex.codecogs.com/svg.latex?\mathrm{y}_{n(T-1)}\mathbf{w}_{T-1}\mathbf{x}_{n(T-1)}\leq0"/>

所以，上面的公式可以简化为：

<img src="http://latex.codecogs.com/svg.latex?||\mathbf{w}_T||^2\leq||\mathbf{w}_{T-1}||^2+||\mathrm{y}_{n(T-1)}\mathbf{x}_{n(T-1)}||^2"/>

<img src="http://latex.codecogs.com/svg.latex?||\mathbf{w}_T||^2\leq||\mathbf{w}_{T-1}||^2+\mathop{\max}_n\,||\mathrm{y}_n\mathbf{x}_n||^2"/>

迭代后有：

<img src="http://latex.codecogs.com/svg.latex?||\mathbf{w}_T||^2\leq||\mathbf{w}_0||^2+T\mathop{\max}_n\,||\mathrm{y}_n\mathbf{x}_n||^2=T\mathop{\max}_n\,||\mathrm{y}_n\mathbf{x}_n||^2"/>

综上，

<img src="http://latex.codecogs.com/svg.latex?\frac{\mathbf{w}^T_f}{||\mathbf{w}_f||}\,\frac{\mathbf{w}_T}{||\mathbf{w}_T||}\geq\frac{T\mathop{\min}\limits_n\mathrm{y}_n\mathbf{w}^T_f\mathbf{x}_n}{||\mathbf{w}_f||\sqrt{T\mathop{\max}\limits_n||\mathrm{y}_n\mathbf{x}_n||^2}}\;(||\mathrm{y}_n||=1)"/>

<img src="http://latex.codecogs.com/svg.latex?\frac{\mathbf{w}^T_f}{||\mathbf{w}_f||}\,\frac{\mathbf{w}_T}{||\mathbf{w}_T||}\geq\sqrt{T}\cdot{C}"/>

其中，

<img src="http://latex.codecogs.com/svg.latex?C=\frac{\mathop{\min}\limits_n\mathrm{y}_n\frac{\mathbf{w}^T_f}{||\mathbf{w}_f||}\mathbf{x}_n}{\sqrt{\mathop{\max}\limits_n||\mathbf{x}_n||^2}}>0"/>


可见两个单位向量的內积会随着 _T_ 的增加而增加，这说明随着PLA的不断循环、更新，两个向量是越来越**接近**的；

同时，因为两个单位向量內积的最大值为 **1**，所以 _T_ 不可能无限增加；

因此，在数据**线性可分**时，PLA的循环**最终会停下来**，找到一个很好的分割线。

####### 怎么样！有没有感受到数学的NB之处！！ #######

### Non-Separable Data

不过，PLA仍然有一些问题：

- 需要数据是线性可分的，但是我们并不知道数据是否线性可分
- 数据是线性可分的假设过于强了，很多时候数据不是线性可分的（比如数据有噪声）
- 尽管当线性是可分的时候，PLA会停下来，但是我们并不知道需要多少个循环才能停下（参数中含有未知的 <img src="http://latex.codecogs.com/svg.latex?\mathbf{w}_f"/> ）


为了解决这些问题，我们首先应该假设**噪声**应该很小，多数的数据都是线性可分的；

因此我们可以找到一条线，使它在这个数据集中出现的错误最少：

<img src="http://latex.codecogs.com/svg.latex?\mathbf{w}_g\gets\mathop{\textrm{argmin}}_\mathbf{w}\sum_{n=1}^N[\![\mathrm{y}_n\ne\textrm{sign}(\mathbf{w}^T\mathbf{x}_n)]\!]"/>

但是这是一个 **NP-hard 问题**。

因此，我们修改了一下PLA的算法。

这个新算法的思路是在PLA的循环中，当每次找到一个新的分类器（线）时，检查这个分类器在所有数据中的表现。如果这个新的分类器比以前（口袋里）分类器的表现好，那么就留下这个新的分类器，否则，还保留旧的分类器。

这个算法叫就做 **口袋算法（Pocket Algorithm）**。

---

## Lecture 3: Types of Learning

—— 介绍不同的学习类型和方式：分类/回归，监督/无监督/强化，Batch/Online，Concrete/Raw/Abstract Feature

### 不同的输出空间

**分类** 问题：Binary Classification => Multiclass Classification

<img src="http://latex.codecogs.com/svg.latex?\mathcal{Y}=\left\{+1,-1\right\};\;\mathcal{Y}=\left\{1,2,3,\cdots,K\right\};"/>

- 健康/病人诊断
- 正常/垃圾邮件
- 衣服大小
- 硬币识别

**回归** 分析：Regression, bounded regression

<img src="http://latex.codecogs.com/svg.latex?\mathcal{Y}=\mathbb{R}\;\textrm{or}\;\mathcal{Y}=[\textrm{lower},\textrm{upper}]\subset\mathbb{R}"/>

- 股价、房价预测
- 天气、温度预测

结构 学习：Structured learning

Structure <img src="http://latex.codecogs.com/svg.latex?\equiv"/> Hyperclass, without class definition

- 自然语言处理，句子形式判断
- 蛋白质折叠预测

### 不同的输出标注

**监督学习**：Supervised Learning

有数据标注，每个数据都有相应的标注 <img src="http://latex.codecogs.com/svg.latex?(\mathbf{x}_n,\mathrm{y}_n)"/>

**无监督学习**：Unsupervised Learning

无数据标注，目标也比较分散

- 数据分群（~ unsupervised multiclass classification）
- 数据密度估计（~ unsupervised bounded regression）
- 异常值检测（~ unsupervised binary classification）

**半监督学习**：

略

**增强学习**：Reinforcement Learning

最难的，但是是最自然的学习方法，比如训练宠物

有输入，有一个“不受控制”控制的输出，还有一个对这个输出的评价 <img src="http://latex.codecogs.com/svg.latex?(\mathbf{x},\mathrm{\tilde{y}},\textrm{goodness})"/>


### 不同的流程

- Batch Leaning：收集一波数据，一波输入机器学习算法（最常用的一种，“填鸭式”）
- Online Learning：实时的输入数据，实时的改进，甚至最优解都可能是实时变化的（强化学习，通常也使用这种方式；还有PLA也可以很简单的实现；“上课式”）
- Active Learning：类似于Online Learning，通过对于特定输入进行“提问”获得标注，通常在获取标记成本比较高的应用中（“提问式”）

### 不同的输入空间

- **Concrete** Feature：输入的每一个维度（特征）都具有一定的物理意义，这些特征带有人类的智慧，相当于是被人类预处理的数据（比如钱币分类中的大小、信用卡用户的工资、图像的对称性等）
- **Raw** Feature：输入的特征更加的抽象，一个维度（特征）的物理意义对于这个问题而言不是那么的有意义（比如图像识别中的像素、声音识别中的信号）
- **Abstract** Feature：输入的特征没有物理意义，（比如用户音乐推荐中的音乐和用户ID）

特征工程（Feature Engineering）是指将Raw feature转换为Concrete feature的过程。

对于机器学习来说，越抽象越难。

---

## Lecture 4: Feasibility of Learning

——

### 哲学思考：机器学习真的是可能的吗？(Learning is impossible?)

- Two Controversial Answers

对于这个问题，可能有不同的答案。
任意一个答案都有可能是正确的，也有可能是错误的；
对于这种问题，再好的算法也可能永远无法完成。

![Snap01](./Snapshot/Snap01.png)

- 'Simple' Binary Classification Problem

![Snap02](./Snapshot/Snap02.png)

对于这个问题，我们可以得到多种函数，这些函数在数据集中都是完全正确的，但我们却不知道在未知的数据集中这些函数的表现如何。
如果任选一种函数，那么它很有可能在未知的数据中是错误的；
如果平均所有的函数，那么就相当于没有进行机器学习。

**没有免费午餐（No Free Lunch）**：

从已知的数据中获得与`目标函数`一样好的`假设函数`在很多情况下是不可能的，必须有某些**前提假设**，否则机器学习是不可能的。

Fun Time：嘲讽一下某些“智商测试”

![Snap03](./Snapshot/Snap03.png)

###















---
