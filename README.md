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

（如果从理论角度出发，需要深入掌握各种机器学习理论，花费大量时间，但却不实用；而如果从技术角度出发，快速介绍多种机器学习方法，但无法清晰理解，难以选择和应用。）


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
<img src="http://latex.codecogs.com/svg.latex?\mathcal{D}=\left\{(\mathbf{x}_1,\mathrm{y}_1),(\mathbf{x}_2,\mathrm{y}_2),\ldots,(\mathbf{x}_N,\mathrm{y}_N)\right\}"/>

机器学习算法（Learning algorithm）：
<img src="http://latex.codecogs.com/svg.latex?\mathcal{A}"/>

函数集合（Hypothesis set）：
<img src="http://latex.codecogs.com/svg.latex?\mathcal{H}=\left\{h_k\right\};\;(g\in\mathcal{H})"/>

假设函数（Hypothesis <=> Skill）：
<img src="http://latex.codecogs.com/svg.latex?g:\mathcal{X}\to\mathcal{Y};\;(g\approx{f})"/>

---

#### 小结：机器学习

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
---
---

## Lecture 2: Learning Answer Yes/No

—— 介绍感知机（线性分类器）的概念和数学表示

—— 介绍感知机的算法（PLA）和数学表示

—— 数学推导、证明 PLA 的可实现性

### 感知机（Perceptron）

考虑一个简单的分类问题，是否给一个顾客办理信用卡。

假设每个顾客有一系列的特征（Feature），比如年薪、花费、债务等：
<img src="http://latex.codecogs.com/svg.latex?\mathbf{x}=(\mathrm{x}_1,\mathrm{x}_2,\ldots,\mathrm{x}_d)"/>

计算特征的加权求和作为分数：

<img src="http://latex.codecogs.com/svg.latex?\sum_{i=1}^d\mathrm{w}_i\mathrm{x}_i"/>

如果客户的得分高于某个分数（threshold），则办理信用卡；若低于某个分数，则不办理信用卡。因此有：

<img src="http://latex.codecogs.com/svg.latex?h(\mathbf{x})=\mathrm{sign}\left(\left(\sum_{i=1}^d\mathrm{w}_i\mathrm{x}_i\right)-\mathrm{threshold}\right)"/>

这就是**感知机**。

---

简化一下这个公式：

<img src="http://latex.codecogs.com/svg.latex?\begin{align*}h(\mathbf{x})&\,=\mathrm{sign}\left(\left(\sum_{i=1}^d\mathrm{w}_i\mathrm{x}_i\right)+\begin{matrix}\underbrace{-\mathrm{threshold}}\\\mathrm{w}_0\end{matrix}\cdot\begin{matrix}\underbrace{+1}\\\mathrm{x}_0\end{matrix}\right)\\&\\&\,=\mathrm{sign}\left(\sum_{i=0}^d\mathrm{w}_i\mathrm{x}_i\right)\\&\\&\,=\mathrm{sign}(\mathbf{w}^T\mathbf{x})\end{align*}"/>

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
> - 找到当前函数判断错误的数据： <img src="http://latex.codecogs.com/svg.latex?\mathrm{sign}(\mathbf{w}_t^T\mathbf{x}_{n(t)})\ne\mathrm{y}_{n(t)}"/>
>
>
> - 使用这个数据修正函数（向量求和）： <img src="http://latex.codecogs.com/svg.latex?\mathbf{w}_{t+1}\gets\mathbf{w}_t+\mathrm{y}_{n(t)}\mathbf{x}_{n(t)}"/>
>
>
> - 直到每个数据都不出现错误时，循环停止，得到权重向量： <img src="http://latex.codecogs.com/svg.latex?\mathbf{w}_{\mathrm{PLA}}\;\mathrm{as}\;g"/>
>

但是，这个算法还有一些问题：
- 算法中的循环不一定会**停止**
- 算法能够保证在已有的数据中是正确的，但未必在**未知数据**中也是正确的

### Guarantee of PLA

- 那么，什么情况下PLA的循环会停止？

数据是线性可分的（Linear Separable）

- 当数据是线性可分的时候，PLA的循环就一定会停止吗？

当数据线性可分时，存在一条线（ <img src="http://latex.codecogs.com/svg.latex?\mathbf{w}_f"/> ）可以完美区分这个数据集，每一个数据都可以被这条线区分在正确的部分，因此有：

<img src="http://latex.codecogs.com/svg.latex?\mathrm{y}_{n(t)}\mathbf{w}^T_f\mathbf{x}_{n(t)}\,\geq\,\mathop{\min}_n\,\mathrm{y}_n\mathbf{w}^T_f\mathbf{x}_n>0"/>

（任意一个数据点的向量表示与分割线法向量的夹角小于90&deg;，向量内积等于向量的长度与夹角cos值的乘积）

我们使用 **向量内积** 的方式来查看这个完美的分割线和我们 _T_ 循环中分割线的相似程度。

如果两个向量越相似，他们的向量内积越大。此外，还需要考虑两个向量的模/长度（如果向量变长，內积也会变大）因此使用单位向量进行内积。

所以，以下公式可以衡量这两个向量的相似程度：

<img src="http://latex.codecogs.com/svg.latex?\frac{\mathbf{w}^T_f}{||\mathbf{w}_f||}\,\frac{\mathbf{w}_T}{||\mathbf{w}_T||}\;(\mathbf{w}_0=\mathbf{0})"/>

对于**分子**部分，有：

<img src="http://latex.codecogs.com/svg.latex?\begin{align*}\mathbf{w}^T_f\mathbf{w}_T&\,=\mathbf{w}^T_f(\mathbf{w}_{T-1}+\mathrm{y}_{n(T-1)}\mathbf{x}_{n(T-1)})\\&\,\geq\,\mathbf{w}^T_f\mathbf{w}_{T-1}+\mathop{\min}_n\,\mathrm{y}_n\mathbf{w}^T_f\mathbf{x}_n\end{align*}"/>

迭代后有：

<img src="http://latex.codecogs.com/svg.latex?\begin{align*}\mathbf{w}^T_f\mathbf{w}_T&\,\geq\,\mathbf{w}^T_f\mathbf{w}_0+T\mathop{\min}_n\,\mathrm{y}_n\mathbf{w}^T_f\mathbf{x}_n\\&\,\geq\,T\mathop{\min}_n\,\mathrm{y}_n\mathbf{w}^T_f\mathbf{x}_n\end{align*}"/>

对于**分母**部分，有：

<img src="http://latex.codecogs.com/svg.latex?\begin{align*}||\mathbf{w}_T||^2&\,=||\mathbf{w}_{T-1}+\mathrm{y}_{n(T-1)}\mathbf{x}_{n(T-1)}||^2\\&\,=||\mathbf{w}_{T-1}||^2+2\,\mathrm{y}_{n(T-1)}\mathbf{w}_{T-1}\mathbf{x}_{n(T-1)}+||\mathrm{y}_{n(T-1)}\mathbf{x}_{n(T-1)}||^2\end{align*}"/>

因为只有在某个数据出现错误时，才会使用这个数据更新向量，所以有：

<img src="http://latex.codecogs.com/svg.latex?\mathrm{y}_{n(T-1)}\mathbf{w}_{T-1}\mathbf{x}_{n(T-1)}\,\leq\,0"/>

所以，上面的公式可以简化为：

<img src="http://latex.codecogs.com/svg.latex?\begin{align*}||\mathbf{w}_T||^2&\,\leq\,||\mathbf{w}_{T-1}||^2+0+||\mathrm{y}_{n(T-1)}\mathbf{x}_{n(T-1)}||^2\\&\,\leq\,||\mathbf{w}_{T-1}||^2+\mathop{\max}_n\,||\mathrm{y}_n\mathbf{x}_n||^2\end{align*}"/>

迭代后有：

<img src="http://latex.codecogs.com/svg.latex?\begin{align*}||\mathbf{w}_T||^2&\,\leq\,||\mathbf{w}_0||^2+T\mathop{\max}_n\,||\mathrm{y}_n\mathbf{x}_n||^2\\&\,\leq\,T\mathop{\max}_n\,||\mathrm{y}_n\mathbf{x}_n||^2\end{align*}"/>

综上，

<img src="http://latex.codecogs.com/svg.latex?\begin{align*}\frac{\mathbf{w}^T_f}{||\mathbf{w}_f||}\,\frac{\mathbf{w}_T}{||\mathbf{w}_T||}&\,\geq\,\frac{T\mathop{\min}\limits{_n}\mathrm{y}_n\mathbf{w}^T_f\mathbf{x}_n}{||\mathbf{w}_f||\sqrt{T\mathop{\max}\limits{_n}||\mathrm{y}_n\mathbf{x}_n||^2}}\;(||\mathrm{y}_n||=1)\\&\\&\,\geq\,\sqrt{T}\cdot{C}\end{align*}"/>

其中，

<img src="http://latex.codecogs.com/svg.latex?C=\frac{\mathop{\min}\limits_n\mathrm{y}_n\frac{\mathbf{w}^T_f}{||\mathbf{w}_f||}\mathbf{x}_n}{\sqrt{\mathop{\max}\limits_n||\mathbf{x}_n||^2}}>0"/>


可见两个单位向量的內积会随着 _T_ 的增加而增加，这说明随着PLA的不断循环、更新，两个向量是越来越**接近**的；

同时，因为两个单位向量內积的最大值为 **1**，所以 _T_ 不可能无限增加；

因此，在数据**线性可分**时，PLA的循环**最终会停下来**，找到一个很好的分割线。

####### 怎么样！有没有感受到数学的NB之处！！ #######

### Non-Separable Data & Pocket Algorithm

不过，PLA仍然有一些问题：

- 需要数据是线性可分的，但是我们并不知道数据是否线性可分
- 数据是线性可分的假设过于强了，很多时候数据不是线性可分的（比如数据有噪声）
- 尽管当线性是可分的时候，PLA会停下来，但是我们并不知道需要多少个循环才能停下（参数中含有未知的 <img src="http://latex.codecogs.com/svg.latex?\mathbf{w}_f"/> ）


为了解决这些问题，我们首先应该假设**噪声**应该很小，多数的数据都是线性可分的；

因此我们可以找到一条线，使它在这个数据集中出现的错误最少：

<img src="http://latex.codecogs.com/svg.latex?\mathbf{w}_g\gets\mathop{\mathrm{argmin}}_\mathbf{w}\sum_{n=1}^N[\![\mathrm{y}_n\ne\mathrm{sign}(\mathbf{w}^T\mathbf{x}_n)]\!]"/>

但是这是一个 **NP-hard 问题**。

因此，我们修改了一下PLA的算法。

这个新算法的思路是在PLA的循环中，当每次找到一个新的分类器（线）时，检查这个分类器在所有数据中的表现。如果这个新的分类器比以前（口袋里）分类器的表现好，那么就留下这个新的分类器，否则，还保留旧的分类器。

这个算法叫就做 **口袋算法（Pocket Algorithm）**。

---
---
---

## Lecture 3: Types of Learning

—— 介绍不同的学习类型和方式：分类/回归，监督/无监督/强化，Batch/Online，Concrete/Raw/Abstract Feature

### 不同的输出空间

**分类** 问题：Binary Classification => Multiclass Classification

<img src="http://latex.codecogs.com/svg.latex?\mathcal{Y}=\left\{+1,-1\right\};\;\mathcal{Y}=\left\{1,2,3,\ldots,K\right\};"/>

- 健康/病人诊断
- 正常/垃圾邮件
- 衣服大小
- 硬币识别

**回归** 分析：Regression, bounded regression

<img src="http://latex.codecogs.com/svg.latex?\mathcal{Y}=\mathbb{R}\;\mathrm{or}\;\mathcal{Y}=[\mathrm{lower},\mathrm{upper}]\subset\mathbb{R}"/>

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

**半监督学习**：略

**增强学习**：Reinforcement Learning

最难的，但是是最自然的学习方法，比如训练宠物

有输入，有一个“不受控制”控制的输出，还有一个对这个输出的评价 <img src="http://latex.codecogs.com/svg.latex?(\mathbf{x},\mathrm{\tilde{y}},\mathrm{goodness})"/>


### 不同的流程

- Batch Leaning：收集一波数据，一波输入机器学习算法（最常用的一种，“填鸭式”）
- Online Learning：实时的输入数据，实时的改进，甚至最优解都可能是实时变化的（强化学习，还有以前我们提到的PLA也可以很简单的实现，“上课式”）
- Active Learning：类似于Online Learning，通过对于特定输入进行“提问”获得标注（通常在获取标记成本比较高的应用中，“提问式”）

### 不同的输入空间

- **Concrete** Feature：输入的每一个维度（特征）都具有一定的物理意义，这些特征带有人类的智慧，相当于是被人类预处理的数据（比如钱币分类中的大小、信用卡用户的工资、图像的对称性等）
- **Raw** Feature：输入的特征更加的抽象，一个维度（特征）的物理意义对于这个问题而言不是那么的有意义（比如图像识别中的像素、声音识别中的信号）
- **Abstract** Feature：输入的特征没有物理意义，（比如用户音乐推荐中的音乐和用户ID）

特征工程（Feature Engineering）是指将Raw feature转换为Concrete feature的过程。

对于机器学习来说，越抽象越难。

---
---
---

## Lecture 4: Feasibility of Learning

—— 哲学思考和数学讨论：**机器学习是否是可能的**

### 哲学思考：机器学习真的是可能的吗？(Learning is impossible?)

- Two Controversial Answers

对于这个问题，可能有不同的答案。任意一个答案都有可能是正确的，也有可能是错误的；对于这种问题，再好的算法也可能永远无法完成。

<div align=center><img width="70%" src="./Snapshot/Snap01.png"/></br></br></div>

- 'Simple' Binary Classification Problem

<div align=center><img width="70%" src="./Snapshot/Snap02.png"/></br></br></div>

对于这个问题，我们可以得到多种函数，这些函数在数据集中都是完全正确的，但我们却不知道在未知的数据集中这些函数的表现如何。

如果任选一种函数，那么它很有可能在未知的数据中是错误的；如果平均所有的函数，那么就相当于没有进行机器学习。

#### 没有免费午餐（No Free Lunch）

从已知的数据中获得与`目标函数`一样好的`假设函数`在很多情况下是不可能的，必须有某些**前提假设**，否则机器学习是不可能的。

Fun Time：嘲讽一下某些“智商测试”

<div align=center><img width="70%" src="./Snapshot/Snap03.png"/></br></br></div>

### 那么怎样才能确保一个问题“能被机器学习”？

—— Inferring Something Unknown

假设有一个罐子，里面有很多很多...很多的球，有一些是绿色的，有一些是橘色的；我们有没有办法估计罐子里面有多少比例的球是橘色的？

当然有！我们可以随机拿出 ***N*** 个球（Sample），看着这几个球中有多少比例的球是橘色的。

假设在罐子中橘色球的比例是 <i>&mu;</i>（未知的），而在我们Sample中橘色球的比例是 <i>&nu;</i>（已知的），那这个Sample内的比例可以说明Sample外的比例吗？

**有可能** 不能说明，但 **很有可能** 能够说明！

在 ***N*** 很大时，这两个比例很相近（相差小于 <i>&epsilon;</i> ）的概率是符合以下不等式：（Hoeffding's Inequality）

<img src="http://latex.codecogs.com/svg.latex?\mathbb{P}[|\nu-\mu|>\epsilon]\,\leq\,2\,\mathrm{exp}\,(-2\epsilon^2N)"/>


这个公式非常有用：
- 不需要"知道"未知的 <i>&mu;</i>
- 对于任何 _N_ 和 <i>&epsilon;</i> 都是有效的
- 在 _N_ 越大的时候偏差越小

### 在机器学习中使用类似方法

上面的讨论和统计的关系比较大，那么下面我们就来把这个转化到机器学习的问题中来。

- 罐子 **相当于** 机器学习问题中`输入`数据的空间
- 拿出来的 _N_ 个球（Sample） **相当于** 机器学习的`数据集`
- 球的颜色 **相当于** 某个`假设函数`在这个数据集（Sample）上的表现的好与不好
- 要估计的罐子中球的颜色 **相当于** 估计这个`假设函数`在整个数据空间上的表现好与不好

当 _N_ 很大时，且这个数据集是独立同分布（i.i.d.）的来自于整个输入数据空间中，我们就可以通过在数据集中`假设函数`的表现来评估`假设函数`在整个输入数据空间中的表现：

<img src="http://latex.codecogs.com/svg.latex?\mathbb{P}[|E_{in}(h)-E_{out}(h)|>\epsilon]\,\leq\,2\,\mathrm{exp}\,(-2\epsilon^2N)"/>

---

那么这样一来我们就能实现学习了吗？

**不一定**，刚才的不等式只能保证某个`假设函数`在符合特定情况下可以在输入空间中的表现很好；但是，`机器学习算法`未必会从`函数集合`中选出这个`假设函数`。

不过，我们可以用上述的方法作为验证（Verification）`机器学习算法`选出的某个`假设函数`的方法。

### 真正的机器学习

对于某个`假设函数`，如果它在输入数据中的表现是好的，要不要选择这个函数呢？

也 **不一定**！因为上述不等式是描述的是`假设函数`和`目标函数`差距很小的概率。即使概率很小，也**有可能**发生（两个函数差距很大的、小概率“不好的”的事件发生）。尤其是在有很多次事件（`函数集合`很大），且这种不好的事件可能会被**选择**的时候！

想象一下，如果有150人（`函数集合`）每人投5次硬币（`数据`），五次都是正面的那个人（`假设函数`），再以后的投硬币（`输出空间`）中就一定能一直正面吗？

这种“不好的”的事件，比如投币五次都是正面，**相当于** 某个`数据集`评价某种`假设函数`“看似”很好，但实际其在输入空间中的表现不好。

---

那么怎么办？还是 Hoeffding 不等式！它也保证了对于某一个`假设函数`，数据集中出现导致`假设函数`和`目标函数`出现差距的数据（“不好的”）**很少**。

当有从有 _M_ 个函数`函数集合`中选择`假设函数`时，某个数据对于某个`假设函数`都有可能是好的或者不好的。

对于任意一个数据，如果它对于这 _M_ 个函数中的某一个函数来说是“不好的”，我们就认为这是个不好的数据。因此，对于整个`函数集合`，不好的（BAD）数据出现的概率有：

<img src="http://latex.codecogs.com/svg.latex?\mathbb{P}_\mathcal{D}[\textbf{BAD}\,\mathcal{D}]=\mathbb{P}_\mathcal{D}[\textbf{BAD}\,\mathcal{D}\,\mathrm{for}\,h_1\,\textbf{or}\,\textbf{BAD}\,\mathcal{D}\,\mathrm{for}\,h_2\,\textbf{or}\,\cdots\,\textbf{or}\,\textbf{BAD}\,\mathcal{D}\,\mathrm{for}\,h_M]"/>

(Union bound)

<img src="http://latex.codecogs.com/svg.latex?\mathbb{P}_\mathcal{D}[\textbf{BAD}\,\mathcal{D}]\,\leq\,\mathbb{P}_\mathcal{D}[\textbf{BAD}\,\mathcal{D}\,\mathrm{for}\,h_1]+[\textbf{BAD}\,\mathcal{D}\,\mathrm{for}\,h_2]+\cdots+[\textbf{BAD}\,\mathcal{D}\,\mathrm{for}\,h_M]"/>

(Hoeffding)

<img src="http://latex.codecogs.com/svg.latex?\begin{align*}\mathbb{P}_\mathcal{D}[\textbf{BAD}\,\mathcal{D}]&\,\leq\,2\,\mathrm{exp}\,(-2\epsilon^2N)+2\,\mathrm{exp}\,(-2\epsilon^2N)+\cdots+2\,\mathrm{exp}\,(-2\epsilon^2N)\\&\\&\,\leq\,2M\,\mathrm{exp}\,(-2\epsilon^2N)\end{align*}"/>

这就是在有限空间中（Finite-bin）的 Hoeffding 公式。

当 ***N*** **很大**，而 ***M*** **有限** 的时候，我们就可以保证我们的数据是“可靠的”；

因此，我们也能够保证`假设函数`在数据中的表现很好的时候，它也在输入空间中的表现很好；

因此，**机器学习是可能实现的！**

思考： 通常 _M_ 都是无限大的，怎么办呢？我们将在后面进行分析

####### 再次感受数学的力量吧！！！ #######

---
---
---

## Lecture 5: Training versus Testing

—— 介绍当函数集合无限大时（ _M_ = &infin; ）机器学习面临的问题，并为解决这个问题做些准备

—— 介绍 Effective Number 和 Shatter 的概念

### 总结概括前面学到的内容

<img src="http://latex.codecogs.com/svg.latex?\begin{matrix}E_{out}(g)\underbrace{\approx}_\mathrm{test}{E}_{in}(g)\underbrace{\approx}_\mathrm{train}0\end{matrix}"/>

经过前面的学习，我们知道机器学习问题可以被分为两个部分：
1. 确保 <i>E</i><sub>in</sub> (<i>g</i>) 和 <i>E</i><sub>out</sub> (<i>g</i>) 是相近的
2. 确保 <i>E</i><sub>in</sub> (<i>g</i>)足够小

<div align=center><img width="70%" src="./Snapshot/Snap04.png"/></br></br></div>

_M_ 在个过程中起到什么作用呢？

<img src="http://latex.codecogs.com/svg.latex?\mathbb{P}[|E_{in}(h)-E_{out}(h)|>\epsilon]\,\leq\,2M\,\mathrm{exp}\,(-2\epsilon^2N)"/>

- 如果 _M_ 很小，那么 (1) 是可以实现的，但是 (2) 不能（因为选择空间小，不一定能够选到让 <i>E</i><sub>in</sub> (<i>g</i>) 很小的 <i>g</i> ）
- 如果 _M_ 很大，那么 (1) “不好的”事情发生的概率会变大，但是 (2) 更有可能实现

因此，_M_ 在这个问题中也是很重要的，当 _M_ 无限大的时候该怎么办？

我们希望能有一个 **有限的** _m_ 来代替无限的 _M_，并且仍然能够保证这个不等式的成立。

### Effective Number of Line

回顾一下 _M_ 这个“讨厌的”项是怎么来的？

是在我们使用 **Union bound** 将“不好的”数据出现的概率拆成对每个 _h_ “不好的”概率之和：

<img src="http://latex.codecogs.com/svg.latex?\mathbb{P}_\mathcal{D}[\textbf{B}_1\,\mathrm{or}\,\textbf{B}_2\,\mathrm{or}\,\ldots\,\textbf{B}_M]\,\leq\,\mathbb{P}[\textbf{B}_1]+\mathbb{P}[\textbf{B}_2]+\cdots+\mathbb{P}[\textbf{B}_M]"/>

当 _M_ 无限大的时候，我们就加和了无限多个项，这导致了我们面临问题。

---

#### Union Bound

不过，这个 **Union bound** 的使用其实是太过宽松了（公式右边远大于左边）：

考虑两个非常相似的 _h_<sub>1</sub> 和 _h_<sub>2</sub>，因为它们非常相似，因此它们的表现也是非常相似的；

所以让它们犯错误的数据（“不好的”数据）也是非常相似的；

所以把这些“重叠”在一起的事件发生的概率，用每个事件**单独**发生的概率加的方式替代，其实是过度高估了（Over-estimation）的。

因此，我们希望我们能够找出`假设函数`中有重叠的部分，把这些`假设函数`分成（有限的）几类（_m_），来减少这个不等式被过度高估的右边。

---

#### Effective Number

下面我们考虑一个平面的直线（线性分类器）：
- 总共（`函数集合`中）有多少条线（_M_）？
**无数条**！
- 根据 1 个数据点 **x**<sub>1</sub>，可能把这些直线分成多少种？
**2种**，产生判断 **x**<sub>1</sub> = -1 的直线和产生判断 **x**<sub>1</sub> = +1 的直线；
- 根据 2 个数据点 **x**<sub>1</sub>, **x**<sub>2</sub>，可能把这些直线分成多少种？
**4种**，可以分别用这些线对 **x**<sub>1</sub>, **x**<sub>2</sub> 的判断值表示：(0,0)，(0,1)，(1,0)，(1,1)；
- 根据 3 个数据点，可能把这些直线分成多少种？
**最多8种**！因为在三点共线的情况下，有些判断值不可能出现！比如(0,1,0) 和 (1,0,1)；
- 根据 4 个数据点，可能把这些直线分成多少种？
**最多** ***14*** **种**！在任意的情况下都会有一些判断值的组合不可能出现！（比如产生这样判断值的直线在这个平面上是不存在的 <img src="http://latex.codecogs.com/svg.latex?\begin{smallmatrix}0&1\\1&0\end{smallmatrix}"/> ）
- 根据 ***N*** 个数据点，可能把这些直线分成多少种？
**最多 2<sup><i>N</i></sup>** 种！不过当 _N_ 超过某个值之后这个值 effective(_N_) < 2<sup>_N_</sup> ！

因此，**如果 (1)** 能够使用这个值替换掉 _M_ ，就有

<img src="http://latex.codecogs.com/svg.latex?\mathbb{P}[|E_{in}(h)-E_{out}(h)|>\epsilon]\,\leq\,2\cdot\,\mathrm{ef\/fective}(N)\cdot\mathrm{exp}\,(-2\epsilon^2N)"/>

那么，**如果 (2)** effective(_N_) << 2<sup>_N_</sup> ，则 **机器学习就是可能的**！

### Effective Number of Hypothesis

在上面我们提到的，一个将数据区分成不同判断值的多种 **Hypotheses集合**，叫做 **Dichotomy**，有：

Hypotheses: &nbsp; <img src="http://latex.codecogs.com/svg.latex?\mathcal{H}\in\mathbb{R}^2"/>

Dichotomies: &nbsp; <img src="http://latex.codecogs.com/svg.latex?\mathcal{H}(\mathbf{x}_1,\mathbf{x}_2,\ldots,\mathbf{x}_N)\,\leq\,2^N"/>

Dichotomy 的大小取决于`输入空间`，因此在某个输入空间中，最大的 Dichotomy 的 **大小** 是`输入空间`的函数。

这个函数叫做 **成长函数**（Growth Function）：

<img src="http://latex.codecogs.com/svg.latex?m_{\mathcal{H}}(N)=\mathop{\max}_{\mathbf{x}_1,\mathbf{x}_2,\ldots,\mathbf{x}_N\in\mathcal{X}}|\mathcal{H}(\mathbf{x}_1,\mathbf{x}_2,\ldots,\mathbf{x}_N)|\,\leq\,2^N"/>

---

考虑一维空间中的几个例子：

- Positive Rays

<div align=center><img width="70%" src="./Snapshot/Snap05.png"/></br></br></div>

- Positive Intervals

<div align=center><img width="70%" src="./Snapshot/Snap06.png"/></br></br></div>

这些例子中的`成长函数`都远远小于2<sup>_N_</sup>

---

考虑二维空间中的一个例子：

如果 **x** 是在一个凸（Convex）的区域中，则为 +1，否则为 -1；

这个`函数集合`的`成长函数`是多少？

考虑将这 _N_ 个点随机放在一个圆上，任意一种分类结果（判断值）都可以通过选取所有判断值为+1的点作为顶点，绘出一个多边形。

因此`成长函数`是 2<sup>_N_</sup>。

<div align=center><img width="70%" src="./Snapshot/Snap07.png"/></br></br></div>

这种情况，我们称为这 _N_ 个输入被这个`函数集合` “击碎”（**Shatter**，完全二分的）

### Break Point

#### 稍微总结一下

上面我们提到，如果
1. 使用 effective(_N_) 替换掉 _M_
2. 且 effective(_N_) << 2<sup>_N_</sup>

那么机器学习就是可能的：

<img src="http://latex.codecogs.com/svg.latex?\mathbb{P}[|E_{in}(h)-E_{out}(h)|>\epsilon]\,\leq\,2\cdot\,m_{\mathcal{H}}(N)\cdot\mathrm{exp}\,(-2\epsilon^2N)"/>

因此，我们希望决定 effective(_N_) 大小的这个 **成长函数** 是比较小的，希望它是多项式形式的而不是指数形式的，这样才能够保证（在 _N_ 足够大的时候）可以进行机器学习。

上面我们还提到了 Shatter 的概念，很明显，Shatter 对于我们来说是不好的，因为 Shatter 的时候成长函数是 2<sup>_N_</sup>。

---

#### 引入 Break Point

下面我们引入一个新的概念：

对于一个函数集合，当 _k_ 个输入不能被 Shatter 的时候，就称 _k_ 为 **Break Point** 。当然，对于 _k_+1, _k_+2, ... 来说，都不能 Shatter。因此最小的 _k_ 对于我们来说就是非常重要的，可以帮助我们减小成长函数。

回顾我们之前的例子：
- Positive Rays

<img src="http://latex.codecogs.com/svg.latex?k=2\,,\,m_{\mathcal{H}}(N)=N+1=O(N)"/>

- Positive Intervals

<img src="http://latex.codecogs.com/svg.latex?k=3\,,\,m_{\mathcal{H}}(N)=\frac{1}{2}N^2+\frac{1}{2}N+1=O(N^2)"/>

- Convex Sets

<img src="http://latex.codecogs.com/svg.latex?k=+\infty\,,\,m_{\mathcal{H}}(N)=2^N"/>

- 2D Perceptrons

<img src="http://latex.codecogs.com/svg.latex?k=4\,,\,m_{\mathcal{H}}(N)<2^N"/>

我们猜测，当有 Break Point _k_ 的时候，<img src="http://latex.codecogs.com/svg.latex?m_{\mathcal{H}}(N)=O(N^{k-1})"/>

下面我们来证明。

---
---
---

## Lecture 6: Theory of Generalization

—— 数学推倒存在 Break Point 时候成长函数的上限

—— 数学推倒解决函数集合（ _M_ ）无限大时的机器学习

—— 证明 2D Perceptrons 是可以机器学习的

### 考虑 Break Point 带来了什么

我们还有两个问题没有解决：
1. 成长函数不能是以 _N_ 为指数的形式
2. 能否用成长函数来代替 _M_

我们先来解决 (1)，下一章解决 (2)。

如果已知 Break Point _k_ = 2，那么：
- 当 _N_ = 1 的时候，成长函数应该是 2 ( 2<sup>1</sup> )；
- 当 _N_ = 2 的时候，成长函数应该小于 4 ( 2<sup>2</sup> )，最大为 3；
- 当 _N_ = 3 的时候，这三个点中的任何两个点都不能 Shatter，否则 _k_ = 2 就不成立；成长函数最大为 4，远小于 2<sup>_N_</sup>！

可见 **Break Point** 对`成长函数`进行了很强的限制！太好了！

我们希望能够：
1. **找到 Break Point**
2. 证明有 Break Point 后，**成长函数** 是一个 **多项式** 的形式

我们先来看 (2)。

### Bounding Function

我们定义一个 **上限函数**（Bounding Function，_B_(_N_, _k_) ）：对 _N_ 个数据来说，在 Break Point 为 _k_ 的时候，`成长函数`可能的最大值。

我们将`成长函数`转化成`上限函数`的好处是：
1. 它是一个 _N_ 和 _k_ 组合和值（很有可能不是 _N_ 为指数的形式）
2. 它和`假设函数`没有关系

---

当 _k_ > _N_ 时，_B_(_N_, _k_) = 2<sup>_N_</sup>；

当 _k_ = _N_ 时，_B_(_N_, _k_) = 2<sup>_N_</sup>-1；

当 _k_ < _N_ 时，（这是我们比较关心的情况）可将这 _B_(_N_, _k_) 个 Dichotomies 分为两类：
- 在 _N_-1 个数据中是成对出现的，它们在第 _N_ 个数据上的判断分别是 -1 和 +1：2&alpha;
- 非成对出现的：&beta;

有 _B_(_N_, _k_) = 2&alpha;+&beta;

假设我们去掉第 _N_ 个数据，再合并那些重复的 &alpha; 个 Dichotomies，在这 _N_-1 个数据中就有 &alpha;+&beta; 个 Dichotomies。

根据 _B_(_N_, _k_) 的定义，_N_ 个数据中的任何 _k_ 个数据都不能 Shatter；又在 _k_ < _N_ 的前提下，有 _k_ &le; _N_-1；所以在这 _N_-1 个数据中的任何 _k_ 也不能 Shatter（否则 Break Point 就不是 _k_）。

因此， &alpha;+&beta; &le; _B_(_N_-1, _k_)。

类似的，如果只看成对出现的部分（ &alpha; ），这 _N_-1 个数据也不能被 _k_-1 Shatter，否则加上 Shatter 的第 _N_ 个数据就会 Shatter 了。

因此， &alpha; &le; _B_(_N_-1, _k_-1)。

综上，有 _B_(_N_, _k_) &le; _B_(_N_-1, _k_) + _B_(_N_-1, _k_-1)

组合中有一个类似的定理：

<img src="http://latex.codecogs.com/svg.latex?C_{N}^{\,i+1}=C_{N-1}^{\,i}+C_{N-1}^{\,i+1}"/>

从 _N_ 个里选 _i_+1 个，等于从 _N_-1 个里选 _i_ 个（再选第 _N_ 个），加上从 _N_-1 个里选 _i_+1 个（不选第 _N_ 个）。

使用上面的公式和数学归纳法可以证明：

<img src="http://latex.codecogs.com/svg.latex?B(N,k)\,\leq\,\sum_{i=0}^{k-1}C_{N}^{\,i}"/>

当 _N_ = 1, _k_ = 1 的时候， _B_(1, 1) = 1，公式成立；
当 _N_ = 1, _k_ &ge; 2 的时候， _B_(1, _k_) = 2，公式成立；

如果对 _N_-1 时公式成立，对于 _N_ 时，有

<img src="http://latex.codecogs.com/svg.latex?\begin{align*}B(N,k)&\,\leq\,B(N-1,k)\,&\,+&\,B(N-1,k-1)\\&&&\\&\,\leq\,\sum_{i=0}^{k-1}C_{N-1}^{\,i}\,&\,+&\,\sum_{i=0}^{k-2}C_{N-1}^{\,i}\\&&&\\&\,\leq\,1+\sum_{i=1}^{k-1}C_{N-1}^{\,i}\,&\,+&\,\sum_{i=1}^{k-1}C_{N-1}^{\,i-1}\\&&&\\&\,\leq\,1+\sum_{i=1}^{k-1}[\,C_{N-1}^{\,i}\,&\,+&\,C_{N-1}^{\,i-1}\,]\\&&&\\&\,\leq\,1+\sum_{i=1}^{k-1}C_{N}^{\,i}&&\\&&&\\&\,\leq\,\sum_{i=0}^{k-1}C_{N}^{\,i}&&\\\end{align*}"/>

所以，不等式成立！这里的最高项是 _N_<sup> _k_-1</sup>！

因此`成长函数`的`上限函数`的**上限**是多项式的，而不是指数形式的！

所以，只要 **有 Break Points**，就可以机器学习！

---

其实，这个不等式是可以证明只有等号是成立的，证明的角度是证明**大于等于**也成立，似乎是用以下思路（ 2&alpha;+&beta; ?）：

<img src="http://latex.codecogs.com/svg.latex?B(N,k)\,\geq\,2B(N-1,k-1)+(B(N-1,k)-B(N-1,j-1))"/>

不过我不会证明，也没找到资料（T_T），向会玩的同学们求助！请在 [Issues #1](https://github.com/chAwater/MachineLearningFoundations/issues/1) 中回答。

---

不过，我们还剩下一个问题没有解决：
- 能否用成长函数来代替 _M_

答案是可以的，不过会在之前的 Hoeffding 不等式会增加一些“无所谓”的常数项：

<img src="http://latex.codecogs.com/svg.latex?\mathbb{P}[\exists\,h\in\mathcal{H}\,s.t.\,|E_{in}(h)-E_{out}(h)|>\epsilon]\,\leq\,4\,m_{\mathcal{H}}(2N)\,\mathrm{exp}\,(-\frac{1}{8}\epsilon^2N)"/>

对于不等式 **左边** 是在`函数集合`中存在一个 _h_ 使得 <i>E</i><sub>in</sub> (<i>h</i>) 和 <i>E</i><sub>out</sub> (<i>h</i>) 的差距很大的概率，我们希望这个概率很小，这样就可以机器学习。

这里的数学过程比较复杂、对多数人来说太细节了，可以 **不必深究**，不过推倒过程中也稍微回顾了一下前面的知识，可以有个大概的整体理解。

#### 替换 <i>E</i><sub>out</sub>

不等式 **左边** 很难处理，因为其中的 <i>E</i><sub>out</sub> 是无限多的，因此我们需要用一个有限的东西代替  <i>E</i><sub>out</sub> 。

怎么代替呢？用验证（Verification）！

对于一个 _h_，可以用一些数据（ <img src="http://latex.codecogs.com/svg.latex?\mathcal{D}^{'}"/> ）来得到 <i>E</i><sub>in</sub><sup><i>'</i></sup> 从而估计 <i>E</i><sub>out</sub> 。如果 <i>E</i><sub>in</sub> 和 <i>E</i><sub>out</sub> 的差距很大，那么假设再进行一次抽样（Sample）的时候，很大概率下 <i>E</i><sub>in</sub><sup><i>'</i></sup> 和 <i>E</i><sub>in</sub> 也会差距很大。类似于之前投硬币的问题，那个五次正面的人如果再投5次硬币，其结果会和之前的五次差距很大。

这个描述，就把无限的 <i>E</i><sub>out</sub> 转换成了有限的、类似于 <i>E</i><sub>in</sub> 的 <i>E</i><sub>in</sub><sup><i>'</i></sup> 。不过其概率和“差距”的量会稍有些变化：

<img src="http://latex.codecogs.com/svg.latex?\small{\mathbb{P}[\exists\,h\in\mathcal{H}\,s.t.\,|E_{in}(h)-E_{out}(h)|>\epsilon]\,\leq\,2\,\mathbb{P}[\exists\,h\in\mathcal{H}\,s.t.\,|E_{in}(h)-E_{in}^{'}(h)|>\frac{\epsilon}{2}]}"/>

这里，假设发生的验证（Verification）所用到的数据叫做“Ghost data”。

#### 替换`函数集合`

另外一个无限多的项是`函数集合`中的 _h_，不过，现在我们公式中的 <i>E</i><sub>in</sub> 和 <i>E</i><sub>in</sub><sup><i>'</i></sup> 都是发生在有限多的数据上了，因此，可以用 Effective Number 来代替无限多的 _h_。这就是我们引入 **Dichotomy**、`成长函数` 和 `上限函数`的时候！对于 <i>E</i><sub>in</sub> 和 <i>E</i><sub>in</sub><sup><i>'</i></sup> 总共有 2 _N_ 个数据，因此最多有 <img src="http://latex.codecogs.com/svg.latex?m_{\mathcal{H}}(2N)"/> 种 _h_，所以有：

<img src="http://latex.codecogs.com/svg.latex?\small{2\,\mathbb{P}[\exists\,h\in\mathcal{H}\,s.t.\,|E_{in}(h)-E_{in}^{'}(h)|>\frac{\epsilon}{2}]\,\leq\,2m_{\mathcal{H}}(2N)\,\cdot\,\mathbb{P}[\mathrm{fixed}\,h\,s.t.\,|E_{in}(h)-E_{in}^{'}(h)|>\frac{\epsilon}{2}]}"/>

#### 运用 Hoeffding (without replacement)

想象有一个罐子里面有 2_N_ 个小球，抓出 _N_ 个，考虑这 _N_ 个小球和所有小球的差别，这就可以使用 Hoeffding 。

<img src="http://latex.codecogs.com/svg.latex?|E_{in}-E_{in}^{'}|>\frac{\epsilon}{2}\iff|E_{in}-\frac{E_{in}+E_{in}^{'}}{2}|>\frac{\epsilon}{4}"/>

<img src="http://latex.codecogs.com/svg.latex?2m_{\mathcal{H}}(2N)\,\mathbb{P}[\mathrm{fixed}\,h\,s.t.\,|E_{in}(h)-E_{in}^{'}(h)>\frac{\epsilon}{2}|]\leq2m_{\mathcal{H}}(2N)\,\cdot\,2\,\mathrm{exp}\left(-2\left(\frac{\epsilon}{4}\right)^2N\right)"/>

#### VC bound

整理一下公式得到：

<img src="http://latex.codecogs.com/svg.latex?\mathbb{P}[\exists\,h\in\mathcal{H}\,s.t.\,|E_{in}(h)-E_{out}(h)|>\epsilon]\,\leq\,4\,m_{\mathcal{H}}(2N)\,\mathrm{exp}\,(-\frac{1}{8}\epsilon^2N)"/>

- 通过 `Verification` 替换无限的 <i>E</i><sub>out</sub>
- 通过 `上限函数` 替换无限的`函数集合`
- 通过 `Hoeffding` 描述概率

这个公式叫做 **Vapnik-Chervonenkis (VC) bound**，描述了的“坏事情”发生概率的上限。

因此，当 Break Point 存在的时候，只要 _N_ 足够大，机器学习就是可能的！

具体来说，我们已经证明了：对于 2D Perceptrons，因为 Break Point 是 4，所以机器学习是可以完成的！

思考：对于其他的机器学习问题，如何使用 Break Point？

####### 数学证明二维空间中的线性分类器是可以通过机器学习完成的！！ #######

---
---
---

## Lecture 7: VC Dimension

——

### Definition

上面我们已经证明了：

<img src="http://latex.codecogs.com/svg.latex?m_{\mathcal{H}}(N)\,\leq\,B(N,\,k)=\sum_{i=0}^{k-1}C_{N}^{\,i}\,\leq\,N^{\,k-1}"/>

一张图总结一下：

![](./Snapshot/Snap08.png)

---

我们给最大的、非 Break Point 的 输入叫做 **VC Dimension**，，标注为 _d_<sub>VC</sub> = _k_-1，它是一个`函数集合`的性质。

如果 _N_(_k_) >  _d_<sub>VC</sub> ，则 _N_(_k_) 就是 Break Point。

VC Dimension 和下面这些都没有关系：
- 机器学习算法
- 输入数据的分布
- 目标函数

因此，在 VC Dimension 是有限的时候，我们无论如何都可以确保 <i>E</i><sub>in</sub> 和 <i>E</i><sub>out</sub> 是接近的。

---

### _d_<sub>VC</sub> for Perceptrons

我们上一章讨论的 2D Perceptrons 因为 _d_<sub>VC</sub> = 3 (Break Point _k_ = 4 )，所以可以学习。那么在更高维度的 Perceptrons 时怎么办呢？

我们通过观察 1D 和 2D Perceptrons 发现 对于 _d_-D Perceptrons 有可能 _d_<sub>VC</sub> = _d_+1

下面我们就从两个角度来证明：
1. _d_<sub>VC</sub> &geq; _d_+1
2. _d_<sub>VC</sub> &leq; _d_+1

- 为了证明 (1)，我们需要找到一组数据，有 _d_+1 个输入，并且能够 Shatter：

![](./Snapshot/Snap09.png)

- 为了证明 (2)，我们需要证明任何 _d_+2 个输入，都不能 Shatter：

![](./Snapshot/Snap10.png)

---

### _d_<sub>VC</sub> 的 `物理意义`

那么 VC Dimension 为什么要叫 "Dimension" 呢？

上面我们已经证明了，VC Dimension 和 Perceptrons 的维度有很密切的关系，可以把 Perceptrons 的 **w** 就当成是这个（假设）函数集合的自由度。

类似的，VC Dimension 就表示了这个`函数集合`的**自由度**，衡量这个`函数集合`能够产生多少 Dichotomies 。

- 如果 _d_<sub>VC</sub> 很小，那么“坏事情”发生的概率很小，但是函数集合可能只有很少的选择；
- 如果 _d_<sub>VC</sub> 很小，“坏事情”发生的概率会变大，但是函数集合中有很多的选择；

---

### _d_<sub>VC</sub> 对于`机器学习`的意义

#### VC Bound

<img src="http://latex.codecogs.com/svg.latex?\mathbb{P}_{\mathcal{D}}[|E_{in}(g)-E_{out}(g)|>\epsilon]\,\leq\,4\,(2N)^{d_{VC}}\,\mathrm{exp}\,(-\frac{1}{8}\epsilon^2N)"/>

不等式左边是“坏事情”发生的概率，如果我们把不等式右边作为 &delta; ，那么“好事情”发生的概率就是 1-&delta;，因此有：

<img src="http://latex.codecogs.com/svg.latex?\begin{align*}4\,(2N)^{d_{VC}}\,\mathrm{exp}\,(-\frac{1}{8}\epsilon^2N)&=\delta\\\epsilon&=\sqrt{\frac{8}{N}\,\mathrm{ln}\left(\frac{4(2N)^{d_{VC}}}{x}\right)}&\end{align*}"/>

等式右边的这个项叫做 (Penalty for) Model Complexity，<img src="http://latex.codecogs.com/svg.latex?\mathbf{\Omega}\,(N,\mathcal{H},\delta)"/>

VC Bound 就告诉我们，有很大的概率 <i>E</i><sub>out</sub> &leq; <i>E</i><sub>in</sub> + &Omega;

所以，就有下面这个在机器学习中非常常见的一张图：

![](./Snapshot/Snap11.png)

#### Sample Complexity

类似的，VC Bound 的这个公式将 &epsilon; , &delta; , _d_<sub>VC</sub> 和 _N_ 联系起来，因此，对于一个机器学习问题，我们就可以根据我们对其准确度的要求（ &epsilon; , &delta; ）和模型的复杂度（ _d_<sub>VC</sub> ）计算出我们对数据集大小的要求（ _N_ ）。

通常情况下，理论上需要的 _N_ &asymp; 10,000 _d_<sub>VC</sub>

不过实际上一般只需要 _N_ &asymp; 10 _d_<sub>VC</sub>

这是因为我们在 VC Bound 推导的过程中使用了很多非常“宽松”的替换：
- Hoeffding for <i>E</i><sub>out</sub>
  - 包含了任意的数据分布
  - 包含了任何的目标函数
- 使用了`成长函数`，而非 Dichotomies
  - 任何数据都符合
- 使用`上限函数`
  - 上限的上限，非常宽松
  - 只需要考虑 _d_<sub>VC</sub> 而不需要考虑`函数集合`的其他细节
- 使用 `Union Bound`
  - 考虑了最坏的情况（可能重叠发生事件的概率被独立发生的概率累加）
  - 机器学习算法可以随意选择

---
---
---

## Lecture 8:










---
