# MachineLearningFoundations
Notebooks for Machine Learning Foundations by @hsuantien

---

## Coursera Links

- [机器学习基石上 (Machine Learning Foundations)-Mathematical Foundations](https://www.coursera.org/learn/ntumlone-mathematicalfoundations)
- [机器学习基石下 (Machine Learning Foundations)-Algorithmic Foundations](https://www.coursera.org/learn/ntumlone-algorithmicfoundations)

by [Hsuan-Tien Lin](https://www.csie.ntu.edu.tw/~htlin/)

## 前言介绍

《机器学习基石》是国立台湾大学资讯工程系的 **林轩田** 老师开设的课程（**中文授课**），旨在从基础的角度介绍机器学习，包括机器学习的**哲学**、关键**理论**和核心**技术**。
（如果从理论角度出发，需要深入掌握各种机器学习理论，花费大量时间，但却不实用；
而如果从技术角度出发，快速介绍多种机器学习方法，但无法清晰理解，难以选择和应用。）
从基础角度出发，既能保证学生能够了解机器学习的基本概念，同时对学生基础的要求最少，也能够保证课程不会太枯燥。

## Lecture 1: The Learning Problem

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
<img src="http://latex.codecogs.com/svg.latex?\mathcal{H=\left\{h_k\right\}};\,(g\in\mathcal{H})"/>

假设函数（Hypothesis <=> Skill）：
<img src="http://latex.codecogs.com/svg.latex?g:\mathcal{X}\to\mathcal{Y};\,(g\approx{f})"/>


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

### 感知机（Perceptron）

考虑一个简单的分类问题，是否给一个顾客办理信用卡。

假设每个顾客有一系列的特征（Feature），比如年薪、花费、债务等：
<img src="http://latex.codecogs.com/svg.latex?\mathbf{x}=(\mathrm{x}_1,\mathrm{x}_2,\cdots,\mathrm{x}_d)"/>

计算特征的加权求和作为分数：
<img src="http://latex.codecogs.com/svg.latex?\sum_{i=1}^d \mathrm{w}_i\mathrm{x}_1"/>

如果客户的得分高于某个分数（threshold），则办理信用卡；若低于某个分数，则不办理信用卡。
因此有：

<img src="http://latex.codecogs.com/svg.latex?h(\mathbf{x})=\textnormal{sign}\left\(\left\(\sum_{i=1}^d \mathrm{w}_i\mathrm{x}_1\right\) - \textnormal{threshold}\right\)"/>

这就是**感知机**。

简化一下这个公式：

<img src="http://latex.codecogs.com/svg.latex?h(\mathbf{x})=\textnormal{sign}\left\(\left\(\sum_{i=1}^d\mathrm{w}_i\mathrm{x}_1\right\) -\begin{matrix}\underbrace{-\textnormal{threshold}}\\\mathrm{w}_0\end{matrix}\cdot\begin{matrix}\underbrace{+1}\\\mathrm{x}_0\end{matrix}\right\)=\textnormal{sign}\left\(\sum_{i=0}^d\mathrm{w}_i\mathrm{x}_1\right\)=\textnormal{sign}\left\(\mathbf{w}^T\mathbf{x}\right\)"/>

每一种`权重`向量（<img src="http://latex.codecogs.com/svg.latex?\mathbf{w}"/>）就是一个假设函数（Hypothesis）_h_。




因此，感知机也叫**线性分类器（linear/binary classifiers）**

### Perceptron Learning Algorithm (PLA)



---
