## FreTS

> title:     **Frequency-domain MLPs are More Effective Learners in Time Series Forecasting**【频域MLP更有效】

### 引入

* 首先，说明了现阶段的时间序列模型的“两大阵营”：**Transformer系列以及MLP系列；**

  * 但是Transformer系列非常容易遇到计算瓶颈【注意力机制的计算太占用资源，速度慢】；

  * 而之前的MLP模型<strong style="color:#00b050;">难以掌控全局时间序列信息</strong>，也都存在<strong style="color:#00b050;">点式映射</strong>和<strong style="color:#00b050;">信息瓶颈</strong>问题，这在很大程度上阻碍了预测性能的提高

  * > <span style="background:#BBFFBB;">**点式映射（Pointwise Mapping）**</span>
    >
    > - **定义**：指的是在神经网络中，<span style="background:#FFFFBB;">将输入的每个元素映射到输出中的一个相应元素</span>。具体而言，这是指网络中的某些层（通常是全连接层或1x1卷积层）执行的逐元素操作，而<span style="background:#FFFFBB;">不是跨多个元素的操作。</span>
    > - **问题**：点式映射可能导致模型在学习时<strong style="color:#c00000;">过于注重输入中的每个细节</strong>，<strong style="color:#c00000;">而缺乏对整体结构和上下文的把握。这可能使模型更容易受到噪声的干扰，丧失对输入数据中潜在结构的抽象能力</strong>。
    >
    > <span style="background:#BBFFBB;">**信息瓶颈（Information Bottleneck）**</span>
    >
    > - **定义**：信息瓶颈理论认为，在学习过程中，模型应该<strong style="color:#c00000;">保留对输入数据的关键信息，而丢弃冗余信息。</strong>这可以通过对网络的某些部分引入约束来实现，以限制信息的传递或压缩信息表示。
    > - **问题**：当信息瓶颈设置得太紧，模型可能会失去对有用信息的捕捉，导致欠拟合。反之，如果信息瓶颈不足，模型可能过度关注冗余信息，导致过拟合。

* 本文提出了:

  * 研究了<strong style="color:#c00000;">频域mlp的学习模式，</strong>并发现它们的两个固有特征有利于预测，
    * (i)**全局视图**:频谱使mlp拥有信号的完整视图，更容易学习<span style="background:#FFFFBB;">全局依赖关系</span>，以及
    * (ii)**能量压缩**:频域mlp集中在<span style="background:#FFFFBB;">频率成分的较小关键部分，信号能量紧凑</span>。
  * 提出了<strong style="color:#c00000;">FreTS</strong>，这是一种**基于频域mlp的简单而有效的结构**，用于时间序列预测。FreTS主要包括两个阶段:
    * (i)**域转换**，将<span style="background:#FFFFBB;">时域信号转换为频域复数信号</span>
    * (ii)**频率学习**，执行我们<span style="background:#FFFFBB;">重新设计的mlp来学习频率分量的实部和虚部</span>。
    * <strong style="color:#c00000;">FreTS 的核心思想是学习频域中的时间序列预测映射。</strong>

### 理论支撑

* 通过实验，我们发现**频域 MLP 比时域 MLP 能捕捉到更明显的全局周期性模式。**
  * <img src="C:\Users\yjc\AppData\Roaming\Typora\typora-user-images\image-20231201170219574.png" alt="image-20231201170219574" style="zoom:67%;" />
* （先前的）**基于 MLP 的网络在时间序列预测任务中的有效性**，并激发了本文中开发频域 MLP 的灵感。
  * ![image-20231201170940335](C:\Users\yjc\AppData\Roaming\Typora\typora-user-images\image-20231201170940335.png)

* **假设 H 是原始时间序列的表示，H 是频谱的相应频率成分，那么时间序列在时域中的能量等于其在频域中的表示能量**：

  * ![image-20231201175607817](C:\Users\yjc\AppData\Roaming\Typora\typora-user-images\image-20231201175607817.png)

  * > 该定理意味着，如果**时间序列的大部分能量都集中在少数频率成分中，那么只用这些成分就能准确地表示时间序列。因此，舍弃其他成分不会对信号的能量产生重大影响**

* **频域 MLP** 的操作可视为**时域的全局卷积**。

  * ![image-20231201180157610](C:\Users\yjc\AppData\Roaming\Typora\typora-user-images\image-20231201180157610.png)
  * 理解：
    * 全局卷积在时域中能够<strong style="color:#3EC1D3;">捕捉输入序列的全局信息</strong>，而不仅仅是局部结构。这有助于模型<strong style="color:#3EC1D3;">理解整体的时间关系。</strong>
    * <strong style="color:#c00000;">频域操作对于处理时序信号时的特殊优势，尤其是在需要捕捉全局时间关系的情况下</strong>。在某些应用中，频域的表示可能更适合捕捉信号的周期性或频率成分，从而提高模型的性能
    * <span style="background:#ffbbff;">说白了：转化为频域后，事半功倍了</span>

### 模型架构![image-20231201180448773](C:\Users\yjc\AppData\Roaming\Typora\typora-user-images\image-20231201180448773.png)

### 实验分析

#### Frequency Channel and Temporal Learners

> <span style="background:#FFDBBB;">相当于对CL和TL做消融实验;    即 FreTS = FreCL + FreTL</span>

![image-20231201180654902](C:\Users\yjc\AppData\Roaming\Typora\typora-user-images\image-20231201180654902.png)

#### FreMLP vs. MLP

我们使用 FreMLP 取代现有基于 SOTA MLP 模型（即 DLinear 和 NLinear [37]）中的原始 MLP 组件，并在相同的实验设置下比较它们与原始 DLinear 和 NLinear 的性能。

<span style="background:#FFDBBB;">结果再次证实了 FreMLP 与 MLP 相比的有效性</span>

![image-20231201180745506](C:\Users\yjc\AppData\Roaming\Typora\typora-user-images\image-20231201180745506.png)

#### Efficiency Analysis

* 我们提出的 FreTS 的复杂度为 O(N log N + L log L)。
* 下图1比较了：**数据量对参数量以及训练时间的关系**；下图2比较了**预测长度对参数量以及训练时间的关系**
* ![image-20231201180821372](C:\Users\yjc\AppData\Roaming\Typora\typora-user-images\image-20231201180821372.png)

#### 可视化分析

从图中可以看出，实部和虚部在学习过程中都起着至关重要的作用：实部或虚部的权重系数表现出能量聚集特征（清晰的对角线模式），这有助于学习重要的特征。

<img src="C:\Users\yjc\AppData\Roaming\Typora\typora-user-images\image-20231201180953428.png" alt="image-20231201180953428" style="zoom:67%;" />

### 结论与小结

* 重新设计了<strong style="color:#c00000;">频率域的 MLP</strong>，它能通过<strong style="color:#c00000;">全局视角和能量压缩</strong>有效捕捉时间序列的基本模式。
* 我们通过一个<strong style="color:#c00000;">简单而有效的架构 FreTS 验证了这一设计</strong>，该架构建立在用于时间序列预测的频域 MLP 上。
* <strong style="color:#3EC1D3;">（展望</strong>）<strong style="color:#00b0f0;">简单的 MLP 具有多种优势，为现代深度学习奠定了基础，在以高效率实现令人满意的性能方面具有巨大潜力。</strong>

### 个人理解

> FreTS汲取了**FEDformer（频域下的注意力机制）**，也结合了**Dlinear、Nlinear（序列分解后使用MLP）**的特点，使得时序预测任务表现的准确而快速，让大众的目光不再盲目的放在Transformer上，可见理论创新的重要性。

### 复现过程

> <span style="background:#FFDBBB;">【将结果可视化以后，想和Informer等模型进行比较，但是发现和预测长度不相同】</span>

**解决方案：**看源码发现：Dataset_Custom_和Dataset_ETT_hour对数据集的划分并不相同，只需将下图的从Dataset_Custom_改为Dataset_ETT_hour即可。【对于其他数据集，采用相同的处理方案即可】

![image-20231201184647429](C:\Users\yjc\AppData\Roaming\Typora\typora-user-images\image-20231201184647429.png)

> <span style="background:#FFDBBB;">保证预测长度相同后，出现了数据的量纲问题</span>

**解决：**发现Informer以及其他项目对数据的归一化处理，与此项目不相同，将此项目的数据归一化改为Informer的处理方式即可，如下图

```
需要改动的地方：
utils/tools.py 中的 StandardScaler类
data_provider/data_loader.py 中的 Dataset_ETT_hour类
```

![image-20231201185102181](C:\Users\yjc\AppData\Roaming\Typora\typora-user-images\image-20231201185102181.png)

![image-20231201185114152](C:\Users\yjc\AppData\Roaming\Typora\typora-user-images\image-20231201185114152.png)

<span style="background:#FF9999;">注：如果使用其他数据集，并且涉及到模型之间的比较时，都要按照上述两个问题的方案去改</span>
