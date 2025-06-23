## 对attention的理解
核心思想：在处理序列时，应该关注输入中重要的部分，通过学习不同部分的权重，将输入序列中重要的进行加权，从而达到这个目的。

## 注意力计算步骤
传统注意力
$$
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

## attention和全连接的区别
全连接没有qkv的概念，当输入为v的时候，输出就是v的加权。attention中使用query作为锚点，注意力分数就是计算锚点的距离得到的。  
例：我们要从图书馆里找一本书，你只记得大概的内容，然后每本书都要花时间去浏览，这就是全连接，而attention则是告诉你一些关键信息，例如封面的颜色，书名里包含“AI”关键词。

## transformer中multi-head attention中每个head为什么要进行降维？
|类型	|每次 QKᵀ 计算量	|并行方式|
|---|------|---|
|单头|	512 × 512 × 768	|串行或单块	|
|多头|12 × (512 × 512 × 64)	|并行	|

单头：
QK^T = (b, sql_len, d)@(b, d, sql_len)=(b, sql_len, sql_len)       这个点积需要做 d 次乘法 + d-1 次加法 ≈ d 次操作, 总计算量也就是 b * (sql_len * sql_len * d)

多头：
QK^T = (b, heads, sql_len, d/heads)@(b, heads, d/heads, sql_len)=(b, heads, sql_len, sql_len)       这个点积需要做 d/heads 次乘法 + d/heads-1 次加法 ≈ d/heads 次操作, 总计算量也就是 b * (heads * (sql_len * sql_len * d/heads))

多头注意力的优势在于：
更好的建模不同子空间的信息。
可以并行计算，提升训练效率。
每个 head 的维度较小，减少了 attention matrix 的计算量（即 QK^T 的计算复杂度从 O(b*n²*d) 变成 O(b*n²*(d/h))）。
每个 head 的计算复杂度是 O(b*n²(d/h))，总共 h 个 head，并行执行。

##  目前主流的attention方法


## 为什么点积模型要做缩放
为什么要除以$$\sqrt{d}$$  
当d的值变大的时候，会造成梯度消失  
以上引出两个问题：  
1.为什么会造成梯度消失？  
2.为什么是$\sqrt{d}$，可以用其他值吗？  
answer1：当d变大的时候，QK^T的方差会变大，方差变大会导致元素之间的差值变大，从而导致softmax退化为argmax，当只有一个值为1的元素，其他都为0的话，反向传播的梯度会变为0，也就是梯度消失产生。  
answer2：为了让输入到 softmax 前的数值保持一个单位方差的标准分布（服从均值为0方差为1的标准分布），避免因维度增长而导致的饱和（函数输出接近其极限值0或1）。  
为什么要让其服从标准分布  
| 原因 | 影响 |
|------|------|
| 防止 softmax 输出过于尖锐 | 避免梯度消失 |
| 利用激活函数的良好梯度区域 | 加快训练 |
| 匹配参数初始化策略 | 提高模型稳定性 |
| 与归一化层兼容 | 更好的泛化能力 |
  
假设分布满足  
$$  
x \sim \mathcal{N}(0, 1)  
$$  
那么  
$$ Var(x) = \mathbb{E}[x^2] - (\mathbb{E}[x])^2 = \mathbb{E}[x^2] \quad \text{（因为 $\mathbb{E}[x] = 0$）} $$
设要除以的值为x，为了满足Var(QK/x)=1， 有  
$$  
\frac{1}{x^2} Var(QK)=\frac{1}{x^2} * \sum_{i = 1}^{d_k} Var[{Q_i}]Var[{K_i}] = \frac{1}{x^2} * {d_k}   
$$  
要使得上式趋近于1，那么  
$$  
x=\frac{1}{\sqrt{d}}  
$$  
