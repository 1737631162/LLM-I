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
QK^T = (b, sql_len, d)@(b, d, sql_len)=(b, sql_len, sql_len)       这个点积需要做 d 次乘法 + d-1 次加法 ≈ d 次操作, 总计算量也就是 sql_len * sql_len * d 

多头：
QK^T = (b, sql_len, heads, d/heads)@(b, d/heads, heads, sql_len)=(b, sql_len, sql_len)       这个点积需要做 d/heads 次乘法 + d/heads-1 次加法 ≈ d/heads 次操作, 总计算量也就是 heads * (sql_len * sql_len * d/heads)

多头注意力的优势在于：
更好的建模不同子空间的信息。
可以并行计算，提升训练效率。
每个 head 的维度较小，减少了 attention matrix 的计算量（即 QK^T 的计算复杂度从 O(n²*d) 变成 O(n²*(d/h))）。
每个 head 的计算复杂度是 O(n²(d/h))，总共 h 个 head，并行执行。

##  目前主流的attention方法


## 为什么点积模型要做缩放
为什么要除以$\sqrt{d}$
