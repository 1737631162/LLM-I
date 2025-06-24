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


##  目前主流的attention方法
- MHA（多头注意力）
建模不同子空间的信息，从而均衡同一种注意力机制可能产生的偏差，让词义拥有来自更多元的表达

- MQA（多查询注意力）
MQA 让所有的头之间共享同一份 Key 和 Value 矩阵，每个头正常的只单独保留了一份 Query 参数，从而大大减少 Key 和 Value 矩阵的参数量，但是会带来精度上的损失。

- GQA（分组查询注意力）
介于MHA和MQA之间  
Query 被分成多个组，每组内的多个查询头共享一个 Key 和 Value 向量。

## FlashAttention
核心思想：尽可能减少冗余计算和访问内存。  
FlashAttention V1：采用了一种基于块（block）的矩阵乘法策略，将输入分割成多个小块，然后分块进行计算。这种方法减少了同时需要加载到高速缓存或内存中的数据量，从而有效地降低了内存带宽需求，并提高了计算效率。  
FlashAttention V2：引入并行扫描算法和硬件感知优化，实现接近线性时间的实际性能，特别适合处理超长序列。  

参考glm4的modeling_chatglm实现  
安装flash_attn（需要考虑torch版本及cuda驱动版本）  
```
try:
    from transformers.utils import is_flash_attn_greater_or_equal_2_10, is_flash_attn_2_available

    if is_flash_attn_2_available():
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
except:
    pass

if attention_mask is not None:
    attn_output_unpad = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_in_batch_q,
        max_seqlen_k=max_seqlen_in_batch_k,
        dropout_p=dropout,
        softmax_scale=None,
        causal=causal,
    )
else:
    attn_output = flash_attn_func(
        query_states, key_states, value_states, dropout, softmax_scale=None, causal=causal
    )
```
