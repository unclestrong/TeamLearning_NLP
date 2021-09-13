# Attention

- 问题：Attention出现的原因是什么？
- 潜在的答案：基于循环神经网络（RNN）一类的seq2seq模型，在处理**长文本时遇到了挑战**，而对长文本中不同位置的信息进行**attention有助于提升RNN的模型效果**。

于是学习的问题就拆解为：

1. 什么是**seq2seq**模型？
2. 基于RNN的seq2seq模型如何处理**文本/长文本**序列？
3. seq2seq模型处理长文本序列时遇到了**什么问题**？
4. 基于RNN的seq2seq模型如何**结合attention**来改善模型效果？

## seq2seq

seq2seq是一种常见的NLP模型结构，全称是：sequence to sequence，翻译为“序列到序列”。顾名思义：从一个文本序列得到一个新的文本序列。典型的任务有：机器翻译任务，文本摘要任务。

## **seq2seq干了什么事情？**

seq2seq模型的

- **输入**可以是一个（单词、字母或者图像特征）序列

- **输出**是另外一个（单词、字母或者图像特征）序列

一个训练好的seq2seq模型如下图所示：

![seq2seq](./pictures/1-seq2seq.gif)

如下图所示，以NLP中的机器翻译任务为例，序列指的是一连串的单词，输出也是一连串单词。

![translation](./pictures/1-2-translation.gif)

### seq2seq细节

将上图中**蓝色**的seq2seq模型进行**拆解**，如下图所示：seq2seq模型由==编码器（Encoder）==和==解码器（Decoder）==组成。

![encoder-decode](./pictures/1-3-encoder-decoder.gif)

**绿色的编码器**会处理输入序列中的每个元素并获得输入信息，这些信息会被转换成为一个黄色的向量（称为==context向量==）。

当我们处理完整个输入序列后，编码器把 context向量 发送给**紫色的解码器**，解码器通过context向量中的信息，逐个元素输出新的序列。



由于seq2seq模型可以用来解决**机器翻译**任务，因此机器翻译被任务seq2seq模型解决过程如下图所示，当作seq2seq模型的一个具体例子来学习。

![encoder-decoder](./pictures/1-3-mt.gif)



深入学习机器翻译任务中的seq2seq模型，如下图所示。seq2seq模型中的编码器和解码器一般采用的是循环神经网络RNN（Transformer模型还没出现的过去时代）。编码器将输入的法语单词序列编码成context向量（在绿色encoder和紫色decoder中间出现），然后解码器根据context向量解码出英语单词序列。

![context向量对应图里中间一个浮点数向量。在下文中，我们会可视化这些向量，使用更明亮的色彩来表示更高的值，如上图右边所示](./pictures/1-4-context-example.png)

图：context向量对应上图中间**浮点数向量**。在下文中，我们会可视化这些数字向量，使用更明亮的色彩来表示更高的值，如上图右边所示



如上图所示，我们来看一下黄色的context向量是什么？

本质上是一组浮点数。而这个**context的数组长度是基于编码器RNN的隐藏层神经元数量**的。上图展示了长度为4的context向量，但在实际应用中，**context向量的长度是自定义的**，比如可能是256，512或者1024。



## 那么RNN是如何具体地处理输入序列的呢？

1. 假设序列输入是一个**句子**，这个句子可以由$n$个**词**表示：$sentence = \{w_1, w_2,...,w_n\}$。
2. RNN首先将句子中的每一个**词**映射成为一个向量得到一个**向量序列**：$X = \{x_1, x_2,...,x_n\}$，每个单词映射得到的向量通常又叫做：==word embedding==。
3. 然后在处理第$t \in [1,n]$个**时间步**的序列输入$x_t$时，RNN网络的输入和输出可以表示为：$h_{t} = RNN(x_t, h_{t-1})$

   - 输入：RNN在**时间步**$t$的输入之一为单词$w_t$经过映射得到的向量$x_t$。
   - 输入：RNN另一个输入为**上一个时间步**$t-1$得到的hidden state向量$h_{t-1}$，同样是一个向量。
   - 输出：RNN在时间步$t$的输出为$h_t$ ==hidden state向量==。

![我们在处理单词之前，需要把他们转换为向量。这个转换是使用 word embedding 算法来完成的。我们可以使用预训练好的 embeddings，或者在我们的数据集上训练自己的 embedding。通常 embedding 向量大小是 200 或者 300，为了简单起见，我们这里展示的向量长度是4](./pictures/1-5-word-vector.png)

图：word embedding例子。我们在处理单词之前，需要将单词映射成为向量，通常使用 word embedding 算法来完成。一般来说，我们可以使用提前训练好的 word embeddings，或者在自有的数据集上训练word embedding。为了简单起见，上图展示的word embedding维度是4。上图左边每个单词经过word embedding算法之后得到中间一个对应的4维的向量。



进一步可视化一下基于RNN的seq2seq模型中的编码器在第1个时间步是如何工作：

![rnn](./pictures/1-6-rnn.gif)

动态图：如图所示，RNN在第2个时间步，采用第1个时间步得到hidden state#10（隐藏层状态）和第2个时间步的输入向量input#1，来得到新的输出hidden state#1。



看下面的动态图，让我们详细观察一下编码器如何在**每个时间步**得到hidden sate，并将最终的hidden state传输给解码器，解码器根据编码器所给予的最后一个hidden state信息解码处输出序列。注意，**最后一个 hidden state实际上是我们上文提到的context向量**。

![](./pictures/1-6-seq2seq.gif)

动态图：编码器逐步得到hidden state并传输最后一个hidden state给解码器。



接着，结合编码器处理输入序列，一起来看下**解码器如何一步步得到输出序列的**。

与编码器类似，解码器在每个时间步也会得到 hidden state（隐藏层状态），而且也需要把 hidden state（隐藏层状态）从一个时间步传递到下一个时间步。

![](./pictures/1-6-seq2seq-decoder.gif)

动态图：**编码器**首先**按照时间步依次编码**每个法语单词，最终将最后一个hidden state也就是context向量传递给解码器，**解码器**根据**context向量逐步解码**得到英文输出。

## seq2seq模型处理文本序列（特别是长文本序列）时会遇到什么问题？

基于RNN的seq2seq模型编码器**所有信息都编码到了一个context向量中**，便是这类模型的瓶颈。

- 一方面**单个向量**很**难包含所有文本序列**的信息
- 另一方面RNN递归地编码文本序列使得模型在处理**长文本时面临非常大的挑战**（比如RNN处理到第500个单词的时候，很难再包含1-499个单词中的所有信息了）。

## 基于RNN的seq2seq模型如何结合attention来改善模型效果？

面对以上问题，Bahdanau等2014发布的[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) 和 Luong等2015年发布的[Effective Approaches to Attention-based Neural Machine Translation
](https://arxiv.org/abs/1508.04025)两篇论文中，提出了一种叫做注意力**attetion**的技术。通过attention技术，seq2seq模型极大地提高了机器翻译的质量。

归其原因是：attention注意力机制，使得seq2seq模型可以**有区分度、有重点地关注输入序列**。

下图依旧是机器翻译的例子：

![在第7个时间步，注意力机制使得解码器在产生英语翻译之前，可以将注意力集中在 "student" 这个词（在法语里，是 "student" 的意思）。这种从输入序列放大相关信号的能力，使得注意力模型，比没有注意力的模型，产生更好的结果。](./pictures/1-7-attetion.png)

图：在第 7 个时间步，注意力机制使得解码器在产生英语翻译student英文翻译之前，可以将注意力集中在法语输入序列的：étudiant。这种有区分度得attention到输入序列的重要信息，使得模型有更好的效果。



让我们继续来理解带**有注意力的seq2seq模型**：一个注意力模型与经典的seq2seq模型主要有**2点不同**：

1. 首先，编码器会把**更多的数据**传递给解码器。编码器把所有时间步的 hidden state（隐藏层状态）传递给解码器，而**不是只传递最后一个 hidden state**（隐藏层状态），如下面的动态图所示:

   ![](./pictures/1-6-mt-1.gif)

2. 注意力模型的**解码器在产生输出之前**，做了一个额外的**attention处理**。如下图所示，具体为：

   1. 由于编码器中每个 hidden state（隐藏层状态）都对应到输入句子中一个单词，那么解码器要查看所有接收到的编码器的 hidden state（隐藏层状态）。

   2. 给每个 hidden state（隐藏层状态）**计算出一个分数**（我们先忽略这个分数的计算过程）。

   3. 所有hidden state（隐藏层状态）的分数经过**softmax进行归一化**。

   4. 将**每个 hidden state（隐藏层状态）乘以所对应的分数**，从而能够让**高分**对应的  hidden state（隐藏层状态）会被**放大**，而**低分**对应的  hidden state（隐藏层状态）会被**缩小**。

   5. 将所有hidden state根据对应分数进行**加权求和**，得到对应时间步的context向量。

      ![](./pictures/1-7-attention-dec.gif)



所以，**attention可以简单理解为：一种有==效的加权==求和技术，其艺术在于如何获得权重**。



现在，让我们把所有内容都融合到下面的图中，来看看**结合注意力的seq2seq模型解码器**全流程，动态图展示的是第4个时间步：

1. 注意力模型的解码器 RNN 的输入包括：一个word embedding 向量，和一个初始化好的解码器 hidden state，图中是$h_{init}$。
2. RNN 处理上述的 2 个输入，产生一个输出和一个新的 hidden state，图中为h4。
3. 注意力的步骤：我们使用编码器的所有 hidden state向量和 h4 向量来计算这个时间步的context向量（C4）。
4. 我们把 **h4 和 C4 拼接起来**，得到一个橙色向量。
5. 我们把这个橙色向量输入一个前馈神经网络（这个网络是和整个模型一起训练的）。
6. 根据前馈神经网络的输出向量得到输出单词：假设输出序列可能的单词有N个，那么这个前馈神经网络的输出向量通常是N维的，每个维度的下标对应一个输出单词，每个维度的数值对应的是该单词的输出概率。
7. 在下一个时间步重复1-6步骤。

![](./pictures/1-7-attention-pro.gif)

