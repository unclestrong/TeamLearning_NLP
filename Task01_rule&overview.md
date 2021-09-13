# 环境配置

## 下载本项目所有文件

```bash
git clone https://github.com/datawhalechina/learn-nlp-with-transformers.git
```

![image-20210913085048267](https://gitee.com/unclestrong/deep-learning21_note/raw/master/imgbed/image-20210913085048267.png)

## 配置本项目本地运行环境(Win,Anaconda)

```bash
pip install -r requirements.txt
```

# Transformers在NLP中的兴起

## 常见的NLP任务

1. **文本分类**：对单个、两个或者多段文本进行分类。

   举例：“这个教程真棒！”这段文本的情感倾向是正向的，“我在学习transformer”和“如何学习transformer”这两段文本是相似的。

2. **序列标注**：对文本序列中的token、字或者词进行分类。

   举例：“我在**国家图书馆**学transformer。”这段文本中的**国家图书馆**是一个**地点**，可以被标注出来方便机器对文本的理解。

3. **问答任务**——抽取式问答和多选问答：

   - 抽取式问答根据**问题**从一段给定的文本中找到**答案**，**答案必须是给定文本的一小段文字**。举例：问题“小学要读多久?”和一段文本“小学教育一般是六年制。”，则答案是“六年”。
   - 多选式问答，从多个选项中选出一个正确答案。举例：“以下哪个模型结构在问答中效果最好？“和4个选项”A、MLP，B、cnn，C、lstm，D、transformer“，则答案选项是D。

4. **生成任务**——语言模型、机器翻译和摘要生成：

   - 根据已有的一段文字生成（generate）一个字通常叫做语言模型
   - 根据一大段文字生成一小段总结性文字通常叫做摘要生成
   - 将源语言比如中文句子翻译成目标语言比如英语通常叫做机器翻译。

## Transformer的兴起

- 2017 [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)论文首次提出了**Transformer**模型结构并在机器翻译任务上取得了The State of the Art(SOTA, 最好)的效果。
- 2018 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)使用Transformer模型结构进行大规模语言模型（language model）预训练（Pre-train），再在多个NLP下游（downstream）任务中进行微调（Finetune），一举刷新了各大NLP任务的榜单最高分，轰动一时。
- 2019-2021 研究人员将Transformer这种模型结构和预训练+微调这种训练方式相结合，提出了一系列Transformer模型结构、训练方式的改进（比如transformer-xl，XLnet，Roberta等等）。如下图所示，各类Transformer的改进不断涌现。

![](https://datawhalechina.github.io/learn-nlp-with-transformers/%E7%AF%87%E7%AB%A01-%E5%89%8D%E8%A8%80/pictures/1-x-formers.png)

图：各类Transformer改进，来源：[A Survey of Transformers](https://arxiv.org/pdf/2106.04554.pdf)



另外，由于Transformer优异的模型结构，使得其**参数量**可以**非常庞大**从而容纳更多的信息，因此Transformer模型的能力随着预训练不断提升，随着近几年计算能力的提升，越来越大的预训练模型以及效果越来越好的Transformers不断涌现，简单的统计可以从下图看出：

![](https://datawhalechina.github.io/learn-nlp-with-transformers/%E7%AF%87%E7%AB%A01-%E5%89%8D%E8%A8%80/pictures/2-model_parameters.png)

图：预训练模型参数不断变大,来源[Huggingface](https://huggingface.co/course/chapter1/4?fw=pt)

