# tfr各个模块含义与实现
论文：[TF-Ranking: Scalable TensorFlow Library for Learning-to-Rank](https://dl.acm.org/doi/abs/10.1145/3292500.3330677)

结构图：
![image-20210524205903503](https://note.youdao.com/yws/api/personal/file/32DBAB8297FB4E399310B03294CAFE1F?method=download&shareKey=d242b3d84678dc6d0dd913f95f393ea0)



**训练部分代码结构示意xmind**

![image-20210524205903503](https://note.youdao.com/yws/api/personal/file/EF02719CFAF44085A1567D36805CCB7C?method=download&shareKey=8575991c55914db64c3e07d120debc0c)


>**问题：**<br>
>1.loss_fn中如何实现优化NDCG的目标?<br>
>2.group_size作用是什么?如何实现？<br>
>3.group_score_fn如何构建？<br>
>4.transform_fn对feature进行了什么变换? context和example feature如何在模型中生效？<br>5.优化器不能选择adam的原因是什么？可选优化器有哪些？<br>6.ranking head的作用是什么？<br>



## 1.loss_fn中如何实现优化NDCG的目标？

tfr优化NDCG的方法有两类：

1. 通过计算ndcg_lambda_weight ，增加权重的方式。原理为优化ndcg的上界函数。

2. 构造可导的ndcg近似函数，直接优化ndcg的近似函数

   

### 实现函数介绍：tfr.losses.make_loss_fn

>loss_keys：单一的loss或者loss list.也就是根据事先已定义好的loss function对应的name。
>
>return: 返回整体的loss_fn函数，可能由多个loss_key_fn结果加和构成

注：如果需要增加自己的`loss function`,可以在`key_to_fn`定义中增加 {  loss_key : ( loss_key_fn,kwargs)  } 调用`tfr.losses.make_loss_fn`或者直接重新定义`loss_fn`

 代码中，定义loss实例，采用loss.compute来计算梯度以及梯度回传。（self.compute结构相同，重点在于self.compute_unreduced_loss的差异，该函数的返回结果均为losses,loss_weights，但不同损失中返回结果表示的内容不同） 

| loss_keys                             | 备注                                         | 是否含lambda weight | gumbel 采样 |
| ------------------------------------- | -------------------------------------------- | ------------------- | ----------- |
| pairwise_hinge_loss                   | pairwise方法,通过增加lambda weight来优化ndcg | 1                   | 0           |
| pairwise_logistic_loss                | pairwise方法,通过增加lambda weight来优化ndcg | 1                   | 0           |
| pairwise_soft_zero_one_loss           | pairwise方法,通过增加lambda weight来优化ndcg | 1                   | 0           |
| softmax_loss                          |                                              | 1                   | 0           |
| unique_softmax_loss                   |                                              | 1                   | 0           |
| list_mle_loss                         |                                              | 1                   | 0           |
| sigmoid_cross_entropy_loss            |                                              | 0                   | 0           |
| mean_squared_loss                     |                                              | 0                   | 0           |
| approx_ndcg_loss                      | listwise方法，直接优化近似ndcg的函数         | 0                   | 0           |
| approx_mrr_loss                       | listwise方法，直接优化近似mrr的函数          | 0                   | 0           |
| neural_sort_cross_entropy_loss        |                                              | 0                   | 0           |
| gumbel_approx_ndcg_loss               | listwise方法，直接优化近似ndcg的函数         | 0                   | 1           |
| gumbel_neural_sort_cross_entropy_loss |                                              | 0                   | 1           |

gumbel 采样：https://en.wikipedia.org/wiki/Gumbel_distribution

使用lambda weight和不使用lambda weight的写法:https://github.com/tensorflow/ranking/issues/83







### 两种实现方法介绍

#### 1)以pairwise_logistic_loss为例说明如何通过增加lambda weight的方式去优化ndcg方法
论文：[The LambdaLoss Framework for Ranking Metric Optimization](https://dl.acm.org/doi/abs/10.1145/3269206.3271784) <br>
思想：通过优化NDCGcost的上界函数来达到优化NDCGcost的效果.

![image](https://note.youdao.com/yws/api/personal/file/39A13B792C2C40AFA1699BAB1AB47E5C?method=download&shareKey=b4bcd96c78ad62ddd99d91cbd1f4a1d2)


```python
#使用方式
loss_fn = tfr.losses.make_loss_fn(
            loss_key='pairwise_logistic_loss',
            lambda_weight=lambda_weight)
```



- lambda weight计算

  lambda_weight计算有以下三种，分别为计算ndcg,listmle,mrr:

  ```
  create_ndcg_lambda_weight(): Creates _LambdaWeight for NDCG metric
  create_reciprocal_rank_lambda_weight():Creates _LambdaWeight based on Position-Aware ListMLE paper.
  create_p_list_mle_lambda_weight():Creates _LambdaWeight for MRR-like metric
  ```
    以ndcg为例：
  ```python
  def create_ndcg_lambda_weight(topn=None, smooth_fraction=0.):
    """Creates _LambdaWeight for NDCG metric."""
    return losses_impl.DCGLambdaWeight(
        topn,
        gain_fn=lambda labels: tf.pow(2.0, labels) - 1.,
        rank_discount_fn=lambda rank: 1. / tf.math.log1p(rank),
        normalized=True,
        smooth_fraction=smooth_fraction)
  ```

- 将lambda weight加入loss中的方式

  1. pairwise_logistic_loss的函数为`_pairwise_logistic_loss`

     该function使用类`losses_impl.PairwiseLogisticLoss`,该类继承`_PairwiseLoss`，`_PairwiseLoss`继承`_RankingLoss`。

  ```python
  def _pairwise_logistic_loss(
      labels,
      logits,
      weights=None,
      lambda_weight=None,
      reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
      name=None):
    ##先定义PairwiseLogisticLoss类
    loss = losses_impl.PairwiseLogisticLoss(name, lambda_weight)  
    with tf.compat.v1.name_scope(loss.name, 'pairwise_logistic_loss',
                                 (labels, logits, weights)):
      return loss.compute(labels, logits, weights, reduction) #计算los
  ```


#### 2)以approx_ndcg_loss为例说明如何优化ndcg近似函数的方法

论文：[A general approximation framework for direct optimization of information retrieval measures](https://link.springer.com/article/10.1007/s10791-009-9124-x)

思想：将position function（也就是是排序1、2、3......）转化为基于score的平滑函数.该文章使用logistic function。

结构示意：
![img](https://note.youdao.com/yws/api/personal/file/E9EEB9939A6A486698C024C510D47E44?method=download&shareKey=40544a7486fe253e966739e6245b0a82)


```python
#使用方式
loss_fn = tfr.losses.make_loss_fn(loss_key='approx_ndcg_loss')
```



```python
#对应的函数
def _approx_ndcg_loss(labels,
                      logits,
                      weights=None,
                      reduction=tf.compat.v1.losses.Reduction.SUM,
                      name=None,
                      temperature=0.1):
  loss = losses_impl.ApproxNDCGLoss(name, temperature=temperature)
  with tf.compat.v1.name_scope(loss.name, 'approx_ndcg_loss',
                               (labels, logits, weights)):
  	return loss.compute(labels, logits, weights, reduction)
```




## 2.group_size作用是什么
论文参考：[Learning Groupwise Multivariate Scoring Functions Using Deep
Neural Networks](https://dl.acm.org/doi/abs/10.1145/3341981.3344218)

思想：不同的预测集同时预测会互相影响预测结果。

![image](https://note.youdao.com/yws/api/personal/file/88847DB0EF734AE49BB8F51D69743148?method=download&shareKey=5fc9812ba2a968c4f618a4288b62d9a2)

**tfr.model.make_groupwise_ranking_fn**

- `_GroupwiseRankingModel()`: #根据groupsize和group_score_fn构造同时预测groupsize元素的打分函数
  - **group_score_fn:** A scoring function for a `group_size` number of examples.
  - **group_size:** An integer denoting the number of examples in `group_score_fn`.  
  - **transform_fn:** A user-provided function that transforms raw
    features into dense Tensors with the following signature.
- `_make_model_fn()`: #使用ranking_model.compute_logits计算分数，**ranking_head中create_estimator_spec生成tf.estimator中的tf.estimator.EstimatorSpec对象
  - **ranking_head:** `create_ranking_head()`中返回`_RankingHead`类
  - ranking_model：`_GroupwiseRankingModel()`返回类

`_GroupwiseRankingModel`步骤：

> 1. Each instance in a mini-batch will form a number of groups
>
>    - use `_update_scatter_gather_indices` to genarate groups.
>
>    - For context features, We have shape [batch_size * num_groups, ...]
>
>    - For example feature, we have shape [batch_size * num_groups,group_size, ...]
>
> 2. Each group of examples are scored by `_score_fn` and scores for individual examples are accumulated into logits
>    - [batch_size * num_groups, logits_size]  -->  [batch_size, num_groups,logits_size]
>    - Scatter scores from [batch_size, num_groups, group_size] to [batch_size, list_size]



## 3.ranking head的作用

思想：可以根据多个label生成多头loss,进行多目标优化

`tfr.head.create_ranking_head`

- loss_fn
- eval_metric_fns
- optimizer
- train_op_fn
- name



## 4.transform_fn的作用

```
作用：Returns dense tensors from features using feature columns
如，将userlevel的1维转8维embedding
```

`tfr.feature.encode_listwise_features`

- features：feature_dict,input_fn  {name:tensor} 
- context_feature_columns：features中属于context_feature的 feature_dict
- example_feature_columns：features中属于example_feature的 feature_dict
- input_size：default to None. 指定example_feature第二维的维数为input_size
- mode：train/eval/predict
- scope：空间名

注：`example_feature_columns/example_feature_columns` ：使用tf.feature_column定义好的各列特征的属性，连续还是离散特征，离散特征是否进行emb，以及设置emb的维度。

**步骤**：

>1、tf.compat.v2.feature_column.make_parse_example_spec
>
>2、encode  context_feature_columns
>
>3、encode  example_feature_columns
>
>- reshape example_feature：[batch_size, input_size] -> [batch * input_size]
>        此处，若input_size未设定，则input_size = list_size 
>- encode reshaped_features
>- reshape reshaped_features:[batch * input_size] ->[batch_size, input_size] 
>
>4、return : context_features={name:tensor}，example_features

















