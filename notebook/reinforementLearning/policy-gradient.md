# Reference
    https://zhuanlan.zhihu.com/p/21725498
# Why Policy Network
    
# 目标函数

    L(\theta) = \mathbb E(r_1+\gamma r_2 + \gamma^2 r_3 + ...|\pi(,\theta)) 所有带衰减reward的累加期望
# 参数更新
- 就给我一个Policy Network，也没有loss，怎么更新
    如果某一个动作得到reward多，那么我们就使其出现的概率增大
    ，如果某一个动作得到的reward少，那么我们就使其出现的概率减小