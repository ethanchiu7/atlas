<!doctype html>
<html>
<head>
    <meta charset="UTF-8">
    <link rel="stylesheet" media="all" href="normalize.css">
    <link rel="stylesheet" media="all" href="core.css">
    <link rel="stylesheet" media="all" href="style.css">
    <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
</head>
<body data-document>&nbsp;</body>
</html>


# Concept

    environment: the world that the agent lives in and interacts with. 
    state: a complete description of the state of the world. 
    observation: a partial description of a state.
    reward: the goal of the agent is to maximize its cumulative reward
    policy: a rule used by an agent to decide what actions to take.
    on-policy: they don’t use old data(Sarsa).
    off-policy: are able to reuse old data very efficiently(Q-Learning).
    
# Exploration vs Exploitation

    Exploration: 去之前没吃过的餐厅试试。
    Exploitation: 根据历史经验，直接去吃过最满意的餐厅
    而Exploitation虽然可以马上吃到比较满意的餐厅
    说白了，探索有风险，不探索就永远没有进步的可能性。
    如何在这两者间进行权衡是需要Agent进行学习的.
    
    
# Model-Free vs Model-Based

    whether the agent has access to (or learns) a model of the environment.
    By a model of the environment, we mean a function which predicts state transitions and rewards.
    
    Model-Based: having a model is that it allows the agent to plan by thinking ahead
    A particularly famous example of this approach is AlphaZero.
    
    Model-Free: a ground-truth model of the environment is usually not available to the agent.
    
    While model-free methods forego the potential gains in sample efficiency from using a model
    , they tend to be easier to implement and tune. 

# Q-Learning

# Policy-Gradient

 \pi_{\theta}(a|s). 
    
# Actor-Critic
   
    The actor takes in the current environment state and determines the best action to take,
    the critic plays the evaluation role by taking in the environment state and action 
    and returning an action score

    
    Actor critic models tend to require much less training time than policy gradient methods
    
# How

    
    


# Example

# Summary

# Conclusion

# Prospect

# Reference
    https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
    https://www.youtube.com/watch?v=w_3mmm0P0j8
    https://www.youtube.com/watch?v=O5BlozCJBSE
