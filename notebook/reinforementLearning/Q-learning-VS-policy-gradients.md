# source
    https://ai.stackexchange.com/questions/6196/q-learning-vs-policy-gradients
   
# question

As far as I understand Q-learning and policy gradients are the two major approaches used to solve RL problems. While Q-learning aims to predict the reward of a certain action taken in a certain state, policy gradients directly predict the action itself.

However, both approaches appear identical to me i.e. predicting the maximum reward for an action (Q-learning) is equivalent to predicting the probability of taking the action directly (PG). Is the difference in the way the loss is backpropagated?
    
# answer

Both methods are theoretically driven by the Markov Decision Process construct, and as a result use similar notation and concepts. In addition, in simple solvable environments you should expect both methods to result in the same - or at least equivalent - optimal policies.


However, they are actually different internally. The most fundamental differences between the approaches is in how they approach action selection, both whilst learning, and as the output (the learned policy). In Q-learning, the goal is to learn a single deterministic action from a discrete set of actions by finding the maximum value. With policy gradients, and other direct policy searches, the goal is to learn a map from state to action, which can be stochastic, and works in continuous action spaces.


As a result, policy gradient methods can solve problems that value-based methods cannot:

- Large and continuous action space. However, with value-based methods, this can still be approximated with discretisation - and this is not a bad choice, since the mapping function in policy gradient has to be some kind of approximator in practice.

- Stochastic policies. A value-based method cannot solve an environment where the optimal policy is stochastic requiring specific probabilities, such as Scissor/Paper/Stone. That is because there are no trainable parameters in Q-learning that control probabilities of action, the problem formulation in TD learning assumes that a deterministic agent can be optimal.


However, value-based methods like Q-learning have some advantages too:


- Simplicity. You can implement Q functions as simple discrete tables, and this gives some guarantees of convergence. There are no tabular versions of policy gradient, because you need a mapping function 𝑝(𝑎∣𝑠,𝜃) which also must have a smooth gradient with respect to 𝜃.

- Speed. TD learning methods that bootstrap are often much faster to learn a policy than methods which must purely sample from the environment in order to evaluate progress.


There are other reasons why you might care to use one or other approach: 

- You may want to know the predicted return whilst the process is running, to help other planning processes associated with the agent.

- The state representation of the problem lends itself more easily to either a value function or a policy function. A value function may turn out to have very simple relationship to the state and the policy function very complex and hard to learn, or vice-versa.


Some state-of-the-art RL solvers actually use both approaches together, such as Actor-Critic. This combines strengths of value and policy gradient methods


