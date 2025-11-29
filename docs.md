Yuge (Jimmy) Shi
Blog Posts
Yuge (Jimmy) Shi
Yuge (Jimmy) Shi
Senior Research Scientist, Google DeepMind

 London, United Kingdom
 Email
 Twitter
 Google Scholar
A vision researcher’s guide to some RL stuff: PPO & GRPO
 20 minute read
 Published: January 31, 2025

First up, some rambles as usual.

It has been a while since I last wrote a blog post. Life has been hectic since I started work, and the machine learning world is also not what it was since I graduated in early 2023. Your average parents having LLM apps installed on their phones is already yesterday’s news – I took two weeks off work to spend Lunar New Year in China, which only serves to give me plenty of time to scroll on twitter and witness DeepSeek’s (quite well-deserved) hype peak on Lunar New Year’s eve while getting completely overwhelmed.

So this feels like a good time to read, learn, do some basic maths, and write some stuff down again.

What this blog post covers, and who is it for
This is a deep dive into Proximal Policy Optimization (PPO), which is one of the most popular algorithm used in RLHF for LLMs, as well as Group Relative Policy Optimization (GRPO) proposed by the DeepSeek folks, and there’s also a quick summary of the tricks I find impressive in the DeepSeek R1 tech report in the end.

This is all done by someone who’s mostly worked on vision and doesn’t know much about RL. If that’s you too, I hope you will find this helpful.

LLM pre-training and post-training
The training of an LLM can be separated into a pre-training and post-training phase:

Pre-training: the classic “throw data at the thing” stage where the model is trained to do next token prediction using large scale web data;
Post-training: This is where we try to improve the model’s reasoning capability. Typically there are two stages to post-training, namely
Stage 1: SFT (Supervised Finetuning): as the name implies, we use supervised learning first by fine-tuning the LLM on a small amount of high quality expert reasoning data; think instruction-following, question-answering and/or chain-of-thoughts. The hope is, by the end of this training stage, the model has learned how to mimic expert demonstrations. This is obviously the ideal way to learn if we had unlimited amount of high quality, expert data, but since we don’t –
Stage 2: RLHF (Reinforcement Learning from Human Feedback): Not enough human expert reasoning data? This is where :sparkles: RL :sparkles: gets to shine! RLHF uses human feedback to train a reward model, which then guides the LLM’s learning via RL. This aligns the model with nuanced human preferences, which…I think we all agree is important :paperclip:.
DeepSeek’s ultra efficient post-training
Notably, one of the most surprising thing about the DeepSeek R1 tech report is that their R1-zero model completely skips the SFT part and applies RL directly to the base model (DeepSeek V3). There are a few benefits to this:

Computational efficiency: skipping one stage of post-training brings computational efficiency;
Open-ended learning: Allows the model to “self-evolve” reasoning capabilities through exploration;
Alignment: Avoiding biases introduced by human-curated SFT data.
Caveat: while it seems like a “duh” moment to see someone saving compute by skipping a whole stage of post-training, I suspect you won’t be able to pull it off without a very good base model.

But they didn’t stop there! DeepSeek also make the RLHF part more efficient by introducing GRPO to replace PPO, which eliminates the need for a separate critic model (typically as large as the policy model), reducing memory and compute overhead by ~50%. To see why and how they did this, and for our own intellectual indulgence, let’s now have a look at exactly how RLHF is done and where these algorithms comes in.

RLHF
Let’s break down the workflow of RLHF into steps:

Step 1: For each prompt, sample multiple responses from the model;
Step 2: Humans rank these outputs by quality;
Step 3: Train a reward model to predict human preferences / ranking, given any model responses;
Step 4: Use RL (e.g. PPO, GRPO) to fine-tune the model to maximise the reward model’s scores.
As we can see the process here is relatively simple, with two learnable components, i.e. the reward model and “the RL”. Now let’s dive into each component with more details.

Reward Model
The reward model is truly on the front-line of automating jobs: realistically, we can’t have humans rank all the outputs of the model. A cost-saving approach is to then have annotators rate a small portion of the LLM outputs, then train a model to predict these annotators’ preferences — and that is where the reward model comes in. With that said, now let’s look at some maths:

Let’s denote our learnable reward model as 
. Given a prompt 
, the LLM generate 
 responses 
. Then given that a response 
 is preferrable to 
 according to the human rater, the reward model is trained to minimise the following objective:

 
 where 
 denotes the sigmoid function.

Side note: The objective is derived from the Bradley-Terry model, which defines the probability that a rater prefers 
 over 
 as:
 

Taking the negative log-likelihood of this probability gives the loss 
 above. The sigmoid 
 emerges naturally from rearranging the Bradley-Terry ratio.

Note that the reward for a partial response is always 0; only for complete responses from the LLM would the reward model return a non-zero scalar score. This important fact will become relevant later.

“The RL part”: PPO
This part is only for the readers who are curious about PPO, and you don’t really need to understand this if your goal of opening this blog post is to understand GRPO. All I can say is though it brought me great joy to finally understand how PPO works, and then great sense of vindication when I realised how much simpler GRPO is compared to PPO. So if you’re ready for an emotional rollercoaster – let’s dive in.

First, a high level overview. PPO stands for proximal policy optimization, and it requires the following components:

Policy (
): the LLM that has been pre-trained / SFT’ed;
Reward model (
): a trained and frozen network that provides scalar reward given complete response to a prompt;
Critic (
): also known as value function, which is a learnable network that takes in partial response to a prompt and predicts the scalar reward.
Congratulations – by calling the LLM a “policy” you are now an RL person :tada:! The purpose of each component becomes a little clearer once we get to know the workflow, which contains five stages:

Generate responses: LLM produces multiple responses for a given prompt;
Score responses: The reward model assigns reward for each response;
Compute advantages: Use GAE to compute advantages (more on this later, it’s used for training the LLM);
Optimise policy: Update the LLM by optimising the total objective;
Update critic: train the value function to be better at predicting the rewards given partial responses.
Now let’s take a look at some of these stages/components in more details, and then see how they all come together.

Terminologies: states and actions
Some more RL terminologies before we move on. In the discussion of this section we are going to use the term state, denote as 
, and action, denote as 
. Note that here the subscript 
 is used to denote the state and action at a token level; in contrast, previously when we defined our prompt 
 and responses 
, the subscript 
 is used to denote the response at an instance level.

To make this a little clearer, let’s say we give our LLM a prompt 
. The LLM then starts generating a response 
 of length 
 one token at a time:

: our state is just the prompt, i.e. 
, and the first action 
 is just the first word token generated by the LLM;
: the state becomes 
, as the LLM is generating the next action 
 while conditioned on the state; …
: the state is 
, and the LLM generates the final action 
.
Connecting this to the previous notations again, all the actions stringing together makes one response, i.e. 
.

General Advantage Estimation (GAE)
Our policy is updated to optimise advantage – intuitively, it defines how much better a specific action 
 (i.e. word) is compared to an average action the policy will take in state 
 (i. e. prompt + generated words so far). Formally:

 
Where 
 is the expected cumulative reward of taking a specific action 
 in state 
, and 
 is the expected cumulative reward of average action the policy takes in state 
.

There are two main ways of estimating this advantage, each with their trade-offs, namely, 1) Monte-Carlo (MC): Use the reward of the full trajectory (i.e. full responses). This approach has high variance due to the sparse reward – it is expensive to take enough samples from the LLM to optimise using MC, but it does have low bias as we can accurately model the reward; 2) Temporal difference (TD): Use one-step trajectory reward (i.e. measure how good is the word that’s just been generated given the prompt). By doing so we can compute reward on a token level, which significantly reduces the variance, but at the same time the bias goes up as we can’t as accurately anticipate the final reward from a partially generated response.

This is where GAE comes in – it is proposed to balance the bias and variance through a multi-step TD. However, recall that previously we mentioned that the reward model will return 0 if the response was incomplete: how will we compute TD without knowing how the reward would change before and after generating a word? We therefore introduce a model that does just that, which we call “the critic”.

The critic (value function) :detective:
The critic is trained to anticipate the final reward given only a partial state, so that we can compute the TD. Training the critic 
 is fairly straightforward:

Given a partial state 
, we want to predict the reward model’s output given the full state 
. The objective for the critic can be written as

 
where 
 denotes the stop gradient operation. As we can see, the critic is trained with a simple L2 loss to the reward model’s score.

You might notice that while the reward model 
 is trained before PPO and frozen, the critic is trained alongside the LLM, even though its job is also just to predict the reward. This is because the value function must estimate the reward for partial response given the current policy; as a result, it must be updated alongside the LLM, to avoid its predictions to become outdated and misaligned. And this, is what they call, actor-critic in RL (mic-drop).

Back to GAE
With the critic 
, we now have a way to anticipate the reward from a partial state. Now let’s get on with GAE, which as mentioned computes a multi-step TD objective:

 
 
where 
 denotes the number of TD steps and 
 (because obviously you can’t compute TD beyond the length of the trajectory). 
 denotes the TD error at step 
, and is computed as:

 
To put simply, the TD error computes the difference between expected total reward of one time step, and 
 estimates advantage by computing the aggregated single-step TD errors over 
 steps. The 
 in the GAE equation controls the trade-off between the variance and the bias: when 
, GAE reduces to single-step TD; and when 
, GAE becomes MC.

In RLHF, we want to maximise this advantage term, thereby maximising the reward for every token the LLM generates.

Side note: ok, I cut some corners for simplicity here. Originally there is also a discount factor 
 in GAE:  
 
 
 which is also used in the TD error 
, and there is also an extra reward term  
 
 But since we almost always have 
, and 
 for 
 which is always the case, I took a shortcut to simplify and omit those terms.

Putting it together – PPO objective
There are a few components to the PPO objective, namely 1) the clipped surrogate objective, 2) the entropy bonus, 3) the KL penalty.

1. The clipped surrogate objective
This is where we maximise 
, so that each token the LLM predicted maximises the reward (or, by definition of advantage earlier, each token the LLM predicts should be much better than its average prediction). The clipped surrogate objective constrains policy updates with a probability ratio 
:

 
where 
 controls the clipping range, 
 the probability ratio of predicting a specific token 
 at given cumulative state 
, before and after the update:

 
 
Concrete example:

Let’s say the LLM assigns the word unlimited with the following probabilities:
Before update: 0.1,
After update: 0.3. Then the probability ratio 
;
If we take 
, 
 gets clipped to 1.2;
The final clipped surrogate loss is 
.
You can think of clipping as a way to prevent overconfidence – without clipping, a large 
 could cause the policy to overcommit to an action.

2. KL divergence penalty
Additionally, we have the KL divergence penalty which prevents the current policy 
 from deviating too far from the original model that we are finetuning from 
:  
 

The KL is simply estimated by taking the average over sequence and batch.

Pseudocode:

# Compute KL divergence between original and current policy/model
logits_orig = original_model(states)  # Original model's logits
logits_current = current_model(states)  # Current model's logits

probs_orig = F.softmax(logits_orig, dim=-1)
log_probs_orig = F.log_softmax(logits_orig, dim=-1)
log_probs_current = F.log_softmax(logits_current, dim=-1)

kl_div = (probs_orig * (log_probs_orig - log_probs_current)).sum(dim=-1)
kl_penalty = kl_div.mean()  # Average over sequence and batch
3. Entropy bonus
The entropy bonus encourages exploration of LLM’s generation by penalising low entropy:

 
Pseudocode:

# Compute entropy of current policy
probs_current = F.softmax(logits_current, dim=-1)
log_probs_current = F.log_softmax(logits_current, dim=-1)

entropy = -(probs_current * log_probs_current).sum(dim=-1)
entropy_bonus = entropy.mean()  # Average over sequence and batch
Finally, the PPO objective
Given the three terms above, in addition to the value function MSE loss (recall it is optimised along with the LLM), the PPO objective is defined as follows:

 
 
  
 
 
  
 
 
  
 
 
 
A summary of the different terms in this objective is as follows:

Term	Purpose
Maximize rewards for high-advantage actions (clipped to avoid instability).
Maximize entropy to encourage exploration.
Penalize deviations from the reference policy (stability).
Minimize error in value predictions (critic L2 loss).
“The RL part”: GRPO
It’s super easy to understand GRPO now that we have a good understanding of PPO, and the key difference lies in how the two algorithms estimate advantage 
: instead of estimating advantage through the critic like in PPO, GRPO does so by taking multiple samples from the LLM using the same prompt.

Workflow:

For each prompt 
, sample a group of 
 responses 
 from the LLM policy 
;
Compute rewards 
 for each response using the reward model 
;
Calculate group-normalised advantage for each response:  
 
 
 where 
 and 
 denotes the within-group mean and standard deviation, respectively.
A lot simpler, right? In GRPO, advantage is approximated as the normalised reward of each response within its group of responses. This removes the need of a critic network calculating per-step rewards, not to mention the mathematical simplicity and elegance. It does somewhat beg the question – why didn’t we do this sooner?

I don’t have a good answer to this question due to a lack of hands-on experience: I’m guessing this is tied to hardware capabilities, as the modern GPUs/TPUs we have access to these days make it possible to sample in a much faster and more efficient manner. Again I’m not an expert, so insights on this are very welcomed!

Update: some insights from @him_sahni on this, who “did RL in his past life”: the reason “why no one has tried GRPO before” is – we have. In REINFORCE, you update the policy by subtracting a baseline (typically the average reward from several trajectories) to reduce variability. In fact, theory shows that the ideal baseline is the total expected future reward from a state, often called the “value”. Using a value function as the baseline is known as the actor-critic approach, and PPO is a stable version of that. Now, in traditional REINFORCE, the baseline can be any function of the current state, and traditionally is just the reward for the trajectories in a single batch; in GRPO, this baseline is computed over 1000 samples generated for each prompt, which is :rainbow: novel :rainbow:.

The GRPO objective
Similar to PPO, GRPO still make use of a clipped surrogate loss as well as the KL penalty. The entropy bonus term is not used here, as the group-based sampling already encourages exploration. The clipped surrogate loss is identical to the one used in PPO, but for completeness sake here it is:  
 
 
 
 
 

then with the KL penalty term, the final GRPO objective can be written as:

 
 
  
 
 
 
 
More thoughts on R1: Brutal Simplicity
Finally, a few words on R1.

Overhyped or not, one thing that really stands out about the R1 from reading the paper is that it embraces a stripped-down, no-nonsense approach to LLM training, prioritising brutal simplicity over sophistication. GRPO is just the tip of the iceberg. Here are some more examples on of its brutal simplicity:

1. Rule-Based, Deterministic Rewards
What: Abandon neural Process Reward Models (PRMs) or Outcome Reward Models (ORMs). Use binary checks, including:
Answer Correctness: Final answer matches ground truth (e.g., math solutions, code compilation).
Formatting: Force answers into <think>...</think><answer>...</answer> templates.
Language Consistency: Penalise mixed-language outputs (e.g., English reasoning for Chinese queries).
Why: Deterministic rules sidestep reward hacking (e.g., models tricking neural reward models with plausible-but-wrong steps) and eliminate reward model training costs.
2. Cold-Start Data: Minimal Human Touch
What: Instead of curating massive SFT datasets, collect a few thousand high-quality CoT examples via:
Prompting the base model with few-shot examples.
Light human post-processing (e.g., adding markdown formatting).
Why: Avoids costly SFT stages while bootstrapping RL with “good enough” starting points.
3. Rejection Sampling: Filter Hard, Train Harder
What: After RL training, generate 600k reasoning trajectories, then throw away all incorrect responses. Only keep the “winners” (correct answers) for supervised fine-tuning (SFT). No fancy reranking, no preference pairs. Just survival-of-the-fittest filtering.
Why: It works, why not!
4. Distillation: Copy-Paste Reasoning
What: To train smaller models, directly fine-tune them on 800k responses generated by DeepSeek-R1. No RL, no iterative alignment—just mimicry.
Why: Smaller models inherit reasoning patterns discovered by the larger model’s brute-force RL, bypassing costly RL for small-scale deployments.
DeepSeek-R1’s design reflects a broader trend in AI: scale and simplicity often outperform clever engineering. By ruthlessly cutting corners — replacing learned components with rules, leveraging massive parallel sampling, and anchoring to pre-trained baselines — R1 achieves SOTA results with fewer failure modes. It’s not elegant, but it’s effective.

Who would’ve thought the best way to incentivise good thinking is to :rainbow: stop overthinking it :rainbow:.

 Tags: Large Language Models Machine Learning RLHF

Share on
 Twitter  Facebook  LinkedInPreviousNext
Leave a Comment


Disqus seems to be taking longer than usual. Reload?

You May Also Enjoy
An incomplete and slightly outdated literature review on augmentation based self-supervise learning
 28 minute read

 Published: December 14, 2021


How I learned to stop worrying and write ELBO (and its gradients) in a billion ways
 19 minute read

 Published: June 19, 2020

Latex equations not rendering? Try using a different browser or this link here.

Gaussian Process, not quite for dummies
 19 minute read

 Published: September 05, 2019

Before diving in
For a long time, I recall having this vague impression about Gaussian Processes (GPs) being able to magically define probability distributions over sets of functions, yet I procrastinated reading up about them for many many moons. However, as always, I’d like to think that this is not just due to my procrastination superpowers. Whenever I look up “Gaussian Process” on Google, I find these well-written tutorials with vivid plots that explain everything up until non-linear regression in detail, but shy away at the very first glimpse of any sort of information theory. The key takeaway is always,

A Gaussian process is a probability distribution over possible functions that fit a set of points.

© 2025 Yuge (Jimmy) Shi. Powered by Jekyll & AcademicPages, a fork of Minimal Mistakes.
For updates follow @adithya_s_k on  Twitter
logo
AI Engineering Academy
Theory Behind GRPO

Search
 
 adithya-s-k/AI-Engineering.academy
1.6k
174
Home
Prompt Engineering
RAG
LLM
Deployment
Agents
Projects
Blog
LLM
Finetuning Techniques
PreTraining LLMs
SFT
PPO(Proximal Policy Optimization)
DPO(Direct Preference Optimization)
ORPO(Odds Ratio Preference Optimization)
GRPO(Group Relative Policy Optimization)
Theory Behind GRPO
Hands on with GRPO
Qwen 0.6 with GRPO
LLM Finetuning Hands on
Gemma
Llama2
Llama3
Mistral
VLM
Florence2
PaliGemma
Serverless Finetuning with Modal
Training NanoGPT on Modal
Training Nanochat on Modal
Fine-tuning Gemma with Unsloth
Multi-GPU Training with Axolotl
LLM Architecture
Theory Behind GRPO
What is GRPO?

Group Relative Policy Optimization (GRPO) is a reinforcement learning algorithm designed to train large language models (LLMs) for complex tasks like solving math problems or writing code. Unlike older methods, GRPO is memory-efficient because it doesn't use a separate "value function" (a model that estimates future rewards). Instead, it generates multiple answers for each question, scores them with a reward model, and uses the average score as a reference to decide which answers are better. This makes it easier to train large models on limited hardware, which is surprising because it still performs well on tough tasks like reasoning.

How Does GRPO Work?

GRPO works in simple steps:

For each question, the model generates several possible answers.
Each answer is scored using a reward model (e.g., giving a high score for a correct math answer).
The average score of all answers for that question is calculated.
The model compares each answer's score to this average to see how good it is (this is called the "advantage").
The model then updates itself to favor answers with higher advantages, ensuring it doesn't change too much at once to stay stable.
This process repeats, helping the model get better over time. A surprising detail is how it uses the group average as a baseline, which reduces the need for extra memory while still improving performance.

Why is GRPO Important?

GRPO is important because it saves memory and computational resources, making it easier to train large models on devices with limited power. It's been used in models like DeepSeek R1, which competes with top AI models in reasoning tasks, showing big improvements in math and coding benchmarks.

A Comprehensive Analysis of Group Relative Policy Optimization (GRPO)

Introduction to Reinforcement Learning and Policy Optimization

Reinforcement learning (RL) is a branch of machine learning where an agent learns to make decisions by interacting with an environment, aiming to maximize a cumulative reward. In the context of large language models (LLMs), RL is used to fine-tune these models to align with human preferences and enhance their performance on specific tasks, such as mathematical reasoning or code generation.

Policy optimization is a class of RL algorithms that directly optimize the policy, which is the strategy the agent uses to decide actions based on states. One of the most popular policy optimization algorithms is Proximal Policy Optimization (PPO), known for its stability and efficiency. PPO uses a clipped surrogate objective to prevent large policy updates and relies on a value function to estimate advantages, ensuring stable training.

However, as LLMs grow larger and tasks become more complex, PPO faces challenges, including high memory overhead from maintaining a value function and increased computational costs. The value function, typically another neural network of comparable size to the policy model, estimates the expected future reward from a given state, adding significant resource demands.

To address these limitations, Group Relative Policy Optimization (GRPO) was introduced, first detailed in the DeepSeekMath paper (DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models). GRPO is designed to enhance the reasoning capabilities of LLMs, particularly for mathematical and coding tasks, by eliminating the need for a value function and leveraging group-based advantage estimation.

The Emergence of Group Relative Policy Optimization (GRPO)

GRPO addresses PPO's limitations by introducing a novel reinforcement learning algorithm that simplifies advantage estimation and reduces memory usage. The key innovation lies in its approach to advantage calculation: instead of relying on a separate value function network, GRPO generates multiple responses for each prompt and uses the mean reward of these responses as the baseline. This group-based method reduces variance in advantage estimates and significantly lowers memory usage, making it suitable for training large models on resource-constrained hardware.

GRPO was first applied in the training of DeepSeek R1, an open-source model challenging OpenAI's o1 in advanced reasoning, as noted in various analyses (DeepSeek R1: Understanding GRPO and Multi-Stage Training | by BavalpreetSinghh | Jan, 2025 | Artificial Intelligence in Plain English). Its effectiveness in improving performance on benchmarks like GSM8K and MATH highlights its potential to revolutionize LLM training for reasoning tasks.

Mathematical Formulation of GRPO

To understand GRPO's mechanics, consider the following formulation, as detailed in resources like The Math Behind DeepSeek: A Deep Dive into Group Relative Policy Optimization (GRPO) | by Sahin Ahmed, Data Scientist | Jan, 2025 | Medium:

For each prompt \(s_j\), generate \(K_j\) responses \(a_{jk}\), where \(k = 1, 2, ..., K_j\).
Each response \(a_{jk}\) is scored using a reward model, yielding a reward \(R_{jk}\).
Calculate the mean reward for the group: \(\(\bar{R}_j = \frac{1}{K_j} \sum_{k=1}^{K_j} R_{jk}\)\)
The advantage for each response is \(A_{jk} = R_{jk} - \bar{R}_j\), reflecting how much better or worse the response is compared to the group average.
The policy update is guided by the following loss function:

\[\mathcal{L} = - \sum_{j=1}^M \sum_{k=1}^{K_j} \left( \frac{\pi_{\theta}(a_{jk} | s_j)}{\pi_{\theta_{\text{old}}}(a_{jk} | s_j)} A_{jk} \right) + \beta \sum_{j=1}^M \text{KL}(\pi_{\theta}( \cdot | s_j) || \pi_{\theta_{\text{old}}}( \cdot | s_j))\]
Here:

\(M\) is the number of prompts.
\(\pi_{\theta}\) is the new policy parameterized by \(\theta\).
\(\pi_{\theta_{\text{old}}}\) is the old policy.
\(\beta\) is a coefficient controlling the strength of the KL divergence penalty, ensuring the new policy doesn't deviate too far from the old one for stability.
The importance ratio:

\[\frac{\pi_{\theta}(a_{jk} | s_j)}{\pi_{\theta_{\text{old}}}(a_{jk} | s_j)}\]
for a sequence \(a_{jk}\) is computed as the product of the ratios for each token in the sequence, reflecting the policy's probability distribution over the entire response.

Implementation Steps of GRPO

Implementing GRPO involves the following steps, as observed in its application to DeepSeekMath and detailed in A Deep Dive into Group Relative Policy Optimization (GRPO) Method: Enhancing Mathematical Reasoning in Open Language Models - MarkTechPost:

Data Preparation: Collect a batch of prompts, typically in chain-of-thought format for reasoning tasks, such as questions from GSM8K and MATH datasets.
Response Generation: For each prompt, generate multiple responses (e.g., 64 samples per question, as used in DeepSeekMath) using the current policy, with a maximum length of 1024 tokens.
Reward Scoring: Use a reward model to assign rewards to each response. The reward model, initially trained on a base model like DeepSeekMath-Base 7B with a learning rate of 2e-5, evaluates response quality based on accuracy and formatting, as noted in AWS | Community | Deep dive into Group Relative Policy Optimization (GRPO).
Advantage Calculation: For each prompt, calculate the mean reward \(\bar{R}_j\) of its responses and compute the advantage for each response: \(A_{jk} = R_{jk} - \bar{R}_j\)
Policy Update: Update the policy parameters to minimize the loss function, with a learning rate of 1e-6 for the policy model, a KL coefficient of 0.04, and a batch size of 1024. Perform a single update per exploration stage to ensure stability, as seen in the training details of DeepSeek R1 (DeepSeek R1: Understanding GRPO and Multi-Stage Training | by BavalpreetSinghh | Jan, 2025 | Artificial Intelligence in Plain English).
This process is iterative, with GRPO improving the model by leveraging data generated during training, making it an online learning algorithm.

Comparison with Other Policy Optimization Methods

To contextualize GRPO, compare it with other methods, as summarized in the following table based on insights from A vision researcher's guide to some RL stuff: PPO & GRPO - Yuge (Jimmy) Shi and r/ChatGPTPro on Reddit: GRPO (Group Relative Policy Optimization) explanation compared to PPO:

Method	Value Function	Advantage Estimation	Stability Mechanism	Memory Usage
PPO	Yes	Uses value function for baseline	Clipped surrogate objective	High (due to value function)
TRPO	Yes	Uses value function, trust region constraint	Hessian-based trust region	High
REINFORCE	No	No baseline, high variance	None	Low
GRPO	No	Group mean as baseline, reduces variance	KL divergence in loss function	Low
PPO: Relies on a value function for advantage estimation, with a clipped importance ratio to prevent large updates. It is stable but memory-intensive, especially for large models.
TRPO: Uses a trust region to constrain policy updates, ensuring stability but at a higher computational cost due to Hessian calculations.
REINFORCE: A basic policy gradient method without constraints, leading to unstable training and high variance, but with low memory usage.
GRPO: Eliminates the value function, using group-based advantages to reduce variance and memory usage, with KL divergence ensuring stable updates. It is particularly efficient for LLMs, as seen in DeepSeek R1's training.
Case Study: Application in DeepSeek R1

DeepSeek R1, an open-source model challenging OpenAI's o1 in advanced reasoning, utilized GRPO to achieve remarkable results. Introduced in the DeepSeekMath paper, GRPO was applied to DeepSeekMath-Instruct 7B, using a subset of English instruction tuning data (~144K questions). The training details included:

Learning rate for policy model: 1e-6
KL coefficient: 0.04
Samples per question: 64
Max length: 1024
Batch size: 1024
Single update per exploration stage
Performance improvements were significant, as noted in DeepSeek R1: Understanding GRPO and Multi-Stage Training | by BavalpreetSinghh | Jan, 2025 | Artificial Intelligence in Plain English:

GSM8K: Improved from 82.9% to 88.2%
MATH: Improved from 46.8% to 51.7%
CMATH (out-of-domain): Improved from 84.6% to 88.8%
These results highlight GRPO's effectiveness in enhancing mathematical reasoning while optimizing resource usage, making it a game-changer for training LLMs on complex tasks.

Advantages and Potential Disadvantages

Advantages:

Reduced Memory Usage: By eliminating the value function, GRPO requires less memory, crucial for large models, as discussed in AWS | Community | Deep dive into Group Relative Policy Optimization (GRPO).
Simplified Advantage Estimation: Using group means for baseline makes advantage calculation straightforward and efficient, reducing variance, as noted in The Math Behind DeepSeek: A Deep Dive into Group Relative Policy Optimization (GRPO) | by Sahin Ahmed, Data Scientist | Jan, 2025 | Medium.
Stable Training: The KL divergence constraint ensures controlled and stable policy updates, enhancing training reliability.
Potential Disadvantages:

Variance in Group Rewards: If the group size is small, the mean reward might not be a good estimator, leading to higher variance, as mentioned in community discussions (r/ChatGPTPro on Reddit: GRPO (Group Relative Policy Optimization) explanation compared to PPO).
Dependence on Reward Model: The quality of the reward model is critical, as inaccurate rewards can affect performance, a concern highlighted in A vision researcher's guide to some RL stuff: PPO & GRPO - Yuge (Jimmy) Shi.
Conclusion and Future Directions

GRPO represents a significant advancement in reinforcement learning for large language models, offering a more efficient and effective way to train models for complex reasoning tasks. Its application in DeepSeek R1 demonstrates its potential to push the boundaries of AI reasoning, achieving state-of-the-art performance with reduced resource requirements. Future research may focus on optimizing group-based methods, exploring adaptive group sizes, and extending GRPO to other domains beyond mathematics and coding, as suggested in A Deep Dive into Group Relative Policy Optimization (GRPO) Method: Enhancing Mathematical Reasoning in Open Language Models - MarkTechPost.

This comprehensive analysis provides a detailed understanding of GRPO, from its theoretical foundations to practical implementations, serving as a valuable resource for researchers and practitioners in artificial intelligence.

Key Citations

DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
AWS Community: Deep dive into Group Relative Policy Optimization (GRPO)
DeepSeek R1: Understanding GRPO and Multi-Stage Training | by BavalpreetSinghh | Jan, 2025 | Artificial Intelligence in Plain English
The Math Behind DeepSeek: A Deep Dive into Group Relative Policy Optimization (GRPO) | by Sahin Ahmed, Data Scientist | Jan, 2025 | Medium
A Deep Dive into Group Relative Policy Optimization (GRPO) Method: Enhancing Mathematical Reasoning in Open Language Models - MarkTechPost
A vision researcher's guide to some RL stuff: PPO & GRPO - Yuge (Jimmy) Shi
r/ChatGPTPro on Reddit: GRPO (Group Relative Policy Optimization) explanation compared to PPO
October 13, 2025
 
October 13, 2025
 Back to top
Previous
ORPO(Odds Ratio Preference Optimization)
Next
Hands on with GRPO
Copyright © 2024 Adithya S Kolavi
Made with Material for MkDocs
Skip to content
Navigation Menu
policy-gradient
GRPO-Zero

Type / to search
Code
Issues
7
Pull requests
Actions
Projects
Security
Insights
Owner avatar
GRPO-Zero
Public
policy-gradient/GRPO-Zero
Go to file
t
Name		
NTT123
NTT123
clean up
d41bb48
 · 
7 months ago
.gitignore
Initial commit
7 months ago
.python-version
first version
7 months ago
LICENSE
Initial commit
7 months ago
README.md
add memory efficient optimizer
7 months ago
config.yaml
add memory efficient optimizer
7 months ago
config_24GB.yaml
add memory efficient optimizer
7 months ago
countdown_task.py
fix regex corner cases
7 months ago
data_types.py
first version
7 months ago
grpo.py
fix end token bug
7 months ago
optimizer.py
clean up
7 months ago
pyproject.toml
add memory efficient optimizer
7 months ago
qwen2_model.py
clean up
7 months ago
tokenizer.py
first version
7 months ago
train.py
add memory efficient optimizer
7 months ago
uv.lock
first version
7 months ago
Repository files navigation
README
Apache-2.0 license
GRPO:Zero
GRPO training with minimal dependencies (and low GPU memory usage!). We implement almost everything from scratch and only depend on tokenizers for tokenization and pytorch for training.

No transformers and vLLM dependencies!
The default config is set to run on a single A40 GPU (48GB VRAM) for a few hours to get good results. (An A40 costs $0.44 per hour if you rent it from RunPod.)
We also support training with a 24GB VRAM GPU (e.g., an RTX 4090 GPU) by offloading the optimizer to CPU. Fortunately, this only adds a small overhead to the training because we only update the policy network a few hundred times during the entire training process.
We support several improvements over the original GRPO algorithm from the DAPO project, including:
Token-level policy gradient loss: every token is equally weighted in the policy gradient loss.
Removing KL Divergence: the KL divergence is not used in the policy gradient loss. This reduces GPU memory usage as we no longer need the reference policy network.
Overlong episode filtering: skips unfinished episodes that exceed context length limits. This stabilizes training. Though we disabled it by default to observe model learning under limited context length. Set skip_unfinished_episodes to true to enable it.
Algorithm
Group Relative Policy Optimization (GRPO) is an algorithm proposed by Deepseek for training large language models with reinforcement learning. The idea is simple: for each question, we randomly sample multiple answers. The advantage of an answer is then defined as the normalized reward. This gets rid of the value estimation network. In particular, we implement the following algorithm:

For each training step, randomly sample 
N
 questions 
q
1
,
q
2
,
⋯
,
q
N
.
For each question 
q
i
, sample 
M
 answers 
a
i
,
1
,
a
i
,
2
,
⋯
,
a
i
,
M
.
Compute the reward 
r
i
,
j
 for each answer 
a
i
,
j
.
Compute the mean and std of the rewards for each question 
q
i
.
μ
i
←
mean
(
r
i
,
1
,
r
i
,
2
,
⋯
,
r
i
,
M
)
σ
i
←
std
(
r
i
,
1
,
r
i
,
2
,
⋯
,
r
i
,
M
)

For each token 
t
 in the answer 
a
i
,
j
, compute the advantage as
A
i
,
j
[
t
]
←
r
i
,
j
−
μ
i
σ
i

Compute policy gradient using PPO surrogate objective. For simplicity, we will only do one policy update per iteration, in which the gradient of the PPO objective is equivalent to following vanilla policy gradient estimation (per token).
∇
θ
log
⁡
π
θ
(
a
i
,
j
[
t
]
)
⋅
A
i
,
j
[
t
]

Update the policy network 
π
(
θ
)
 using the gradient. Go back to step 1.
CountDown Task
We are going to train the Qwen2.5 models on the CountDown task. Given a list of 3 or 4 numbers and a target number, the model needs to generate a mathematical expression using simple arithmetic operations (+, -, *, /) that evaluates to the target number. For example:

Question: Given 1 2 3 4 and a target number 11. Show an expression that evaluates to 11.
Answer: 1 + (2 * 3) + 4
Reward Function
To solve the CountDown task, we will use the GRPO algorithm to train the model to generate the chain of thought reasoning before generating the final expression. Specifically, the model is trained to follow the format:

<think>Model step by step reasoning</think>
<answer>Final answer</answer>
The reward is the sum of two components:

Format Reward: The model earns a reward of 0.1 when it correctly follows the specified format with thinking and answer tags, and 0 otherwise.
Answer Reward: The model receives a reward of 1 if its final answer uses each provided number exactly once and correctly evaluates to the target value, otherwise it receives 0.
Training
We use the Qwen2.5-3B-Instruct model for training. To train the model, run the following commands:

# initialize the environment
pip install uv
uv sync

# install git-lfs
apt update; apt install git-lfs -y; git lfs install

# download the dataset
git clone https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4

# download the pretrained model
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
# train the model
uv run train.py
# train the model with a 24GB VRAM GPU (e.g., an RTX 4090 GPU)
uv run train.py --config config_24GB.yaml
Acknowledgements
This project builds upon the work of several outstanding projects:

DeepSeekMath for pioneering the GRPO algorithm.
DAPO for their enhancements to the original GRPO algorithm.
TinyZero for their implementation of GRPO and creation of the CountDown-Tasks-3to4 dataset.
nano-aha-moment for their clear implementation and tutorial on the GRPO algorithm.
Qwen2.5 for developing the high-quality pretrained model used in this project.
About
Implementing DeepSeek R1's GRPO algorithm from scratch

Resources
 Readme
License
 Apache-2.0 license
 Activity
 Custom properties
Stars
 1.7k stars
Watchers
 5 watching
Forks
 76 forks
Report repository
Releases
No releases published
Packages
No packages published
Languages
Python
100.0%
Footer
© 2025 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Community
Docs
Contact
Manage cookies
Do not share my personal information
×
📊 Curious about the rust data we used in this post?
Check out the dataset and fine-tune it yourself!

🔗 View Dataset on Oxen.ai
Oxen.ai
Home
Docs
Datasets
Arxiv Dives
Sign up

Sign in
Subscribe
Why GRPO is Important and How it Works
Greg Schoeninger
Greg Schoeninger
  Feb 11, 2025  12 min read  Arxiv Dives
Why GRPO is Important and How it Works
Last week on Arxiv Dives we dug into research behind DeepSeek-R1, and uncovered that one of the techniques they use in the their training pipeline is called Group Relative Policy Optimization (GRPO).

At it’s core, GRPO is a Reinforcement Learning (RL) algorithm that is aimed at improving the model’s reasoning ability. It was first introduced in their paper DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, but was also used in the post-training of DeepSeek-R1.

The process to go from DeepSeek’s base pre-trained language model to a reasoning model was laid out in detail in the DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning paper.


Last week we didn’t get too deep into the math or process behind GRPO or look at any code, so today the goal is to fully understand what is going on in GRPO and help apply it to your own work.

💡
Looking to build a Reinforcement Learning pipeline? We are looking for testers for our tooling to iterate between SFT and RFT. Reach out to hello@oxen.ai and we will get you early access.
Recap: How R1 Used GRPO
As a recap - the full pipeline for improving DeepSeek’s base model to the reasoning model alternates between using Supervised Fine Tuning (SFT) and Group Relative Policy Optimization (GRPO).

Supervised Fine Tuning (SFT)
Cold start the training with high quality data
A couple thousand examples verified by humans
Reinforcement Learning w/ GRPO
Train the model to have reasoning traces <reasoning></reasoning>
Deterministic rewards for formatting, consistency, and correctness
Supervised Fine Tuning (SFT)
Generate 800k Synthetic SFT data points and reject and filter
LLM As A Judge to filter incorrect responses
Reinforcement Learning w/ GRPO
Align the model to be helpful and harmless

In this post we will dive into the details of GRPO to give you a sense of how it works and where you can apply it to training your own models. I’ve been doing some experiments training smaller models with GRPO and will follow up with a post on implementation details and code examples to help tie together all the concepts.

Why is GRPO Important?
TLDR ~ The compute requirements drop significantly and it simplifies the RL. It just about cuts in half the compute requirements to do Reinforcement Learning from Human Feedback (RLHF) compared to what was used for ChatGPT (PPO). When you start considering LoRAs in the mix, this unlocks RL training for even the poorest of the GPU poor, and I got news for you. I tried it. And it works. I was able to successfully turn a 1B parameter Llama 3.2 model into a reasoning model with 16GB of VRAM. More on that in a subsequent post where I’ll share the code and hardware requirements.

💡 I was able to successfully turn a 1B parameter Llama 3.2 model into a reasoning model with 16GB of VRAM.
Basically we can all train reasoning models from our garages by spending < $100 on cloud GPU services. Or essentially “for free” if you are talking smol models on your own hardware. How does this work in under the hood? The next section will talk through the evolution from PPO to GRPO.

From PPO to GRPO
The Reinforcement Learning (RL) technique behind ChatGPT is rumored to be PPO which stands from Proximal Policy Optimization. The process was laid out in the InstructGPT paper to create a model that can follow instructions and go beyond simple predicting the next word.

The training process requires you to collect a lot of labeled data. For a given user query, you have the model generate multiple candidate responses. Then you have a human or AI in the loop to label and rank the outputs form best to worst. This can then be used as training data for a “reward model” who’s job is to calculate a “reward” for a new prompt that it sees. The reward should represent how good this response is, given the user query.


After you have collected all this ranked and labeled data you can kick off PPO to train your LLM.

The problem is that PPO can be pretty expensive to train. The diagram from the GRPO paper shows all the different LLMs that are in the mix during PPO and GRPO. There are 4 different LLMs in the blue and yellow boxes below.


To help demystify some of the lingo above, here are my quick definitions:

Policy Model - Fancy name for the current LLM you are training
Reference Model - A frozen version of the original LLM you are training
Reward Model - The model that was trained on human preferences (from the technique in InstructGPT above)
Value Model - A model that is trying to estimate the long term reward given certain actions
Reducing Memory Usage with GRPO
In PPO both the policy model and the value model have trainable parameters that need to be back-propagated through. Backprop requires a significant amount of memory. If you look at the diagram above, GRPO drops the value model.


PPO has 4 LLMs in the mix, which all require substantial memory and compute. The value and reward models are typically of a comparable parameter count to the LLM you are training. The reference model is usually a frozen copy of the initial language model.


Not only is this computationally expensive, there are a lot of moving parts and multiple models you are optimizing. The more moving parts, typically the harder it is to optimize. GRPO helps simplify things.

For fun, I decided to try a bunch of different model sizes on an H100 and see how easy it was to fine tune with GRPO


If you want all the technical details check them out here:

🧠 GRPO VRAM Requirements For the GPU Poor | Oxen.ai
Since the release of DeepSeek-R1, Group Relative Policy Optimization (GRPO) has become the talk of the town for Reinforcement Learning in Large Language Models due to its effectiveness and ease of training. The R1 paper demonstrated how you can use GRPO to go from a base instruction following LLM (DeepSeek-v3) to a reasoning model (DeepSeek-R1). To learn more about instruction following, reasoning models, and the full DeepSeek-R1 model, I suggest you checkout some of our other deep dives. How

Oxen.ai

If you understand where all the system requirements comes from, you can start to contribute to open source or help optimize your own libraries like this most recent PR I saw on the trl repo:


Group Relative Advantages
The main signal you are trying to get out of your LLMs during RL is represented by “A” which stands for the “Advantage”. This helps give direction to update the original LLM’s weights. If the the advantage is high, you want to encourage the model to keep doing the same actions. If it is low, you want to encourage the model to try something different.


In PPO, the value model’s original job is to try to estimate how good the tokens that are generated are, or how likely they are to give a high reward. In order to do this well, it required you to train large LLM to make these value judgements. So how does GRPO remove the need for this?

The first trick is that instead of generating one output per query, GRPO starts by generating multiple outputs.


Concretely if the question is a math question the model might try a few different approaches to solve it. For example, if the question is:

Mr. Curtis has 325 chickens on his farm where 28 are roosters and the rest are hens. Twenty hens do not lay eggs while the rest of the hens do. How many egg-laying hens does Mr. Curtis have on his farm?
The model might come up with a few different reasoning traces, some correct (answer=227), and some incorrect (answer=305).


Correct Output

<reasoning>First, let's find out how many hens there are. The total number of chickens is 325, and 28 are roosters. So, the number of hens is 325 - 28 = 297. Of these 297 hens, 20 do not lay eggs, so the number of egg-laying hens is 297 - 20 = 277.</reasoning>
<answer>277</answer>
Incorrect Output

<reasoning>You need to subtract the 20 hens that do not lay eggs from the total number of hens to find the number of egg-laying hens. So, the number of egg-laying hens is 325 - 20 = 305.</reasoning>
<answer>305</answer>
Then for each output, we calculate a “reward” for how well that output answers the query. There can be multiple reward functions that evaluate different properties of the response. We will leave the reward functions as a black box for now, but just know that they return some numeric value that is higher if the response is good, and lower if the response is bad.

Rewards may look like:

Formatting = 1.0
Answer = 0.0
Consistency = 0.5
Once we have our set of rewards (r) given our outputs, GRPO calculates our “advantage” (A) by simply looking at the mean and standard deviation of all the rewards.


This equation is pretty handy for feature engineering in machine learning in general. It helps normalize arbitrary values to more a more learnable positive or negative signal. The intuition is “how many standard deviations from the mean is the data point?”

Let’s look at some examples.

# o_0 = <reasoning>I have some reasoning</reasoning><answer>12</answer>
r_0 = 1.0

# o_1 = <reasoning></reasoning><answer>12</answer>
r_1 = 0.5

# o_2 = The answer is 312
r_2 = 0.0

# o_3 = <reason>I did not have valid formatting or answer.
r_3 = 0.0

In raw numpy it would look something like this:

import numpy as np

advantages = [(r_i - np.mean(r)) / np.std(r) for r_i in r]

Let’s try it with another set of numbers:

rewards = [4.0, 2.5, 0.5, 0.1]
advantage = [(r_i - np.mean(r)) / np.std(r) for r_i in r]
[1.4137674241360643, 0.4606657898870322, -0.8101363891116772, -1.064296824911419]
You’ll notice the values center around 0.0 and tell you how good or bad the score is relative to all the other ones. This gives us a sort of baseline of “given this prompt, how good are the average responses going to be?”. Reward the good outputs, and penalize the bad ones in this batch.

This is pretty similar to what the value model was originally trying to do: estimate what our reward will be given a response. Since the policy model we are training is a language model, we can just tweak the temperature and have generate multiple possible completions instead. Then the average reward for all these generations is a good signal for how well the current model is doing, and if we should reinforce the behavior.

KL-Divergence
The final piece of the equation is the KL Divergence term.


Without getting too deep into the math, this is why we have been keeping around a “reference model” during the training. The idea is that we do not want to drift too far from the original model. For each token, we want to make sure the new predictions do not drift too far from the original ones.


The intuition behind enforcing the KL Divergence is that the model we are starting with already knows how to write coherent sentences and follow instructions. We don’t want the new model to “reward hack” or exploit some sort of property in our reward signal that is not aligned with the original model. If it finds out that saying the word “pamplemousse” gets a high reward because it is a rarer word (and fun one to say) we don’t want it latching onto this behavior if it was not common in the pre-training.

Put all this together and you have this final equation!


Or as our trusty Ox Eric says…The math looks more complicated than it is…


The Reward Signals
What’s super interesting about the DeepSeek-R1-Zero work is that they go even further to slash the memory usage because don’t use a “neural reward model”.


What does this mean? It means they are literally using regexes and string matching for reward signals. They argue that this helps with “reward hacking” and simplifies the whole training pipeline.

If you took the definitions in Accuracy Rewards and Format Rewards sections above and turned it into code, it would look like this:


reference:

GRPO Llama-1B
GRPO Llama-1B. GitHub Gist: instantly share code, notes, and snippets.

Gist
262588213843476

You don’t even need a full reward model LLM in the loop during training. This leaves us with the policy model and the reference model as the main memory requirements. Dropping the need of 4 LLMs to 2 gives us the huge reduction in GPU requirements.

If your spidey senses are tingling and asking “do these reward function really generalize?” you would be right. They work well for whatever you specify in the rewards, but not much else. In other words, with these two rewards the model does get good at following the <reasoning></reasoning><answer></answer> format and does gets good at reasoning through math, but it fails at other useful tasks.


My prediction is that “The Bitter Lesson” will strike again here. Given enough compute and data, models just want to learn. The less we hand code rules, and more we let the model learn on it’s own, the better it will perform. GRPO rewards here feel a little to hand coded. Why wouldn’t you just have the model learn the weights of the reward signals?

That being said, it is kind of fun playing with different rewards. What’s cool about GRPO is that as long as you can define it in code as a function that returns a value given responses, you can optimize against it. This could even be an external API call to another LLM. I have a feeling in the coming weeks / month people are going to start getting creative with what rewards are fed into GRPO because it is so accessible to train.

💡
Want help implementing GRPO, RL, and other post-training techniques? Contact us if you would like to consult with our ML experts or use Oxen.ai to simplify fine-tuning.
Next Up: Training a Rust Reasoner 🦀
What does this look like in practice? The “Transformer Reinforcement Learning” or trl library already has GRPO implemented, and it’s relatively easy to pass in your own reward functions.

Some of the folks in our community are going to try to train a smol 1B param LLM optimized for Rust. I think there’s a lot of juice we could squeeze out of low resource programming languages to get models we could run locally.

The idea would be this:


We have already started collecting a dataset in an oxen repository that is a combination of synthetic data and stack overflow questions that we plan on training on.

ox/Rust/filtered_data.jsonl at main
This is a dataset of rust questions and generated code created to fine tune small language models on rust.. Contribute to the ox/Rust repository by creating an account on Oxen.ai


If you’d like to join us, let us know in the Discord!

If you are working on fine-tuning LLMs and want to simplify your workflow, check out Oxen.ai...or reach out and we could fine-tune for you if you'd like. We’re happy to bring our expertise to your use case and give you guidance bringing AI in your product from idea to reality. You could also join our community of over 1,300+ AI enthusiasts, engineers, and researchers to ask any fine-tuning questions.

Published by:
Greg Schoeninger
Greg Schoeninger
You might also like...
Apr
15
How RWKV-7 Goose Works 🪿 + Notes from the Author
17 min read
Mar
25
How Phi-4 Cracked Small Multimodality
8 min read
Feb
04
How DeepSeek R1, GRPO, and Previous DeepSeek Models Work
15 min read
Jan
29
No Hype DeepSeek-R1 Reading List
27 min read
Jan
21
arXiv Dive: RAGAS - Retrieval Augmented Generation Assessment
13 min read
Member discussion
1 comment

Oxen.ai © 2025
Powered by Ghost
HTML conversions sometimes display errors due to content that did not convert correctly from the source. This paper uses the following packages that are not yet supported by the HTML conversion tool. Feedback on these issues are not necessary; they are known and are being worked on.

failed: deepseek.cls
failed: datetime.sty
failed: mdframed.sty
failed: dramatist.sty
failed: xltabular.sty
Authors: achieve the best HTML results from your LaTeX submissions by following these best practices.

License: arXiv.org perpetual non-exclusive license
arXiv:2402.03300v3 [cs.CL] 27 Apr 2024
\reportnumber
001 \correspondingauthor  ∗ Core contributors.
  † Work done during internship at DeepSeek-AI.

DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
   Zhihong Shao1,2∗†
Peiyi Wang1,3∗†
Qihao Zhu1,3∗†
Runxin Xu1
Junxiao Song1
Xiao Bi1
Haowei Zhang1
Mingchuan Zhang1
Y.K. Li1
Y. Wu1
Daya Guo1∗ 1DeepSeek-AI
2Tsinghua University
3Peking University
{zhihongshao,wangpeiyi,zhuqh,guoday}@deepseek.com
https://github.com/deepseek-ai/DeepSeek-Math
Abstract
Mathematical reasoning poses a significant challenge for language models due to its complex and structured nature. In this paper, we introduce DeepSeekMath 7B, which continues pre-training DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens sourced from Common Crawl, together with natural language and code data. DeepSeekMath 7B has achieved an impressive score of 51.7% on the competition-level MATH benchmark without relying on external toolkits and voting techniques, approaching the performance level of Gemini-Ultra and GPT-4. Self-consistency over 64 samples from DeepSeekMath 7B achieves 60.9% on MATH. The mathematical reasoning capability of DeepSeekMath is attributed to two key factors: First, we harness the significant potential of publicly available web data through a meticulously engineered data selection pipeline. Second, we introduce Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO.

Refer to caption
Figure 1: Top1 accuracy of open-source models on the competition-level MATH benchmark (Hendrycks et al., 2021) without the use of external toolkits and voting techniques.
1Introduction
Large language models (LLM) have revolutionized the approach to mathematical reasoning in artificial intelligence, spurring significant advancements in both the quantitative reasoning benchmark (Hendrycks et al., 2021) and the geometry reasoning benchmark (Trinh et al., 2024). Moreover, these models have proven instrumental in assisting humans in solving complex mathematical problems (Tao, 2023). However, cutting-edge models such as GPT-4 (OpenAI, 2023) and Gemini-Ultra (Anil et al., 2023) are not publicly available, and the currently accessible open-source models considerably trail behind in performance.

In this study, we introduce DeepSeekMath, a domain-specific language model that significantly outperforms the mathematical capabilities of open-source models and approaches the performance level of GPT-4 on academic benchmarks. To achieve this, we create the DeepSeekMath Corpus, a large-scale high-quality pre-training corpus comprising 120B math tokens. This dataset is extracted from the Common Crawl (CC) using a fastText-based classifier (Joulin et al., 2016). In the initial iteration, the classifier is trained using instances from OpenWebMath (Paster et al., 2023) as positive examples, while incorporating a diverse selection of other web pages to serve as negative examples. Subsequently, we employ the classifier to mine additional positive instances from the CC, which are further refined through human annotation. The classifier is then updated with this enhanced dataset to improve its performance. The evaluation results indicate that the large-scale corpus is of high quality, as our base model DeepSeekMath-Base 7B achieves 64.2% on GSM8K (Cobbe et al., 2021) and 36.2% on the competition-level MATH dataset (Hendrycks et al., 2021), outperforming Minerva 540B (Lewkowycz et al., 2022a). In addition, the DeepSeekMath Corpus is multilingual, so we notice an improvement in Chinese mathematical benchmarks (Wei et al., 2023; Zhong et al., 2023). We believe that our experience in mathematical data processing is a starting point for the research community, and there is significant room for improvement in the future.

DeepSeekMath-Base is initialized with DeepSeek-Coder-Base-v1.5 7B (Guo et al., 2024), as we notice that starting from a code training model is a better choice compared to a general LLM. Furthermore, we observe the math training also improves model capability on MMLU (Hendrycks et al., 2020) and BBH benchmarks (Suzgun et al., 2022), indicating it does not only enhance the model’s mathematical abilities but also amplifies general reasoning capabilities.

After pre-training, we apply mathematical instruction tuning to DeepSeekMath-Base with chain-of-thought (Wei et al., 2022), program-of-thought (Chen et al., 2022; Gao et al., 2023), and tool-integrated reasoning (Gou et al., 2023) data. The resulting model DeepSeekMath-Instruct 7B beats all 7B counterparts and is comparable with 70B open-source instruction-tuned models.

Furthermore, we introduce the Group Relative Policy Optimization (GRPO), a variant reinforcement learning (RL) algorithm of Proximal Policy Optimization (PPO) (Schulman et al., 2017). GRPO foregoes the critic model, instead estimating the baseline from group scores, significantly reducing training resources. By solely using a subset of English instruction tuning data, GRPO obtains a substantial improvement over the strong DeepSeekMath-Instruct, including both in-domain (GSM8K: 82.9% 
→
 88.2%, MATH: 46.8% 
→
 51.7%) and out-of-domain mathematical tasks (e.g., CMATH: 84.6% 
→
 88.8%) during the reinforcement learning phase. We also provide a unified paradigm to understand different methods, such as Rejection Sampling Fine-Tuning (RFT) (Yuan et al., 2023a), Direct Preference Optimization (DPO) (Rafailov et al., 2023), PPO and GRPO. Based on such a unified paradigm, we find that all these methods are conceptualized as either direct or simplified RL techniques. We also conduct extensive experiments, e.g., online v.s. offline training, outcome v.s. process supervision, single-turn v.s. iterative RL and so on, to deeply investigate the essential elements of this paradigm. At last, we explain why our RL boosts the performance of instruction-tuned models, and further summarize potential directions to achieve more effective RL based on this unified paradigm.

1.1Contributions
Our contribution includes scalable math pre-training, along with the exploration and analysis of reinforcement learning.

Math Pre-Training at Scale

• Our research provides compelling evidence that the publicly accessible Common Crawl data contains valuable information for mathematical purposes. By implementing a meticulously designed data selection pipeline, we successfully construct the DeepSeekMath Corpus, a high-quality dataset of 120B tokens from web pages filtered for mathematical content, which is almost 7 times the size of the math web pages used by Minerva (Lewkowycz et al., 2022a) and 9 times the size of the recently released OpenWebMath (Paster et al., 2023).
• Our pre-trained base model DeepSeekMath-Base 7B achieves comparable performance with Minerva 540B (Lewkowycz et al., 2022a), indicating the number of parameters is not the only key factor in mathematical reasoning capability. A smaller model pre-trained on high-quality data could achieve strong performance as well.
• We share our findings from math training experiments. Code training prior to math training improves models’ ability to solve mathematical problems both with and without tool use. This offers a partial answer to the long-standing question: does code training improve reasoning abilities? We believe it does, at least for mathematical reasoning.
• Although training on arXiv papers is common, especially in many math-related papers, it brings no notable improvements on all mathematical benchmarks adopted in this paper.
Exploration and Analysis of Reinforcement Learning

• We introduce Group Relative Policy Optimization (GRPO), an efficient and effective reinforcement learning algorithm. GRPO foregoes the critic model, instead estimating the baseline from group scores, significantly reducing training resources compared to Proximal Policy Optimization (PPO).
• We demonstrate that GRPO significantly enhances the performance of our instruction-tuned model DeepSeekMath-Instruct, by solely using the instruction-tuning data. Furthermore, we observe enhancements in the out-of-domain performance during the reinforcement learning process.
• We provide a unified paradigm to understand different methods, such as RFT, DPO, PPO, and GRPO. We also conduct extensive experiments, e.g., online v.s. offline training, outcome v.s. process supervision, single-turn v.s. iterative reinforcement learning, and so on to deeply investigate the essential elements of this paradigm.
• Based on our unified paradigm, we explore the reasons behind the effectiveness of reinforcement learning, and summarize several potential directions to achieve more effective reinforcement learning of LLMs.
1.2Summary of Evaluations and Metrics
• English and Chinese Mathematical Reasoning: We conduct comprehensive assessments of our models on English and Chinese benchmarks, covering mathematical problems from grade-school level to college level. English benchmarks include GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021), SAT (Azerbayev et al., 2023), OCW Courses (Lewkowycz et al., 2022a), MMLU-STEM (Hendrycks et al., 2020). Chinese benchmarks include MGSM-zh (Shi et al., 2023), CMATH (Wei et al., 2023), Gaokao-MathCloze (Zhong et al., 2023), and Gaokao-MathQA (Zhong et al., 2023). We evaluate models’ ability to generate self-contained text solutions without tool use, and also the ability to solve problems using Python.
On English benchmarks, DeepSeekMath-Base is competitive with the closed-source Minerva 540B (Lewkowycz et al., 2022a), and surpasses all open-source base models (e.g., Mistral 7B (Jiang et al., 2023) and Llemma-34B (Azerbayev et al., 2023)), regardless of whether they’ve undergone math pre-training or not, often by a significant margin. Notably, DeepSeekMath-Base is superior on Chinese benchmarks, likely because we don’t follow previous works (Lewkowycz et al., 2022a; Azerbayev et al., 2023) to collect English-only math pre-training data, and also include high-quality non-English ones. With mathematical instruction tuning and reinforcement learning, the resulting DeepSeekMath-Instruct and DeepSeekMath-RL demonstrate strong performance, obtaining an accuracy of over 50% on the competition-level MATH dataset for the first time within the open-source community.

• Formal Mathematics: We evaluate DeepSeekMath-Base using the informal-to-formal theorem proving task from (Jiang et al., 2022) on miniF2F (Zheng et al., 2021) with Isabelle (Wenzel et al., 2008) chosen to be the proof assistant. DeepSeekMath-Base demonstrates strong few-shot autoformalization performance.
• Natural Language Understanding, Reasoning, and Code: To build a comprehensive profile of models’ general understanding, reasoning, and coding capabilities, we evaluate DeepSeekMath-Base on the Massive Multitask Language Understanding (MMLU) benchmark (Hendrycks et al., 2020) which encompasses 57 multiple-choice tasks covering diverse subjects, BIG-Bench Hard (BBH) (Suzgun et al., 2022) which consists of 23 challenging tasks that mostly require multi-step reasoning to solve, as well as HumanEval (Chen et al., 2021) and MBPP (Austin et al., 2021) which are widely used to evaluate code language models. Math pre-training benefits both language understanding and reasoning performance.
2Math Pre-Training
2.1Data Collection and Decontamination
In this section, we will outline the process of constructing the DeepSeekMath Corpus from Common Crawl. As depicted in Figure 2, we present an iterative pipeline that demonstrates how to systematically gather a large-scale mathematical corpus from Common Crawl, starting with a seed corpus (e.g., a small but high-quality collection of math-related dataset). It’s worth noting that this approach is also applicable to other domains, such as coding.

Refer to caption
Figure 2:An iterative pipeline that collects mathematical web pages from Common Crawl.
First, we choose OpenWebMath (Paster et al., 2023), a collection of high-quality mathematical web texts, as our initial seed corpus. Using this corpus, we train a fastText model (Joulin et al., 2016) to recall more OpenWebMath-like mathematical web pages. Specifically, we randomly select 500,000 data points from the seed corpus as positive training examples and another 500,000 web pages from Common Crawl as negative ones. We employ an open-source library1
1https://fasttext.cc
 for training, configuring the vector dimension to 256, learning rate to 0.1, the maximum length of word n-gram to 3, the minimum number of word occurrences to 3, and the number of training epochs to 3. To reduce the size of the original Common Crawl, we employ URL-based deduplication and near-deduplication techniques, resulting in 40B HTML web pages. We then recall mathematical web pages from deduplicated Common Crawl with the fastText model. To filter out low-quality mathematical content, we rank the collected pages according to their scores predicted by the fastText model, and only preserve the top-ranking ones. The volume of data preserved is assessed through pre-training experiments on the top 40B, 80B, 120B, and 160B tokens. In the first iteration, we choose to keep the top 40B tokens.

After the first iteration of data collection, numerous mathematical web pages remain uncollected, mainly because the fastText model is trained on a set of positive examples that lacks sufficient diversity. We therefore identify additional mathematical web sources to enrich the seed corpus, so that we can optimize the fastText model. Specifically, we first organize the entire Common Crawl into disjoint domains; a domain is defined as web pages sharing the same base URL. For each domain, we calculate the percentage of web pages that are collected in the first iteration. Domains where over 10% of the web pages have been collected are classified as math-related (e.g., mathoverflow.net). Subsequently, we manually annotate the URLs associated with mathematical content within these identified domains (e.g., mathoverflow.net/questions). Web pages linked to these URLs, yet uncollected, will be added to the seed corpus. This approach enables us to gather more positive examples, thereby training an improved fastText model capable of recalling more mathematical data in the subsequent iteration. After four iterations of data collection, we end up with 35.5M mathematical web pages, totaling 120B tokens. In the fourth iteration, we notice that nearly 98% of the data has already been collected in the third iteration, so we decide to cease data collection.

To avoid benchmark contamination, we follow Guo et al. (2024) to filter out web pages containing questions or answers from English mathematical benchmarks such as GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021) and Chinese benchmarks such as CMATH (Wei et al., 2023) and AGIEval (Zhong et al., 2023). The filtering criteria are as follows: any text segment containing a 10-gram string that matches exactly with any sub-string from the evaluation benchmarks is removed from our math training corpus. For benchmark texts that are shorter than 10 grams but have at least 3 grams, we employ exact matching to filter out contaminated web pages.

2.2Validating the Quality of the DeepSeekMath Corpus
We run pre-training experiments to investigate how the DeepSeekMath Corpus is compared with the recently released math-training corpora:

• MathPile (Wang et al., 2023c): a multi-source corpus (8.9B tokens) aggregated from textbooks, Wikipedia, ProofWiki, CommonCrawl, StackExchange, and arXiv, with the majority (over 85%) sourced from arXiv;
• OpenWebMath (Paster et al., 2023): CommonCrawl data filtered for mathematical content, totaling 13.6B tokens;
• Proof-Pile-2 (Azerbayev et al., 2023): a mathematical corpus consisting of OpenWebMath, AlgebraicStack (10.3B tokens of mathematical code), and arXiv papers (28.0B tokens). When experimenting on Proof-Pile-2, we follow Azerbayev et al. (2023) to use an arXiv:Web:Code ratio of 2:4:1.
2.2.1Training Setting
We apply math training to a general pre-trained language model with 1.3B parameters, which shares the same framework as the DeepSeek LLMs (DeepSeek-AI, 2024), denoted as DeepSeek-LLM 1.3B. We separately train a model on each mathematical corpus for 150B tokens. All experiments are conducted using the efficient and light-weight HAI-LLM (High-flyer, 2023) training framework. Following the training practice of DeepSeek LLMs, we use the AdamW optimizer (Loshchilov and Hutter, 2017) with 
β
1
=
0.9
, 
β
2
=
0.95
, and 
weight
​
_
​
decay
=
0.1
, along with a multi-step learning rate schedule where the learning rate reaches the peak after 2,000 warmup steps, decreases to its 31.6% after 80% of the training process, and further decreases to 10.0% of the peak after 90% of the training process. We set the maximum value of learning rate to 5.3e-4, and use a batch size of 4M tokens with a 4K context length.

Math Corpus	Size	English Benchmarks	Chinese Benchmarks
GSM8K	MATH	OCW	SAT	 
MMLU
STEM
 	CMATH	 
Gaokao
MathCloze
 	 
Gaokao
MathQA
 
No Math Training	N/A	2.9%	3.0%	2.9%	15.6%	19.5%	12.3%	0.8%	17.9%
MathPile	8.9B	2.7%	3.3%	2.2%	12.5%	15.7%	1.2%	0.0%	2.8%
OpenWebMath	13.6B	11.5%	8.9%	3.7%	31.3%	29.6%	16.8%	0.0%	14.2%
Proof-Pile-2	51.9B	14.3%	11.2%	3.7%	43.8%	29.2%	19.9%	5.1%	11.7%
DeepSeekMath Corpus	120.2B	23.8%	13.6%	4.8%	56.3%	33.1%	41.5%	5.9%	23.6%
 

Table 1: Performance of DeepSeek-LLM 1.3B trained on different mathematical corpora, evaluated using few-shot chain-of-thought prompting. Corpus sizes are calculated using our tokenizer with a vocabulary size of 100K.
Refer to caption
Figure 3:Benchmark curves of DeepSeek-LLM 1.3B trained on different mathematical corpora.
2.2.2Evaluation Results
The DeepSeekMath Corpus is of high quality, covers multilingual mathematical content, and is the largest in size.

• High-quality: We evaluate downstream performance on 8 mathematical benchmarks using few-shot chain-of-thought prompting Wei et al. (2022). As shown in Table 1, there is a clear performance lead of the model trained on the DeepSeekMath Corpus. Figure 3 shows that the model trained on the DeepSeekMath Corpus demonstrates better performance than Proof-Pile-2 at 50B tokens (1 full epoch of Proof-Pile-2), indicating the average quality of DeepSeekMath Corpus is higher.
• Multilingual: The DeepSeekMath Corpus encompasses data in multiple languages, predominantly featuring English and Chinese as the two most represented languages. As shown in Table 1, training on the DeepSeekMath Corpus enhances mathematical reasoning performance in both English and Chinese. In contrast, existing mathematical corpora, which are primarily English-centric, show limited improvement and may even hinder performance in Chinese mathematical reasoning.
• Large-scale: The DeepSeekMath Corpus is several times larger than existing mathematical corpora. As depicted in Figure 3, DeepSeek-LLM 1.3B, when trained on the DeepSeekMath Corpus, shows a steeper learning curve along with more lasting improvements. In contrast, the baseline corpora are much smaller, and have already been repeated multiple rounds during training, with the resulting model performance quickly reaching a plateau.
2.3Training and Evaluating DeepSeekMath-Base 7B
In this section, we introduce DeepSeekMath-Base 7B, a base model with strong reasoning abilities, especially in mathematics. Our model is initialized with DeepSeek-Coder-Base-v1.5 7B (Guo et al., 2024) and trained for 500B tokens. The distribution of the data is as follows: 56% is from the DeepSeekMath Corpus, 4% from AlgebraicStack, 10% from arXiv, 20% is Github code, and the remaining 10% is natural language data from Common Crawl in both English and Chinese. We mainly adopt the training setting specified in Section 2.2.1, except that we set the maximum value of the learning rate to 4.2e-4 and use a batch size of 10M tokens.

We conduct a comprehensive assessment of the mathematical capabilities of DeepSeekMath-Base 7B, focusing on its ability to produce self-contained mathematical solutions without relying on external tools, solve mathematical problems using tools, and conduct formal theorem proving. Beyond mathematics, we also provide a more general profile of the base model, including its performance of natural language understanding, reasoning, and programming skills.

Mathematical Problem Solving with Step-by-Step Reasoning
We evaluate DeepSeekMath-Base’s performance of solving mathematical problems using few-shot chain-of-thought prompting (Wei et al., 2022), across eight benchmarks in English and Chinese. These benchmarks encompass quantitative reasoning (e.g., GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021), and CMATH (Wei et al., 2023)) and multiple-choice problems (e.g., MMLU-STEM (Hendrycks et al., 2020) and Gaokao-MathQA (Zhong et al., 2023)), covering diverse fields of mathematics from elementary to college-level complexity.

As shown in Table 2, DeepSeekMath-Base 7B leads in performance across all eight benchmarks among the open-source base models (including the widely-used general model Mistral 7B (Jiang et al., 2023) and the recently released Llemma 34B (Azerbayev et al., 2023) which underwent math training on Proof-Pile-2 (Azerbayev et al., 2023)). Notably, on the competition-level MATH dataset, DeepSeekMath-Base surpasses existing open-source base models by over 10% absolute, and outperforms Minerva 540B (Lewkowycz et al., 2022a), a closed-source base model 77 times larger which builds on PaLM (Lewkowycz et al., 2022b) and is further trained on mathematical texts.

Model	Size	English Benchmarks	Chinese Benchmarks
GSM8K	MATH	OCW	SAT	 
MMLU
STEM
 	CMATH	 
Gaokao
MathCloze
 	 
Gaokao
MathQA
 
Closed-Source Base Model
Minerva	7B	16.2%	14.1%	7.7%	-	35.6%	-	-	-
Minerva	62B	52.4%	27.6%	12.0%	-	53.9%	-	-	-
Minerva	540B	58.8%	33.6%	17.6%	-	63.9%	-	-	-
Open-Source Base Model
Mistral	7B	40.3%	14.3%	9.2%	71.9%	51.1%	44.9%	5.1%	23.4%
Llemma	7B	37.4%	18.1%	6.3%	59.4%	43.1%	43.4%	11.9%	23.6%
Llemma	34B	54.0%	25.3%	10.3%	71.9%	52.9%	56.1%	11.9%	26.2%
DeepSeekMath-Base	7B	64.2%	36.2%	15.4%	84.4%	56.5%	71.7%	20.3%	35.3%
 

Table 2: Comparisons between DeepSeekMath-Base 7B and strong base models on English and Chinese mathematical benchmarks. Models are evaluated with chain-of-thought prompting. Minerva results are quoted from Lewkowycz et al. (2022a).
Mathematical Problem Solving with Tool Use
We evaluate program-aided mathematical reasoning on GSM8K and MATH using few-shot program-of-thought prompting (Chen et al., 2022; Gao et al., 2023). Models are prompted to solve each problem by writing a Python program where libraries such as math and sympy can be utilized for intricate computations. The execution result of the program is evaluated as the answer. As shown in Table 3, DeepSeekMath-Base 7B outperforms the prior state-of-the-art Llemma 34B.

Model	Size	Problem Solving w/ Tools	Informal-to-Formal Proving
GSM8K+Python	MATH+Python	miniF2F-valid	miniF2F-test
Mistral	7B	48.5%	18.2%	18.9%	18.0%
CodeLlama	7B	27.1%	17.2%	16.3%	17.6%
CodeLlama	34B	52.7%	23.5%	18.5%	18.0%
Llemma	7B	41.0%	18.6%	20.6%	22.1%
Llemma	34B	64.6%	26.3%	21.0%	21.3%
DeepSeekMath-Base	7B	66.9%	31.4%	25.8%	24.6%
Table 3: Few-shot evaluation of base models’ ability to solve mathematical problems using tools and the ability to conduct informal-to-formal theorem proving in Isabelle.
Formal Mathematics
Formal proof automation is beneficial to ensure the accuracy and reliability of mathematical proofs and enhance efficiency, with increasing attention in recent years. We evaluate DeepSeekMath-Base 7B on the task of informal-to-formal proving from (Jiang et al., 2022) which is to generate a formal proof based on an informal statement, a formal counterpart of the statement, and an informal proof. We evaluate on miniF2F (Zheng et al., 2021), a benchmark for formal Olympiad-level mathematics, and generate a formal proof in Isabelle for each problem with few-shot prompting. Following Jiang et al. (2022), we leverage models to generate proof sketches, and execute the off-the-shelf automated prover Sledgehammer (Paulson, 2010) to fill in the missing details. As shown in Table 3, DeepSeekMath-Base 7B demonstrates strong performance in proof autoformalization.

Model	Size	MMLU	BBH	HumanEval (Pass@1)	MBPP (Pass@1)
Mistral	7B	62.4%	55.7%	28.0%	41.4%
DeepSeek-Coder-Base-v1.5
†
 	7B	42.9%	42.9%	40.2%	52.6%
DeepSeek-Coder-Base-v1.5	7B	49.1%	55.2%	43.2%	60.4%
DeepSeekMath-Base	7B	54.9%	59.5%	40.9%	52.6%
Table 4: Evaluation on natural language understanding, reasoning, and code benchmarks. DeepSeek-Coder-Base-v1.5
†
 is the checkpoint right before learning rate decay, which is used to train DeepSeekMath-Base. On MMLU and BBH, we use few-shot chain-of-thought prompting. On HumanEval and MBPP, we evaluate model performance under the zero-shot setting and a few-shot setting, respectively.
Natural Language Understanding, Reasoning, and Code
We evaluate model performance of natural language understanding on MMLU (Hendrycks et al., 2020), reasoning on BBH (Suzgun et al., 2022), and coding capabilities on HumanEval (Chen et al., 2021) and MBPP (Austin et al., 2021). As shown in Table 4, DeepSeekMath-Base 7B exhibits significant enhancements in performance on MMLU and BBH over its precursor, DeepSeek-Coder-Base-v1.5 (Guo et al., 2024), illustrating the positive impact of math training on language understanding and reasoning. Additionally, by including code tokens for continual training, DeepSeekMath-Base 7B effectively maintains the performance of DeepSeek-Coder-Base-v1.5 on the two coding benchmarks. Overall, DeepSeekMath-Base 7B significantly outperforms the general model Mistral 7B (Jiang et al., 2023) on the three reasoning and coding benchmarks.

3Supervised Fine-Tuning
3.1SFT Data Curation
We construct a mathematical instruction-tuning dataset covering English and Chinese problems from different mathematical fields and of varying complexity levels: problems are paired with solutions in chain-of-thought (CoT) (Wei et al., 2022), program-of-thought (PoT) (Chen et al., 2022; Gao et al., 2023), and tool-integrated reasoning format (Gou et al., 2023). The total number of training examples is 776K.

• English mathematical datasets: We annotate GSM8K and MATH problems with tool-integrated solutions, and adopt a subset of MathInstruct (Yue et al., 2023) along with the training set of Lila-OOD (Mishra et al., 2022) where problems are solved with CoT or PoT. Our English collection covers diverse fields of mathematics, e.g., algebra, probability, number theory, calculus, and geometry.
• Chinese mathematical datasets: We collect Chinese K-12 mathematical problems spanning 76 sub-topics such as linear equations, with solutions annotated in both CoT and tool-integrated reasoning format.
3.2Training and Evaluating DeepSeekMath-Instruct 7B
In this section, we introduce DeepSeekMath-Instruct 7B which undergoes mathematical instruction tuning based on DeepSeekMath-Base. Training examples are randomly concatenated until reaching a maximum context length of 4K tokens. We train the model for 500 steps with a batch size of 256 and a constant learning rate of 5e-5.

We evaluate models’ mathematical performance both without and with tool use, on 4 quantitative reasoning benchmarks in English and Chinese. We benchmark our model against the leading models of the time:

• Closed-source models include: (1) the GPT family among which GPT-4 (OpenAI, 2023) and GPT-4 Code Interpreter 2
2https://openai.com/blog/chatgpt-plugins##code-interpreter
 are the most capable ones, (2) Gemini Ultra and Pro (Anil et al., 2023), (3) Inflection-2 (Inflection AI, 2023), (4) Grok-1 3
3https://x.ai/model-card
, as well as models recently released by Chinese companies including (5) Baichuan-3 4
4https://www.baichuan-ai.com
, (6) the latest GLM-4 5
5https://open.bigmodel.cn/dev/api#glm-4
 from the GLM family (Du et al., 2022). These models are for general purposes, most of which have undergone a series of alignment procedures.
• Open-source models include: general models like (1) DeepSeek-LLM-Chat 67B (DeepSeek-AI, 2024), (2) Qwen 72B (Bai et al., 2023), (3) SeaLLM-v2 7B (Nguyen et al., 2023), and (4) ChatGLM3 6B (ChatGLM3 Team, 2023), as well as models with enhancements in mathematics including (5) InternLM2-Math 20B 6
6https://github.com/InternLM/InternLM-Math
 which builds on InternLM2 and underwent math training followed by instruction tuning, (6) Math-Shepherd-Mistral 7B which applys PPO training (Schulman et al., 2017) to Mistral 7B (Jiang et al., 2023) with a process-supervised reward model, (7) the WizardMath series (Luo et al., 2023) which improves mathematical reasoning in Mistral 7B and Llama-2 70B (Touvron et al., 2023) using evolve-instruct (i.e., a version of instruction tuning that uses AI-evolved instructions) and PPO training with training problems primarily sourced from GSM8K and MATH, (8) MetaMath 70B (Yu et al., 2023) which is Llama-2 70B fine-tuned on an augmented version of GSM8K and MATH, (9) ToRA 34B Gou et al. (2023) which is CodeLlama 34B fine-tuned to do tool-integrated mathematical reasoning, (10) MAmmoTH 70B (Yue et al., 2023) which is Llama-2 70B instruction-tuned on MathInstruct.
Model	Size	English Benchmarks	Chinese Benchmarks
GSM8K	MATH	MGSM-zh	CMATH
Chain-of-Thought Reasoning
Closed-Source Model
Gemini Ultra	-	94.4%	53.2%	-	-
GPT-4	-	92.0%	52.9%	-	86.0%
Inflection-2	-	81.4%	34.8%	-	-
GPT-3.5	-	80.8%	34.1%	-	73.8%
Gemini Pro	-	86.5%	32.6%	-	-
Grok-1	-	62.9%	23.9%	-	-
Baichuan-3	-	88.2%	49.2%	-	-
GLM-4	-	87.6%	47.9%	-	-
Open-Source Model
InternLM2-Math	20B	82.6%	37.7%	-	-
Qwen	72B	78.9%	35.2%	-	-
Math-Shepherd-Mistral	7B	84.1%	33.0%	-	-
WizardMath-v1.1	7B	83.2%	33.0%	-	-
DeepSeek-LLM-Chat	67B	84.1%	32.6%	74.0%	80.3%
MetaMath	70B	82.3%	26.6%	66.4%	70.9%
SeaLLM-v2	7B	78.2%	27.5%	64.8%	-
ChatGLM3	6B	72.3%	25.7%	-	-
WizardMath-v1.0	70B	81.6%	22.7%	64.8%	65.4%
DeepSeekMath-Instruct	7B	82.9%	46.8%	73.2%	84.6%
DeepSeekMath-RL	7B	88.2%	51.7%	79.6%	88.8%
Tool-Integrated Reasoning
Closed-Source Model
GPT-4 Code Interpreter	-	97.0%	69.7%	-	-
Open-Source Model
InternLM2-Math	20B	80.7%	54.3%	-	-
DeepSeek-LLM-Chat	67B	86.7%	51.1%	76.4%	85.4%
ToRA	34B	80.7%	50.8%	41.2%	53.4%
MAmmoTH	70B	76.9%	41.8%	-	-
DeepSeekMath-Instruct	7B	83.7%	57.4%	72.0%	84.3%
DeepSeekMath-RL	7B	86.7%	58.8%	78.4%	87.6%
 

Table 5: Performance of Open- and Closed-Source models with both Chain-of-Thought and Tool-Integrated Reasoning on English and Chinese Benchmarks. Scores in gray denote majority votes with 32 candidates; The others are Top1 scores. DeepSeekMath-RL 7B beats all open-source models from 7B to 70B, as well as the majority of closed-source models. Although DeepSeekMath-RL 7B is only further trained on chain-of-thought-format instruction tuning data of GSM8K and MATH, it improves over DeepSeekMath-Instruct 7B on all benchmarks.
As shown in Table 5, under the evaluation setting where tool use is disallowed, DeepSeekMath-Instruct 7B demonstrates strong performance of step-by-step reasoning. Notably, on the competition-level MATH dataset, our model surpasses all open-source models and the majority of proprietary models (e.g., Inflection-2 and Gemini Pro) by at least 9% absolute. This is true even for models that are substantially larger (e.g., Qwen 72B) or have been specifically enhanced through math-focused reinforcement learning (e.g., WizardMath-v1.1 7B). While DeepSeekMath-Instruct rivals the Chinese proprietary models GLM-4 and Baichuan-3 on MATH, it still underperforms GPT-4 and Gemini Ultra.

Under the evaluation setting where models are allowed to integrate natural language reasoning and program-based tool use for problem solving, DeepSeekMath-Instruct 7B approaches an accuracy of 60% on MATH, surpassing all existing open-source models. On the other benchmarks, our model is competitive with DeepSeek-LLM-Chat 67B, the prior state-of-the-art that is 10 times larger.

4Reinforcement Learning
4.1Group Relative Policy Optimization
Reinforcement learning (RL) has been proven to be effective in further improving the mathematical reasoning ability of LLMs after the Supervised Fine-Tuning (SFT) stage (Wang et al., 2023b; Luo et al., 2023). In this section, we introduce our efficient and effective RL algorithm, Group Relative Policy Optimization (GRPO).

4.1.1From PPO to GRPO
Proximal Policy Optimization (PPO) (Schulman et al., 2017) is an actor-critic RL algorithm that is widely used in the RL fine-tuning stage of LLMs (Ouyang et al., 2022). In particular, it optimizes LLMs by maximizing the following surrogate objective:

𝒥
P
​
P
​
O
​
(
θ
)
=
𝔼
​
[
q
∼
P
​
(
Q
)
,
o
∼
π
θ
o
​
l
​
d
​
(
O
|
q
)
]
​
1
|
o
|
​
∑
t
=
1
|
o
|
min
⁡
[
π
θ
​
(
o
t
|
q
,
o
<
t
)
π
θ
o
​
l
​
d
​
(
o
t
|
q
,
o
<
t
)
​
A
t
,
clip
​
(
π
θ
​
(
o
t
|
q
,
o
<
t
)
π
θ
o
​
l
​
d
​
(
o
t
|
q
,
o
<
t
)
,
1
−
ε
,
1
+
ε
)
​
A
t
]
,
(1)
where 
π
θ
 and 
π
θ
o
​
l
​
d
 are the current and old policy models, and 
q
,
o
 are questions and outputs sampled from the question dataset and the old policy 
π
θ
o
​
l
​
d
, respectively. 
ε
 is a clipping-related hyper-parameter introduced in PPO for stabilizing training. 
A
t
 is the advantage, which is computed by applying Generalized Advantage Estimation (GAE) (Schulman et al., 2015), based on the rewards 
{
r
≥
t
}
 and a learned value function 
V
ψ
. Thus, in PPO, a value function needs to be trained alongside the policy model and to mitigate over-optimization of the reward model, the standard approach is to add a per-token KL penalty from a reference model in the reward at each token (Ouyang et al., 2022), i.e.,

r
t
=
r
φ
​
(
q
,
o
≤
t
)
−
β
​
log
⁡
π
θ
​
(
o
t
|
q
,
o
<
t
)
π
r
​
e
​
f
​
(
o
t
|
q
,
o
<
t
)
,
(2)
where 
r
φ
 is the reward model, 
π
r
​
e
​
f
 is the reference model, which is usually the initial SFT model, and 
β
 is the coefficient of the KL penalty.

Refer to caption
Figure 4:Demonstration of PPO and our GRPO. GRPO foregoes the value model, instead estimating the baseline from group scores, significantly reducing training resources.
As the value function employed in PPO is typically another model of comparable size as the policy model, it brings a substantial memory and computational burden. Additionally, during RL training, the value function is treated as a baseline in the calculation of the advantage for variance reduction. While in the LLM context, usually only the last token is assigned a reward score by the reward model, which may complicate the training of a value function that is accurate at each token. To address this, as shown in Figure 4, we propose Group Relative Policy Optimization (GRPO), which obviates the need for additional value function approximation as in PPO, and instead uses the average reward of multiple sampled outputs, produced in response to the same question, as the baseline. More specifically, for each question 
q
, GRPO samples a group of outputs 
{
o
1
,
o
2
,
⋯
,
o
G
}
 from the old policy 
π
θ
o
​
l
​
d
 and then optimizes the policy model by maximizing the following objective:

𝒥
G
​
R
​
P
​
O
​
(
θ
)
=
𝔼
​
[
q
∼
P
​
(
Q
)
,
{
o
i
}
i
=
1
G
∼
π
θ
o
​
l
​
d
​
(
O
|
q
)
]
1
G
∑
i
=
1
G
1
|
o
i
|
∑
t
=
1
|
o
i
|
{
min
[
π
θ
​
(
o
i
,
t
|
q
,
o
i
,
<
t
)
π
θ
o
​
l
​
d
​
(
o
i
,
t
|
q
,
o
i
,
<
t
)
A
^
i
,
t
,
clip
(
π
θ
​
(
o
i
,
t
|
q
,
o
i
,
<
t
)
π
θ
o
​
l
​
d
​
(
o
i
,
t
|
q
,
o
i
,
<
t
)
,
1
−
ε
,
1
+
ε
)
A
^
i
,
t
]
−
β
𝔻
K
​
L
[
π
θ
|
|
π
r
​
e
​
f
]
}
,
(3)
where 
ε
 and 
β
 are hyper-parameters, and 
A
^
i
,
t
 is the advantage calculated based on relative rewards of the outputs inside each group only, which will be detailed in the following subsections. The group relative way that GRPO leverages to calculate the advantages, aligns well with the comparative nature of rewards models, as reward models are typically trained on datasets of comparisons between outputs on the same question. Also note that, instead of adding KL penalty in the reward, GRPO regularizes by directly adding the KL divergence between the trained policy and the reference policy to the loss, avoiding complicating the calculation of 
A
^
i
,
t
. And different from the KL penalty term used in (2), we estimate the KL divergence with the following unbiased estimator (Schulman, 2020):

𝔻
K
​
L
[
π
θ
|
|
π
r
​
e
​
f
]
=
π
r
​
e
​
f
​
(
o
i
,
t
|
q
,
o
i
,
<
t
)
π
θ
​
(
o
i
,
t
|
q
,
o
i
,
<
t
)
−
log
π
r
​
e
​
f
​
(
o
i
,
t
|
q
,
o
i
,
<
t
)
π
θ
​
(
o
i
,
t
|
q
,
o
i
,
<
t
)
−
1
,
(4)
which is guaranteed to be positive.

Algorithm 1 Iterative Group Relative Policy Optimization
Input initial policy model 
π
θ
init
; reward models 
r
φ
; task prompts 
𝒟
; hyperparameters 
ε
, 
β
, 
μ

1:policy model 
π
θ
←
π
θ
init
2:for iteration = 1, …, I do
3:  reference model 
π
r
​
e
​
f
←
π
θ
4:  for step = 1, …, M do
5:    Sample a batch 
𝒟
b
 from 
𝒟
6:    Update the old policy model 
π
θ
o
​
l
​
d
←
π
θ
7:    Sample 
G
 outputs 
{
o
i
}
i
=
1
G
∼
π
θ
o
​
l
​
d
(
⋅
∣
q
)
 for each question 
q
∈
𝒟
b
8:    Compute rewards 
{
r
i
}
i
=
1
G
 for each sampled output 
o
i
 by running 
r
φ
9:    Compute 
A
^
i
,
t
 for the 
t
-th token of 
o
i
 through group relative advantage estimation.
10:    for GRPO iteration = 1, …, 
μ
 do
11:     Update the policy model 
π
θ
 by maximizing the GRPO objective (Equation 21)       
12:  Update 
r
φ
 through continuous training using a replay mechanism.
Output 
π
θ

4.1.2Outcome Supervision RL with GRPO
Formally, for each question 
q
, a group of outputs 
{
o
1
,
o
2
,
⋯
,
o
G
}
 are sampled from the old policy model 
π
θ
o
​
l
​
d
. A reward model is then used to score the outputs, yielding 
G
 rewards 
𝐫
=
{
r
1
,
r
2
,
⋯
,
r
G
}
 correspondingly. Subsequently, these rewards are normalized by subtracting the group average and dividing by the group standard deviation. Outcome supervision provides the normalized reward at the end of each output 
o
i
 and sets the advantages 
A
^
i
,
t
 of all tokens in the output as the normalized reward, i.e., 
A
^
i
,
t
=
r
~
i
=
r
i
−
mean
​
(
𝐫
)
std
​
(
𝐫
)
, and then optimizes the policy by maximizing the objective defined in equation (3).

4.1.3Process Supervision RL with GRPO
Outcome supervision only provides a reward at the end of each output, which may not be sufficient and efficient to supervise the policy in complex mathematical tasks. Following Wang et al. (2023b), we also explore process supervision, which provides a reward at the end of each reasoning step. Formally, given the question 
q
 and 
G
 sampled outputs 
{
o
1
,
o
2
,
⋯
,
o
G
}
, a process reward model is used to score each step of the outputs, yielding corresponding rewards: 
𝐑
=
{
{
r
1
i
​
n
​
d
​
e
​
x
​
(
1
)
,
⋯
,
r
1
i
​
n
​
d
​
e
​
x
​
(
K
1
)
}
,
⋯
,
{
r
G
i
​
n
​
d
​
e
​
x
​
(
1
)
,
⋯
,
r
G
i
​
n
​
d
​
e
​
x
​
(
K
G
)
}
}
, where 
i
​
n
​
d
​
e
​
x
​
(
j
)
 is the end token index of the 
j
-th step, and 
K
i
 is the total number of steps in the 
i
-th output. We also normalize these rewards with the average and the standard deviation, i.e., 
r
~
i
i
​
n
​
d
​
e
​
x
​
(
j
)
=
r
i
i
​
n
​
d
​
e
​
x
​
(
j
)
−
mean
​
(
𝐑
)
std
​
(
𝐑
)
. Subsequently, the process supervision calculates the advantage of each token as the sum of the normalized rewards from the following steps, i.e., 
A
^
i
,
t
=
∑
i
​
n
​
d
​
e
​
x
​
(
j
)
≥
t
r
~
i
i
​
n
​
d
​
e
​
x
​
(
j
)
, and then optimizes the policy by maximizing the objective defined in equation (3).

4.1.4Iterative RL with GRPO
As the reinforcement learning training process progresses, the old reward model may not be sufficient to supervise the current policy model. Therefore, we also explore the iterative RL with GRPO. As shown in Algorithm 1, in iterative GRPO, we generate new training sets for the reward model based on the sampling results from the policy model and continually train the old reward model using a replay mechanism that incorporates 10% of historical data. Then, we set the reference model as the policy model, and continually train the policy model with the new reward model.

4.2Training and Evaluating DeepSeekMath-RL
We conduct RL based on DeepSeekMath-Instruct 7B. The training data of RL are chain-of-thought-format questions related to GSM8K and MATH from the SFT data, which consists of around 144K questions. We exclude other SFT questions to investigate the impact of RL on benchmarks that lack data throughout the RL phase. We construct the training set of reward models following (Wang et al., 2023b). We train our initial reward model based on the DeepSeekMath-Base 7B with a learning rate of 2e-5. For GRPO, we set the learning rate of the policy model as 1e-6. The KL coefficient is 0.04. For each question, we sample 
64
 outputs. The max length is set to 1024, and the training batch size is 1024. The policy model only has a single update following each exploration stage. We evaluate DeepSeekMath-RL 7B on benchmarks following DeepSeekMath-Instruct 7B. For DeepSeekMath-RL 7B, GSM8K and MATH with chain-of-thought reasoning can be regarded as in-domain tasks and all the other benchmarks can be regarded as out-of-domain tasks.

Table 5 demonstrates the performance of open- and closed-source models with both chain-of-thought and tool-integrated reasoning on English and Chinese benchmarks. We find that: 1) DeepSeekMath-RL 7B attains accuracies of 88.2% and 51.7% on GSM8K and MATH, respectively, utilizing chain-of-thought reasoning. This performance surpasses that of all open-source models in the 7B to 70B range, as well as the majority of closed-source models. 2) Crucially, DeepSeekMath-RL 7B is only trained on chain-of-thought-format instruction tuning data of GSM8K and MATH, starting from DeepSeekMath-Instruct 7B. Despite the constrained scope of its training data, it outperforms DeepSeekMath-Instruct 7B across all evaluation metrics, showcasing the effectiveness of reinforcement learning.

5Discussion
In this section, we will share our findings in pre-training and RL experiments.

5.1Lessons Learnt in Pre-Training
We first share our experience in pre-training. Unless otherwise specified, we will adhere to the training settings outlined in Section 2.2.1. It is worth noting that, when referring to the DeepSeekMath Corpus in this section, we use an 89B-token dataset from the second iteration of the data collection process.

5.1.1Code Training Benefits Mathematical Reasoning
A popular yet unverified hypothesis suggests that code training improves reasoning. We attempt to offer a partial response to this, particularly within the mathematical domain: code training improves models’ ability to do mathematical reasoning both with and without tool use.

To study how code training affects mathematical reasoning, we experimented with the following two-stage training and one-stage training settings:

Two-Stage Training

• Code Training for 400B Tokens 
→
 Math Training for 150B Tokens: We train DeepSeek-LLM 1.3B for 400B code tokens followed by 150B math tokens;
• General Training for 400B Tokens 
→
 Math Training for 150B Tokens: As a control experiment, we also experiment with general tokens (sampled from a large-scale general corpus created by DeepSeek-AI) instead of code tokens in the first stage of training, in an attempt to investigate the advantages of code tokens over general tokens in improving mathematical reasoning.
One-Stage Training

• Math Training for 150B Tokens: We train DeepSeek-LLM 1.3B for 150B math tokens;
• Training on a mixture of 400B Code Tokens and 150B Math Tokens: Math training following code training degrades coding performance. We investigate whether code tokens, when mixed with math tokens for one-stage training, would still improve mathematical reasoning and also alleviate the problem of catastrophic forgetting.
Training Setting	Training Tokens	w/o Tool Use	w/ Tool Use
General	Code	Math	GSM8K	MATH	CMATH	GSM8K+Python	MATH+Python
No Continual Training	–	–	–	2.9%	3.0%	12.3%	2.7%	2.3%
Two-Stage Training
Stage 1: General Training	400B	–	–	2.9%	3.2%	14.8%	3.3%	2.3%
Stage 2: Math Training	–	–	150B	19.1%	14.4%	37.2%	14.3%	6.7%
Stage 1: Code Training	–	400B	–	5.9%	3.6%	19.9%	12.4%	10.0%
Stage 2: Math Training	–	–	150B	21.9%	15.3%	39.7%	17.4%	9.4%
One-Stage Training
Math Training	–	–	150B	20.5%	13.1%	37.6%	11.4%	6.5%
Code & Math Mixed Training	–	400B	150B	17.6%	12.1%	36.3%	19.7%	13.5%
 

Table 6: Investigation of how code affects mathematical reasoning under different training settings. We experiment with DeepSeek-LLM 1.3B, and evaluate its mathematical reasoning performance without and with tool use via few-shot chain-of-thought prompting and few-shot program-of-thought prompting, respectively.
Training Setting	Training Tokens	MMLU	BBH	HumanEval (Pass@1)	MBPP (Pass@1)
General	Code	Math
No Continual Training	–	–	–	24.5%	28.1%	12.2%	13.0%
Two-Stage Training
Stage 1: General Training	400B	–	–	25.9%	27.7%	15.2%	13.6%
Stage 2: Math Training	–	–	150B	33.1%	32.7%	12.8%	13.2%
Stage 1: Code Training	–	400B	–	25.0%	31.5%	25.0%	40.0%
Stage 2: Math Training	–	–	150B	36.2%	35.3%	12.2%	17.0%
One-Stage Training
Math Training	–	–	150B	32.3%	32.5%	11.6%	13.2%
Code & Math Mixed Training	–	400B	150B	33.5%	35.6%	29.3%	39.4%
 

Table 7: Investigation of how different settings of code and math training affect model performance of language understanding, reasoning, and coding. We experiment with DeepSeek-LLM 1.3B. We evaluate the models on MMLU and BBH using few-shot chain-of-thought prompting. On HumanEval and MBPP, we conduct zero-shot and few-shot evaluations, respectively.
Results
Table 6 and Table 7 demonstrate the downstream performance under different training settings.

Code training benefits program-aided mathematical reasoning, both under the two-stage training and one-stage training settings. As shown in Table 6, under the two-stage training setting, code training alone already significantly enhances the ability to solve GSM8K and MATH problems using Python. Math training in the second stage yields further improvements. Interestingly, under the one-stage training setting, mixing code tokens and math tokens effectively mitigates the issue of catastrophic forgetting that arises from two-stage training, and also synergizes coding (Table 7) and program-aided mathematical reasoning (Table 6).

Code training also improves mathematical reasoning without tool use. Under the two-stage training setting, the initial stage of code training already results in moderate enhancements. It also boosts the efficiency of the subsequent math training, eventually leading to the best performance. However, combining code tokens and math tokens for one-stage training compromises mathematical reasoning without tool use. One conjecture is that DeepSeek-LLM 1.3B, due to its limited scale, lacks the capacity to fully assimilate both code and mathematical data simultaneously.

Model	Size	ArXiv Corpus	English Benchmarks	Chinese Benchmarks
GSM8K	MATH	OCW	SAT	 
MMLU
STEM
 	CMATH	 
Gaokao
MathCloze
 	 
Gaokao
MathQA
 
DeepSeek-LLM	1.3B	No Math Training	2.9%	3.0%	2.9%	15.6%	19.5%	12.3%	0.8%	17.9%
MathPile	2.7%	3.3%	2.2%	12.5%	15.7%	1.2%	0.0%	2.8%
ArXiv-RedPajama	3.3%	3.4%	4.0%	9.4%	9.0%	7.4%	0.8%	2.3%
DeepSeek-Coder-Base-v1.5	7B	No Math Training	29.0%	12.5%	6.6%	40.6%	38.1%	45.9%	5.9%	21.1%
MathPile	23.6%	11.5%	7.0%	46.9%	35.8%	37.9%	4.2%	25.6%
ArXiv-RedPajama	28.1%	11.1%	7.7%	50.0%	35.2%	42.6%	7.6%	24.8%
 

Table 8: Effect of math training on different arXiv datasets. Model performance is evaluated with few-shot chain-of-thought prompting.
ArXiv Corpus	miniF2F-valid	miniF2F-test
No Math Training	20.1%	21.7%
MathPile	16.8%	16.4%
ArXiv-RedPajama	14.8%	11.9%
Table 9: Effect of math training on different arXiv corpora, the base model being DeepSeek-Coder-Base-v1.5 7B. We evaluate informal-to-formal proving in Isabelle.
5.1.2ArXiv Papers Seem Ineffective in Improving Mathematical Reasoning
ArXiv papers are commonly included as a component of math pre-training data (Lewkowycz et al., 2022a; Polu and Sutskever, 2020; Azerbayev et al., 2023; Wang et al., 2023c). However, detailed analysis regarding their impact on mathematical reasoning has not been extensively conducted. Perhaps counter-intuitively, according to our experiments, arXiv papers seem ineffective in improving mathematical reasoning. We experiment with models of different sizes, including DeepSeek-LLM 1.3B and DeepSeek-Coder-Base-v1.5 7B (Guo et al., 2024), using arXiv corpora that underwent varied processing pipelines:

• MathPile (Wang et al., 2023c): an 8.9B-token corpus developed with cleaning and filtering heuristic rules, over 85% of which are scientific arXiv papers;
• ArXiv-RedPajama (Computer, 2023): the entirety of arXiv LaTeX files with preambles, comments, macros, and bibliographies removed, totaling 28.0B tokens.
In our experiments, we separately train DeepSeek-LLM 1.3B for 150B tokens and DeepSeek-Coder-Base-v1.5 7B for 40B tokens on each arXiv corpus. It seems that arXiv papers are ineffective in improving mathematical reasoning. When trained on a arXiv-only corpus, both models display no notable improvements or even deterioration across various mathematical benchmarks of different complexities employed in this study. These benchmarks include quantitative reasoning datasets like GSM8K and MATH (Table 8), multiple-choice challenges like MMLU-STEM (Table 8), and formal mathematics like miniF2F (Table 9).

However, this conclusion has its limitations and should be taken with a grain of salt. We have not yet studied:

• The impact of arXiv tokens on specific math-related tasks not included in this research, such as informalization of theorems which is to convert formal statements or proofs to their informal versions;
• The effect of arXiv tokens when combined with other types of data;
• Whether the benefits of arXiv papers would manifest themselves at a larger model scale.
Thus, further exploration is required, which we leave for future studies.

5.2Insights of Reinforcement Learning
5.2.1Towards to a Unified Paradigm
In this section, we provide a unified paradigm to analyze different training methods, such as SFT, RFT, DPO, PPO, GRPO, and further conduct experiments to explore the factors of the unified paradigm. Generally, the gradient with respect to the parameter 
θ
 of a training method can be written as:

∇
θ
𝒥
𝒜
​
(
θ
)
=
𝔼
​
[
(
q
,
o
)
∼
𝒟
⏟
D
​
a
​
t
​
a
​
S
​
o
​
u
​
r
​
c
​
e
]
​
(
1
|
o
|
​
∑
t
=
1
|
o
|
G
​
C
𝒜
​
(
q
,
o
,
t
,
π
r
​
f
)
⏟
G
​
r
​
a
​
d
​
i
​
e
​
n
​
t
​
C
​
o
​
e
​
f
​
f
​
i
​
c
​
i
​
e
​
n
​
t
​
∇
θ
log
⁡
π
θ
​
(
o
t
|
q
,
o
<
t
)
)
.
(5)
There exist three key components: 1) Data Source 
𝒟
, which determines the training data; 2) Reward Function 
π
r
​
f
, which is the source of the training reward signal; 3) Algorithm 
𝒜
: which processes the training data and the reward signal to the gradient coefficient 
G
​
C
 that determines the magnitude of the penalty or reinforcement for the data. We analyze several representative methods based on such a unified paradigm:

Methods	Data Source	Reward Function	Gradient Coefficient
SFT	
q
,
o
∼
P
s
​
f
​
t
​
(
Q
,
O
)
-	1
RFT	
q
∼
P
s
​
f
​
t
​
(
Q
)
, 
o
∼
π
s
​
f
​
t
​
(
O
|
q
)
Rule	Equation 10
DPO	
q
∼
P
s
​
f
​
t
​
(
Q
)
, 
o
+
,
o
−
∼
π
s
​
f
​
t
​
(
O
|
q
)
Rule	Equation 14
Online RFT	
q
∼
P
s
​
f
​
t
​
(
Q
)
, 
o
∼
π
θ
​
(
O
|
q
)
Rule	Equation 10
PPO	
q
∼
P
s
​
f
​
t
​
(
Q
)
, 
o
∼
π
θ
​
(
O
|
q
)
Model	Equation 18
GRPO	
q
∼
P
s
​
f
​
t
​
(
Q
)
, 
{
o
i
}
i
=
1
G
∼
π
θ
​
(
O
|
q
)
Model	Equation 21
Table 10:The data source and gradient coefficient of different methods. 
P
s
​
f
​
t
 denotes the data distribution of supervised fine-tuning datasets. 
π
θ
s
​
f
​
t
 and 
π
θ
 denote the supervised fine-tuned model and the real-time policy model during the online training process, respectively.
• Supervised Fine-tuning (SFT): SFT fine-tunes pretrained model on human selected SFT data.
• Rejection Sampling Fine-tuning (RFT): RFT further fine-tunes the SFT model on the filtered outputs sampled from the SFT model based on SFT questions. RFT filters the outputs based on the correctness of their answers.
• Direct Preference Optimization (DPO): DPO further refines the SFT model by fine-tuning it on augmented outputs sampled from the SFT model, using pair-wise DPO loss.
• Online Rejection Sampling Fine-tuning (Online RFT): Different from RFT, Online RFT initiates the policy model using the SFT model and refines it by fine-tuning with the augmented outputs sampled from the real-time policy model.
• PPO/GRPO: PPO/GRPO initializes the policy model using the SFT model and reinforces it with the outputs sampled from the real-time policy model.
We summarize the components of these methods in Table 10. Please refer to Appendix A.1 for a more detailed derivation process.

Refer to caption
Figure 5:Performance of the DeepSeekMath-Instruct 1.3B model, which was further trained using various methods, on two benchmarks.
Refer to caption
Figure 6:Performance of iterative reinforcement learning with DeepSeekMath-Instruct 7B on two benchmarks.
Observation about Data Source
We divide the data source into two categories, online sampling, and offline sampling. Online sampling denotes that the training data is from the exploration results of the real-time training policy model, while offline sampling denotes that the training data is from the sampling results of the initial SFT model. RFT and DPO follow the offline style, while Online RFT and GRPO follow the online style.

As shown in Figure 5, we find that the Online RFT significantly outperforms RFT on two benchmarks. Specifically, Online RFT is comparable to RFT in the early stage of training but gains an absolute advantage in the later stage, demonstrating the superiority of online training. This is intuitive, as in the initial stage, the actor and the SFT model exhibit close resemblance, with the sampled data revealing only minor differences. In the later stage, however, the data sampled from the actor will exhibit more significant differences, and real-time data sampling will offer greater advantages.

Observation about Gradient Coefficient
The algorithm processes the reward signal to the gradient coefficient to update the model parameter. We divide the reward function as ‘Rule’ and ‘Model’ in our experiments. Rule refers to judging the quality of a response based on the correctness of the answer, and Model denotes that we train a reward model to score each response. The training data of the reward model is based on the rule judgment. Equations 10 and 21 highlight a key difference between GRPO and Online RFT: GRPO uniquely adjusts its gradient coefficient based on the reward value provided by the reward model. This allows for differential reinforcement and penalization of responses according to their varying magnitudes. In contrast, Online RFT lacks this feature; it does not penalize incorrect responses and uniformly reinforces all responses with correct answers at the same level of intensity.

As demonstrated in Figure 5, GRPO surpasses online RFT, thereby highlighting the efficiency of altering positive and negative gradient coefficients. In addition, GRPO+PS shows superior performance compared to GRPO+OS, indicating the benefits of using fine-grained, step-aware gradient coefficients. Furthermore, we explore the iterative RL, in our experiments, we conduct two rounds of iteration. As shown in Figure 6, we notice that the iterative RL significantly improves the performance, especially at the first iteration.

Refer to caption
Figure 7:The Maj@K and Pass@K of SFT and RL DeepSeekMath 7B on GSM8K and MATH (temperature 
0.7
). It was noted that RL enhances Maj@K but not Pass@K.
5.2.2Why RL Works?
In this paper, we conduct reinforcement learning based on a subset of instruction tuning data, and it achieves significant performance enhancement upon the instruction tuning model. To further explain why reinforcement learning works. We evaluate the Pass@K and Maj@K accuracy of the Instruct and RL models on two benchmarks. As shown in Figure 7, RL enhances Maj@K’s performance but not Pass@K. These findings indicate that RL enhances the model’s overall performance by rendering the output distribution more robust, in other words, it seems that the improvement is attributed to boosting the correct response from TopK rather than the enhancement of fundamental capabilities. Similarly, (Wang et al., 2023a) identified a misalignment problem in reasoning tasks within the SFT model, showing that the reasoning performance of SFT models can be improved through a series of preference alignment strategies (Yuan et al., 2023b; Song et al., 2023; Wang et al., 2023a).

5.2.3How to Achieve More Effective RL?
We demonstrate RL works pretty well in mathematical reasoning tasks. We also provide a unified paradigm to understand different representative training methods. Within this paradigm, all methods are conceptualized as either direct or simplified RL techniques. As summarized in Equation 5, there exist three key components: Data Source, Algorithm, and Reward Function. We provide some potential future directions about the three components.

Data Source
Data source is the raw material of all training methods. In the context of RL, we specifically refer to the data source as the unlabeled questions with the outputs sampled from the policy model. In this paper, we only use the questions from the instruction tuning stage and a naive nucleus sampling to sample outputs. We think this is a potential reason that our RL pipeline only improves the Maj@K performance. In the future, we will explore our RL pipeline on out-of-distribution question prompts, in conjunction with advanced sampling (decoding) strategies, like those based on tree-search methods (Yao et al., 2023). Also, the efficient inference techniques (Xia et al., 2023; Leviathan et al., 2023; Kwon et al., 2023; Xia et al., 2024), which determines the exploration efficiency of policy models, also play an exceedingly important role.

Algorithms
Algorithms process the data and reward signal to the gradient coefficient to update the model parameter. Based on Equation 5, to some extent, all methods now fully TRUST the signal of the reward function to increase or decrease the conditional probability of a certain token. However, it is impossible to ensure the reward signal is always reliable, especially in extremely complex tasks. For example, even the PRM800K datasets (Lightman et al., 2023), which have been carefully annotated by well-trained annotators, still contain approximately 20% of incorrectly annotations7
7https://github.com/openai/prm800k/issues/12#issuecomment-1728491852
. To this end, we will explore the reinforcement learning algorithm that is robust against noisy reward signals. We believe such WEAK-TO-STRONG (Burns et al., 2023) alignment methods will bring a fundamental change to the learning algorithms.

Reward Function
Reward function is the source of the training signal. In RL, the reward function is usually the neural reward model. We think there exist three important directions for reward models: 1) How to enhance the generalization ability of the reward model. The reward model must be effectively generalized to handle out-of-distribution questions and advanced decoding outputs; otherwise, reinforcement learning may merely stabilize the distribution of LLMs rather than improve their fundamental capabilities; 2) How to reflect the uncertainty of reward model. The uncertainty could potentially act as a linking bridge between the weak reward model and the weak-to-strong learning algorithms; 3) How to efficiently build high-quality process reward models that can provide fine-grained training signals for the reasoning process (Lightman et al., 2023; Wang et al., 2023b).

6Conclusion, Limitation, and Future Work
We present DeepSeekMath, which outperforms all open-source models on the competition-level MATH benchmark and approaches the performance of closed models. DeepSeekMath is initialized with DeepSeek-Coder-v1.5 7B and undergoes continual training for 500B tokens, with a significant component of the training data being 120B math tokens sourced from Common Crawl. Our extensive ablation study shows web pages offer significant potential for high-quality mathematical data, while arXiv may not as beneficial as we expected. We introduce Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), which can notably improve mathematical reasoning capabilities with less memory consumption. The experiment results show that GRPO is effective even if DeepSeekMath-Instruct 7B has reached a high score on benchmarks. We also provide a unified paradigm to understand a series of methods and summarize several potential directions for more effective reinforcement learning.

Although DeepSeekMath achieves impressive scores on quantitative reasoning benchmarks, its capability on geometry and theorem-proof are relatively weaker than closed models. For instance, in our dry run, the model cannot handle problems related to triangles and ellipses, which may indicate data selection bias in pre-training and fine-tuning. In addition, restricted by the model scale, DeepSeekMath is worse than GPT-4 on few-shot capability. GPT-4 could improve its performance with few-shot inputs, while DeepSeekMath shows similar performance in zero-shot and few-shot evaluation. In the future, we will further improve our engineered data selection pipeline to construct more high-quality pre-trained corpus. In addition, we will explore the potential directions (Section 5.2.3) for more effective reinforcement learning of LLMs.

References
Anil et al. (2023)
R. Anil, S. Borgeaud, Y. Wu, J. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, D. Silver, S. Petrov, M. Johnson, I. Antonoglou, J. Schrittwieser, A. Glaese, J. Chen, E. Pitler, T. P. Lillicrap, A. Lazaridou, O. Firat, J. Molloy, M. Isard, P. R. Barham, T. Hennigan, B. Lee, F. Viola, M. Reynolds, Y. Xu, R. Doherty, E. Collins, C. Meyer, E. Rutherford, E. Moreira, K. Ayoub, M. Goel, G. Tucker, E. Piqueras, M. Krikun, I. Barr, N. Savinov, I. Danihelka, B. Roelofs, A. White, A. Andreassen, T. von Glehn, L. Yagati, M. Kazemi, L. Gonzalez, M. Khalman, J. Sygnowski, and et al.Gemini: A family of highly capable multimodal models.CoRR, abs/2312.11805, 2023.10.48550/ARXIV.2312.11805.URL https://doi.org/10.48550/arXiv.2312.11805.
Austin et al. (2021)
J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry, Q. Le, et al.Program synthesis with large language models.arXiv preprint arXiv:2108.07732, 2021.
Azerbayev et al. (2023)
Z. Azerbayev, H. Schoelkopf, K. Paster, M. D. Santos, S. McAleer, A. Q. Jiang, J. Deng, S. Biderman, and S. Welleck.Llemma: An open language model for mathematics.arXiv preprint arXiv:2310.10631, 2023.
Bai et al. (2023)
J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, et al.Qwen technical report.arXiv preprint arXiv:2309.16609, 2023.
Burns et al. (2023)
C. Burns, P. Izmailov, J. H. Kirchner, B. Baker, L. Gao, L. Aschenbrenner, Y. Chen, A. Ecoffet, M. Joglekar, J. Leike, et al.Weak-to-strong generalization: Eliciting strong capabilities with weak supervision.arXiv preprint arXiv:2312.09390, 2023.
ChatGLM3 Team (2023)
ChatGLM3 Team.Chatglm3 series: Open bilingual chat llms, 2023.URL https://github.com/THUDM/ChatGLM3.
Chen et al. (2021)
M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba.Evaluating large language models trained on code.CoRR, abs/2107.03374, 2021.URL https://arxiv.org/abs/2107.03374.
Chen et al. (2022)
W. Chen, X. Ma, X. Wang, and W. W. Cohen.Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks.CoRR, abs/2211.12588, 2022.10.48550/ARXIV.2211.12588.URL https://doi.org/10.48550/arXiv.2211.12588.
Cobbe et al. (2021)
K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al.Training verifiers to solve math word problems.arXiv preprint arXiv:2110.14168, 2021.
Computer (2023)
T. Computer.Redpajama: an open dataset for training large language models, Oct. 2023.URL https://github.com/togethercomputer/RedPajama-Data.
DeepSeek-AI (2024)
DeepSeek-AI.Deepseek LLM: scaling open-source language models with longtermism.CoRR, abs/2401.02954, 2024.10.48550/ARXIV.2401.02954.URL https://doi.org/10.48550/arXiv.2401.02954.
Du et al. (2022)
Z. Du, Y. Qian, X. Liu, M. Ding, J. Qiu, Z. Yang, and J. Tang.Glm: General language model pretraining with autoregressive blank infilling.In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 320–335, 2022.
Gao et al. (2023)
L. Gao, A. Madaan, S. Zhou, U. Alon, P. Liu, Y. Yang, J. Callan, and G. Neubig.PAL: program-aided language models.In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 10764–10799. PMLR, 2023.URL https://proceedings.mlr.press/v202/gao23f.html.
Gou et al. (2023)
Z. Gou, Z. Shao, Y. Gong, Y. Shen, Y. Yang, M. Huang, N. Duan, and W. Chen.Tora: A tool-integrated reasoning agent for mathematical problem solving.CoRR, abs/2309.17452, 2023.10.48550/ARXIV.2309.17452.URL https://doi.org/10.48550/arXiv.2309.17452.
Guo et al. (2024)
D. Guo, Q. Zhu, D. Yang, Z. Xie, K. Dong, W. Zhang, G. Chen, X. Bi, Y. Wu, Y. K. Li, F. Luo, Y. Xiong, and W. Liang.Deepseek-coder: When the large language model meets programming – the rise of code intelligence, 2024.
Hendrycks et al. (2020)
D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt.Measuring massive multitask language understanding.arXiv preprint arXiv:2009.03300, 2020.
Hendrycks et al. (2021)
D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt.Measuring mathematical problem solving with the math dataset.arXiv preprint arXiv:2103.03874, 2021.
High-flyer (2023)
High-flyer.Hai-llm: 高效且轻量的大模型训练工具, 2023.URL https://www.high-flyer.cn/en/blog/hai-llm.
Inflection AI (2023)
Inflection AI.Inflection-2, 2023.URL https://inflection.ai/inflection-2.
Jiang et al. (2022)
A. Q. Jiang, S. Welleck, J. P. Zhou, W. Li, J. Liu, M. Jamnik, T. Lacroix, Y. Wu, and G. Lample.Draft, sketch, and prove: Guiding formal theorem provers with informal proofs.arXiv preprint arXiv:2210.12283, 2022.
Jiang et al. (2023)
A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. l. Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, et al.Mistral 7b.arXiv preprint arXiv:2310.06825, 2023.
Joulin et al. (2016)
A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, and T. Mikolov.Fasttext. zip: Compressing text classification models.arXiv preprint arXiv:1612.03651, 2016.
Kwon et al. (2023)
W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica.Efficient memory management for large language model serving with pagedattention.In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.
Leviathan et al. (2023)
Y. Leviathan, M. Kalman, and Y. Matias.Fast inference from transformers via speculative decoding.In International Conference on Machine Learning, pages 19274–19286. PMLR, 2023.
Lewkowycz et al. (2022a)
A. Lewkowycz, A. Andreassen, D. Dohan, E. Dyer, H. Michalewski, V. Ramasesh, A. Slone, C. Anil, I. Schlag, T. Gutman-Solo, et al.Solving quantitative reasoning problems with language models.Advances in Neural Information Processing Systems, 35:3843–3857, 2022a.
Lewkowycz et al. (2022b)
A. Lewkowycz, A. Andreassen, D. Dohan, E. Dyer, H. Michalewski, V. V. Ramasesh, A. Slone, C. Anil, I. Schlag, T. Gutman-Solo, Y. Wu, B. Neyshabur, G. Gur-Ari, and V. Misra.Solving quantitative reasoning problems with language models.In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022b.URL http://papers.nips.cc/paper_files/paper/2022/hash/18abbeef8cfe9203fdf9053c9c4fe191-Abstract-Conference.html.
Lightman et al. (2023)
H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe.Let’s verify step by step.arXiv preprint arXiv:2305.20050, 2023.
Loshchilov and Hutter (2017)
I. Loshchilov and F. Hutter.Decoupled weight decay regularization.arXiv preprint arXiv:1711.05101, 2017.
Luo et al. (2023)
H. Luo, Q. Sun, C. Xu, P. Zhao, J. Lou, C. Tao, X. Geng, Q. Lin, S. Chen, and D. Zhang.Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct.arXiv preprint arXiv:2308.09583, 2023.
Mishra et al. (2022)
S. Mishra, M. Finlayson, P. Lu, L. Tang, S. Welleck, C. Baral, T. Rajpurohit, O. Tafjord, A. Sabharwal, P. Clark, and A. Kalyan.LILA: A unified benchmark for mathematical reasoning.In Y. Goldberg, Z. Kozareva, and Y. Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 5807–5832. Association for Computational Linguistics, 2022.10.18653/V1/2022.EMNLP-MAIN.392.URL https://doi.org/10.18653/v1/2022.emnlp-main.392.
Nguyen et al. (2023)
X. Nguyen, W. Zhang, X. Li, M. M. Aljunied, Q. Tan, L. Cheng, G. Chen, Y. Deng, S. Yang, C. Liu, H. Zhang, and L. Bing.Seallms - large language models for southeast asia.CoRR, abs/2312.00738, 2023.10.48550/ARXIV.2312.00738.URL https://doi.org/10.48550/arXiv.2312.00738.
OpenAI (2023)
OpenAI.GPT4 technical report.arXiv preprint arXiv:2303.08774, 2023.
Ouyang et al. (2022)
L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al.Training language models to follow instructions with human feedback.Advances in Neural Information Processing Systems, 35:27730–27744, 2022.
Paster et al. (2023)
K. Paster, M. D. Santos, Z. Azerbayev, and J. Ba.Openwebmath: An open dataset of high-quality mathematical web text.CoRR, abs/2310.06786, 2023.10.48550/ARXIV.2310.06786.URL https://doi.org/10.48550/arXiv.2310.06786.
Paulson (2010)
L. C. Paulson.Three years of experience with sledgehammer, a practical link between automatic and interactive theorem provers.In R. A. Schmidt, S. Schulz, and B. Konev, editors, Proceedings of the 2nd Workshop on Practical Aspects of Automated Reasoning, PAAR-2010, Edinburgh, Scotland, UK, July 14, 2010, volume 9 of EPiC Series in Computing, pages 1–10. EasyChair, 2010.10.29007/TNFD.URL https://doi.org/10.29007/tnfd.
Polu and Sutskever (2020)
S. Polu and I. Sutskever.Generative language modeling for automated theorem proving.CoRR, abs/2009.03393, 2020.URL https://arxiv.org/abs/2009.03393.
Rafailov et al. (2023)
R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn.Direct preference optimization: Your language model is secretly a reward model.2023.
Schulman (2020)
J. Schulman.Approximating kl divergence, 2020.URL http://joschu.net/blog/kl-approx.html.
Schulman et al. (2015)
J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel.High-dimensional continuous control using generalized advantage estimation.arXiv preprint arXiv:1506.02438, 2015.
Schulman et al. (2017)
J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov.Proximal policy optimization algorithms.arXiv preprint arXiv:1707.06347, 2017.
Shi et al. (2023)
F. Shi, M. Suzgun, M. Freitag, X. Wang, S. Srivats, S. Vosoughi, H. W. Chung, Y. Tay, S. Ruder, D. Zhou, D. Das, and J. Wei.Language models are multilingual chain-of-thought reasoners.In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.URL https://openreview.net/pdf?id=fR3wGCk-IXp.
Song et al. (2023)
F. Song, B. Yu, M. Li, H. Yu, F. Huang, Y. Li, and H. Wang.Preference ranking optimization for human alignment.arXiv preprint arXiv:2306.17492, 2023.
Suzgun et al. (2022)
M. Suzgun, N. Scales, N. Schärli, S. Gehrmann, Y. Tay, H. W. Chung, A. Chowdhery, Q. V. Le, E. H. Chi, D. Zhou, et al.Challenging big-bench tasks and whether chain-of-thought can solve them.arXiv preprint arXiv:2210.09261, 2022.
Tao (2023)
T. Tao.Embracing change and resetting expectations, 2023.URL https://unlocked.microsoft.com/ai-anthology/terence-tao/.
Touvron et al. (2023)
H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. Canton-Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom.Llama 2: Open foundation and fine-tuned chat models.CoRR, abs/2307.09288, 2023.10.48550/arXiv.2307.09288.URL https://doi.org/10.48550/arXiv.2307.09288.
Trinh et al. (2024)
T. H. Trinh, Y. Wu, Q. V. Le, H. He, and T. Luong.Solving olympiad geometry without human demonstrations.Nature, 625(7995):476–482, 2024.
Wang et al. (2023a)
P. Wang, L. Li, L. Chen, F. Song, B. Lin, Y. Cao, T. Liu, and Z. Sui.Making large language models better reasoners with alignment.arXiv preprint arXiv:2309.02144, 2023a.
Wang et al. (2023b)
P. Wang, L. Li, Z. Shao, R. Xu, D. Dai, Y. Li, D. Chen, Y. Wu, and Z. Sui.Math-shepherd: Verify and reinforce llms step-by-step without human annotations.CoRR, abs/2312.08935, 2023b.
Wang et al. (2023c)
Z. Wang, R. Xia, and P. Liu.Generative AI for math: Part I - mathpile: A billion-token-scale pretraining corpus for math.CoRR, abs/2312.17120, 2023c.10.48550/ARXIV.2312.17120.URL https://doi.org/10.48550/arXiv.2312.17120.
Wei et al. (2022)
J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. H. Chi, Q. V. Le, and D. Zhou.Chain-of-thought prompting elicits reasoning in large language models.In NeurIPS, 2022.URL http://papers.nips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html.
Wei et al. (2023)
T. Wei, J. Luan, W. Liu, S. Dong, and B. Wang.Cmath: Can your language model pass chinese elementary school math test?, 2023.
Wenzel et al. (2008)
M. Wenzel, L. C. Paulson, and T. Nipkow.The isabelle framework.In O. A. Mohamed, C. A. Muñoz, and S. Tahar, editors, Theorem Proving in Higher Order Logics, 21st International Conference, TPHOLs 2008, Montreal, Canada, August 18-21, 2008. Proceedings, volume 5170 of Lecture Notes in Computer Science, pages 33–38. Springer, 2008.10.1007/978-3-540-71067-7_7.URL https://doi.org/10.1007/978-3-540-71067-7_7.
Xia et al. (2023)
H. Xia, T. Ge, P. Wang, S.-Q. Chen, F. Wei, and Z. Sui.Speculative decoding: Exploiting speculative execution for accelerating seq2seq generation.In H. Bouamor, J. Pino, and K. Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 3909–3925, Singapore, Dec. 2023. Association for Computational Linguistics.10.18653/v1/2023.findings-emnlp.257.URL https://aclanthology.org/2023.findings-emnlp.257.
Xia et al. (2024)
H. Xia, Z. Yang, Q. Dong, P. Wang, Y. Li, T. Ge, T. Liu, W. Li, and Z. Sui.Unlocking efficiency in large language model inference: A comprehensive survey of speculative decoding.arXiv preprint arXiv:2401.07851, 2024.
Yao et al. (2023)
S. Yao, D. Yu, J. Zhao, I. Shafran, T. L. Griffiths, Y. Cao, and K. Narasimhan.Tree of thoughts: Deliberate problem solving with large language models.arXiv preprint arXiv:2305.10601, 2023.
Yu et al. (2023)
L. Yu, W. Jiang, H. Shi, J. Yu, Z. Liu, Y. Zhang, J. T. Kwok, Z. Li, A. Weller, and W. Liu.Metamath: Bootstrap your own mathematical questions for large language models.CoRR, abs/2309.12284, 2023.10.48550/ARXIV.2309.12284.URL https://doi.org/10.48550/arXiv.2309.12284.
Yuan et al. (2023a)
Z. Yuan, H. Yuan, C. Li, G. Dong, C. Tan, and C. Zhou.Scaling relationship on learning mathematical reasoning with large language models.arXiv preprint arXiv:2308.01825, 2023a.
Yuan et al. (2023b)
Z. Yuan, H. Yuan, C. Tan, W. Wang, S. Huang, and F. Huang.Rrhf: Rank responses to align language models with human feedback without tears.arXiv preprint arXiv:2304.05302, 2023b.
Yue et al. (2023)
X. Yue, X. Qu, G. Zhang, Y. Fu, W. Huang, H. Sun, Y. Su, and W. Chen.Mammoth: Building math generalist models through hybrid instruction tuning.CoRR, abs/2309.05653, 2023.10.48550/ARXIV.2309.05653.URL https://doi.org/10.48550/arXiv.2309.05653.
Zheng et al. (2021)
K. Zheng, J. M. Han, and S. Polu.Minif2f: a cross-system benchmark for formal olympiad-level mathematics.arXiv preprint arXiv:2109.00110, 2021.
Zhong et al. (2023)
W. Zhong, R. Cui, Y. Guo, Y. Liang, S. Lu, Y. Wang, A. Saied, W. Chen, and N. Duan.AGIEval: A human-centric benchmark for evaluating foundation models.CoRR, abs/2304.06364, 2023.10.48550/arXiv.2304.06364.URL https://doi.org/10.48550/arXiv.2304.06364.
Appendix AAppendix
A.1Analysis of Reinforcement Learning
We provide the detailed derivation of the data source and gradient coefficient (algorithm and reward function) across various methods, including SFT, RFT, Online RFT, DPO, PPO, and GRPO.

A.1.1Supervised Fine-tuning
The objective of Supervised Fine-tuning is maximizing the following objective:

𝒥
S
​
F
​
T
​
(
θ
)
=
𝔼
​
[
q
,
o
∼
P
s
​
f
​
t
​
(
Q
,
O
)
]
​
(
1
|
o
|
​
∑
t
=
1
|
o
|
log
⁡
π
θ
​
(
o
t
|
q
,
o
<
t
)
)
.
(6)
The gradient of 
𝒥
S
​
F
​
T
​
(
θ
)
 is:

∇
θ
𝒥
S
​
F
​
T
=
𝔼
​
[
q
,
o
∼
P
s
​
f
​
t
​
(
Q
,
O
)
]
​
(
1
|
o
|
​
∑
t
=
1
|
o
|
∇
θ
log
⁡
π
θ
​
(
o
t
|
q
,
o
<
t
)
)
.
(7)
Data Source: The dataset employed for SFT. Reward Function: This can be regarded as human selection. Gradient Coefficient: always set to 1.

A.1.2Rejection Sampling Fine-tuning
Rejection Sampling Fine-tuning first samples multiple outputs from the supervised fine-tuned LLMs for each question, and then trains LLMs on the sampled outputs with the correct answer. Formally, the objective of RFT is to maximize the following objectives:

𝒥
R
​
F
​
T
​
(
θ
)
=
𝔼
​
[
q
∼
P
s
​
f
​
t
​
(
Q
)
,
o
∼
π
s
​
f
​
t
​
(
O
|
q
)
]
​
(
1
|
o
|
​
∑
t
=
1
|
o
|
𝕀
​
(
o
)
​
log
⁡
π
θ
​
(
o
t
|
q
,
o
<
t
)
)
.
(8)
The gradient of 
𝒥
R
​
F
​
T
​
(
θ
)
 is:

∇
θ
𝒥
R
​
F
​
T
​
(
θ
)
=
𝔼
​
[
q
∼
P
s
​
f
​
t
​
(
Q
)
,
o
∼
π
s
​
f
​
t
​
(
O
|
q
)
]
​
(
1
|
o
|
​
∑
t
=
1
|
o
|
𝕀
​
(
o
)
​
∇
θ
log
⁡
π
θ
​
(
o
t
|
q
,
o
<
t
)
)
.
(9)
Data Source: question in SFT dataset with outputs sampled from SFT model. Reward Function: Rule (whether the answer is correct or not). Gradient Coefficient:

G
C
R
​
F
​
T
(
q
,
o
,
t
)
=
𝕀
(
o
)
=
{
1
the
​
answer
​
of
​
o
​
is
​
correct
0
the
​
answer
​
of
​
o
​
is
​
incorrect
(10)
A.1.3Online Rejection Sampling Fine-tuning
The only difference between RFT and Online RFT is that the outputs of Online RFT are sampled from the real-time policy model 
π
θ
, rather than from the SFT model 
π
θ
s
​
f
​
t
. Therefore, the gradient of online RFT is:

∇
θ
𝒥
O
​
n
​
R
​
F
​
T
​
(
θ
)
=
𝔼
​
[
q
∼
P
s
​
f
​
t
​
(
Q
)
,
o
∼
π
θ
​
(
O
|
q
)
]
​
(
1
|
o
|
​
∑
t
=
1
|
o
|
𝕀
​
(
o
)
​
∇
θ
log
⁡
π
θ
​
(
o
t
|
q
,
o
<
t
)
)
.
(11)
A.1.4Direct Preference Optimization (DPO)
The objective of DPO is:

𝒥
D
​
P
​
O
​
(
θ
)
=
𝔼
​
[
q
∼
P
s
​
f
​
t
​
(
Q
)
,
o
+
,
o
−
∼
π
s
​
f
​
t
​
(
O
|
q
)
]
​
log
⁡
σ
​
(
β
​
1
|
o
+
|
​
∑
t
=
1
|
o
+
|
log
⁡
π
θ
​
(
o
t
+
|
q
,
o
<
t
+
)
π
ref
​
(
o
t
+
|
q
,
o
<
t
+
)
−
β
​
1
|
o
−
|
​
∑
t
=
1
|
o
−
|
log
⁡
π
θ
​
(
o
<
t
−
|
q
,
o
<
t
−
)
π
ref
​
(
o
<
t
−
|
q
,
o
<
t
−
)
)
(12)
The gradient of 
𝒥
D
​
P
​
O
​
(
θ
)
 is:

∇
θ
𝒥
D
​
P
​
O
​
(
θ
)
=
𝔼
​
[
q
∼
P
s
​
f
​
t
​
(
Q
)
,
o
+
,
o
−
∼
π
s
​
f
​
t
​
(
O
|
q
)
]
(
1
|
o
+
|
∑
t
=
1
|
o
+
|
G
C
D
​
P
​
O
(
q
,
o
,
t
)
∇
θ
log
π
θ
(
o
t
+
|
q
,
o
<
t
+
)
−
1
|
o
−
|
∑
t
=
1
|
o
−
|
G
C
D
​
P
​
O
(
q
,
o
,
t
)
∇
θ
log
π
θ
(
o
t
−
|
q
,
o
<
t
−
)
)
(13)
Data Source: question in SFT dataset with outputs sampled from SFT model. Reward Function: human preference in the general domain (can be ‘Rule’ in mathematical tasks). Gradient Coefficient:

G
​
C
D
​
P
​
O
​
(
q
,
o
,
t
)
=
σ
​
(
β
​
log
⁡
π
θ
​
(
o
t
−
|
q
,
o
<
t
−
)
π
ref
​
(
o
t
−
|
q
,
o
<
t
−
)
−
β
​
log
⁡
π
θ
​
(
o
t
+
|
q
,
o
<
t
+
)
π
ref
​
(
o
t
+
|
q
,
o
<
t
+
)
)
(14)
A.1.5Proximal Policy Optimization (PPO)
The objective of PPO is:

𝒥
P
​
P
​
O
​
(
θ
)
=
𝔼
​
[
q
∼
P
s
​
f
​
t
​
(
Q
)
,
o
∼
π
θ
o
​
l
​
d
​
(
O
|
q
)
]
​
1
|
o
|
​
∑
t
=
1
|
o
|
min
⁡
[
π
θ
​
(
o
t
|
q
,
o
<
t
)
π
θ
o
​
l
​
d
​
(
o
t
|
q
,
o
<
t
)
​
A
t
,
clip
​
(
π
θ
​
(
o
t
|
q
,
o
<
t
)
π
θ
o
​
l
​
d
​
(
o
t
|
q
,
o
<
t
)
,
1
−
ε
,
1
+
ε
)
​
A
t
]
.
(15)
To simplify the analysis, it is assumed that the model only has a single update following each exploration stage, thereby ensuring that 
π
θ
o
​
l
​
d
=
π
θ
. In this case, we can remove the 
min
 and 
clip
 operation:

𝒥
P
​
P
​
O
​
(
θ
)
=
𝔼
​
[
q
∼
P
s
​
f
​
t
​
(
Q
)
,
o
∼
π
θ
o
​
l
​
d
​
(
O
|
q
)
]
​
1
|
o
|
​
∑
t
=
1
|
o
|
π
θ
​
(
o
t
|
q
,
o
<
t
)
π
θ
o
​
l
​
d
​
(
o
t
|
q
,
o
<
t
)
​
A
t
.
(16)
The gradient of 
𝒥
P
​
P
​
O
​
(
θ
)
 is:

∇
θ
𝒥
P
​
P
​
O
​
(
θ
)
=
𝔼
​
[
q
∼
P
s
​
f
​
t
​
(
Q
)
,
o
∼
π
θ
o
​
l
​
d
​
(
O
|
q
)
]
​
1
|
o
|
​
∑
t
=
1
|
o
|
A
t
​
∇
θ
log
⁡
π
θ
​
(
o
t
|
q
,
o
<
t
)
(17)
Data Source: question in SFT dataset with outputs sampled from policy model. Reward Function: reward model. Gradient Coefficient:

G
​
C
P
​
P
​
O
​
(
q
,
o
,
t
,
π
θ
r
​
m
)
=
A
t
,
(18)
where 
A
t
 is the advantage, which is computed by applying Generalized Advantage Estimation (GAE) (Schulman et al., 2015), based on the rewards 
{
r
≥
t
}
 and a learned value function 
V
ψ
.

A.1.6Group Relative Policy Optimization (GRPO)
The objective of GRPO is (assume 
π
θ
o
​
l
​
d
=
π
θ
 for simplified analysis):

𝒥
G
​
R
​
P
​
O
​
(
θ
)
=
𝔼
​
[
q
∼
P
s
​
f
​
t
​
(
Q
)
,
{
o
i
}
i
=
1
G
∼
π
θ
o
​
l
​
d
​
(
O
|
q
)
]
1
G
​
∑
i
=
1
G
1
|
o
i
|
​
∑
t
=
1
|
o
i
|
[
π
θ
​
(
o
i
,
t
|
q
,
o
i
,
<
t
)
π
θ
o
​
l
​
d
​
(
o
i
,
t
|
q
,
o
i
,
<
t
)
​
A
^
i
,
t
−
β
​
(
π
r
​
e
​
f
​
(
o
i
,
t
|
q
,
o
i
,
<
t
)
π
θ
​
(
o
i
,
t
|
q
,
o
i
,
<
t
)
−
log
⁡
π
r
​
e
​
f
​
(
o
i
,
t
|
q
,
o
i
,
<
t
)
π
θ
​
(
o
i
,
t
|
q
,
o
i
,
<
t
)
−
1
)
]
.
(19)
The gradient of 
𝒥
G
​
R
​
P
​
O
​
(
θ
)
 is:

∇
θ
𝒥
G
​
R
​
P
​
O
​
(
θ
)
=
𝔼
​
[
q
∼
P
s
​
f
​
t
​
(
Q
)
,
{
o
i
}
i
=
1
G
∼
π
θ
o
​
l
​
d
​
(
O
|
q
)
]
1
G
​
∑
i
=
1
G
1
|
o
i
|
​
∑
t
=
1
|
o
i
|
[
A
^
i
,
t
+
β
​
(
π
r
​
e
​
f
​
(
o
i
,
t
|
o
i
,
<
t
)
π
θ
​
(
o
i
,
t
|
o
i
,
<
t
)
−
1
)
]
​
∇
θ
log
⁡
π
θ
​
(
o
i
,
t
|
q
,
o
i
,
<
t
)
.
(20)
Data Source: question in SFT dataset with outputs sampled from policy model. Reward Function: reward model. Gradient Coefficient:

G
​
C
G
​
R
​
P
​
O
​
(
q
,
o
,
t
,
π
θ
r
​
m
)
=
A
^
i
,
t
+
β
​
(
π
r
​
e
​
f
​
(
o
i
,
t
|
o
i
,
<
t
)
π
θ
​
(
o
i
,
t
|
o
i
,
<
t
)
−
1
)
,
(21)
where 
A
^
i
,
t
 is computed based on the group reward scores.