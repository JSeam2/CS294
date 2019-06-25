#  Week 2 Notes
## Auto-regressive models
## 1st Half
### Likelihood-based models
- Motivations:
    - Generating Data: synthesizing images, videos, speech, text
    - Compressing Data: constructing efficient codes
        - Compression is closely related to Generation
        - Compressing well is all about prediction. If you can predict well you don't
          need to store information.
    - Anomaly Detection

- Likelihood-based models: essentially a joint distribution over data
    - Given a dataset, sample the data and figure out the distribution
    - Once you have the distribution you can take a datapoint and find out the
      probability of the datapoint given the distribution
    - estimate p<sub>data</sub> from samples
    x<sup>(1)</sup>, ..., x<sup>(n)</sup> ~ p<sub>data</sub>(x)

- Learns a distribution p that allows:
    - Computing p(x) for arbitrary x
    - Sampling x ~ p(x), ie sample from the model

- Cases for both discrete data and continuous data


### Desiderata
- The methods in this course differ from classical methods of estimating
  distributions because we want to deal with **complex, high-dimensional data**

- 128 x 128 x 3 looks small on your computer screen but it is actually huge as
  it lies in a ~50,000-dimensional space

- Designing algorithms to handle high dimensionality has a lot of tradeoffs.
  **We want efficient algorithms (computational efficiency and statistical efficiency)**

    - Efficient training and model generation
    - Expressiveness and generalization
    - Sampling quality and speed
    - Compression rate and speed

### Estimating Frequencies by counting
- The goal of generating modelling is to estimate p<sub>data</data> from samples
  x<sup>(1)</sup>, ..., x<sup>(n)</sup> ~ p<sub>data</sub>(x)

- Suppose the samples take on values in a finite set {1,...,k} and the range is
  real number. We can describe the model as a **histogram**


- (Redundantly) described by k nonnegative numbers: p<sub>1</sub>,...,<sub>

- To train this model: count frequencies
    p<sub>i</sub> = (# times i appears in the dataset) / (# points in the dataset)

### At runtime
- **Inference** (querying p<sub>i</sub> for arbitrary i) simply a lookup into
the array p<sub>1</sub>, ..., p <sub>k</sub>

- **Sampling** (lookup into the inverse cuulative distribution function)
    1. From the model probabilities p<sub>1</sub>, ..., p<sub>k</sub> compute
       the cumulative distribution

        <img src="https://render.githubusercontent.com/render/math?math=F_{i} = p_{1} + ... + p_{i}">
        for all
        <img src="https://render.githubusercontent.com/render/math?math=i \in {1, ..., k}">

    2. Draw a uniform random number u ~ [0, 1]

    3. Return the smallest i such that
        <img src="https://render.githubusercontent.com/render/math?math=u \leq F_{i}">

    4. While this approach is good for low dimension it is insufficient at
       higher dimensions. Counting fails when there are too many bins

### Failure at high dimensions
- The reason why counting fails is due to the **curse of dimensionality**

- Consider MNIST which are 28 x 28 images and consider the binary case where
  each pixel is either {0, 1}

- In this case we have 2<sup>784</sup> probabilities to estimate. This is a huge number!

- If you tried to fit a histogram you cannot possibly fit this into a computer.
  In addition, the MNIST dataset is only 60,000 images and doesn't even scratch
  that large number. Each image influences only one parameter and we cannot
  generalize. Essentially, the model is just the dataset

- To solve this issue. We have to use **function approximation** instead of
  storing each probability we store a parameterized function
  <img src="https://render.githubusercontent.com/render/math?math=p_{\theta}(x)">
  this then allows us to generalize

### Likelihood-based generative models
- Recall: the goal is to estimate p<sub>data</sub> from
  x<sup>(1)</sup>,...,x<sup>(n)</sup> ~ p<sub>data</sub>(x)

- Now we introduce function approximation: learn theta so that
  <img src="https://render.githubusercontent.com/render/math?math=p_{\theta}(x) \approx p_{data}(x)">

- So now wwe face a few issues:
    - How do we design function approximators to effectively represent complex joint
    distributions over x, yet remain easy to train?
    - There will be many choices for model design, each with different tradeoffs and
    different compatibility criteria

- As such the model and training goes hand in hand, we have to design both of
  them at the same time

### Fitting distribution
- Essentially, given x<sup>(1)</sup>, ..., x<sup>(n)</sup> samples from "true"
  distribution p<sub>data</sub> set up a model class which is set of
  parameterized distributions p<sub>theta</sub>. Then solve the search problem
  over parameters
  <img
  src="https://render.githubusercontent.com/render/math?math=argmin_{\theta} loss(\theta, x^{1}, ..., x^(n)">

- It is easy to fit data over small dataset but it is hard when it comes to
  large datasets

- The biggest challnge is that we have loss sample that is supposed to
  generalize well but we only have a finite set of samples. Finding the true
  distribution is going to be tricky

### Maximum likelihood
- Given a dataset find theta by solving the optimization problem

  <img
  src="https://render.githubusercontent.com/render/math?math=argmin_{\theta} loss(\theta, x^{1}, ..., x^(n) = \frac{1}{n} \sum^{n}_{i=1} - \log p_{\theta}(x^{(i)})">
- Maximum likelihood is the loss function which satisfy the issues faced when
  fitting distribution

- The reason why we use maximum likelihood vs other methods like methods of
  moments is because Maximum likelihood works in practice

- Why maximum likelihood works in practice, is that if you do have enough data
  and the model is expressive enough. Then maximum likelihood would find you a model.

- This is also equivalent to minimizing KL divergence between the empirical data
  distribution and the model.

### Stochastic Gradient Descent
- Since maximum likelihood works this gives us the gradient descent algorithm
- SGD minimizes the expectation of f and the noise due to sampling from the dataset
- This is done because it works in practice

### Designing the model
- The key requirement for MLE (Max likelihood) + SGD is that we must be able to
  compute log p(x) and its gradient efficiently
- We will choose models p<sub>theta</sub> to be deep neural networks, which work
  in the regime of high expressiveness and efficient computation (assuming
  specialized hardware)
- To design these network we need the following
    - Any setting of theta must define a valid probability distribution over x
    for all <img src="https://render.githubusercontent.com/render/math?math=\theta">,
    <img src="https://render.githubusercontent.com/render/math?math=\sum_{x}(p_{\theta}(x)) = 1">
    and
    <img src="https://render.githubusercontent.com/render/math?math=p_{\theta}(x) \geq 0">
    for all x

    - <img src="https://render.githubusercontent.com/render/math?math=\log{p_{\theta}(x)}">
     should be easy to evaluate and differentiate wrt to theta

    - This can be tricky to set up

### Bayes nets and neural nets
- Main idea: place a Bayes net structure (a directed acyclic graph) over the
  variables in the data, and model the conditional distributions with neural
  networks

- Reduces the problem to designing conditional likelihood-basedm models for
  single variables. We know how to do this: the neural net takes variables being
  conditioned on as input, and outputs the distribution for the variable being predicted.

- In way, neural nets condition alot of stuff and reduces the conditions to a
  single variable

### Autoregressive models
- The bayes nets fit nicely into autoregressive models.
- First, given a Bayes net structure, setting the conditional distributiosn to
  neural networks will yield a tractable log likelihood and gradient. Great for
  maximum likelihood training

    <img
    src="https://render.githubusercontent.com/render/math?math=\log{p_{\theta}(x)} = \sum^{d}_{i=1} \log{p_{\theta} (x_{i} | parents(x_{i}))}">

- This model is also expressive enough. If we assume a fully expressive Bayes
  net structure any joint distribution can be written as a product of
  conditionals. You will not lose expressiveness when you write like this.
    <img
    src="https://render.githubusercontent.com/render/math?math=\log{p(x)} = \sum^{d}_{i=1} \log{p (x_{i} | x_{i:i-1})}">

- This is called an autoregressive model. So, an expressive Bayes net structure
  with neural network conditional distributions yields an expressive model for
  p(x) with tractable maximum likelihood training.


### Toy autoregressive models
Two variables: x<sub>1</sub> x<sub>2</sub>

Model: p()x<sub>1</sub>, x<sub>2</sub> = p(x<sub>2</sub>|x<sub>1</sub>)

- p(x<sub>1</sub>) is a histogram
- p(x<sub>2</sub>|x<sub>1</sub>) is a multilayer perceptron
    - Input is x<sub>1</sub>
    - Output is a distribution over x<sub>2</sub> (logits, followed by softmax)

### One function approximator per conditional
Does this extend to high dimensions?

- Somewhat. The model scales linearly with dimensions. For d-dimensional data,
  O(d) parameters
    - This is much better than O(exp(d)) in tabular case
    - In text generation a sentence can be arbitrarily long things don't look
      too good
- Limited generalization
    - No information sharing among different conditionals

- Solution: share parameters among conditional distributions. Two approaches
    - RNN
    - Masking

### RNN autoregressive models - char-rnn
- An RNN fits in to the autoregressive model. You can think of it as a machine
  that takes in sequences and outputs sequences
- For example an RNN takes an intial probability x<sub>1</sub> and returns an
  output probability x<sub>2</sub> so on.

  <img src="https://render.githubusercontent.com/render/math?math=\log{p(x)} = \sum^{d}_{i=1} \log(x_{i} | x_{1:i-1})">


## 2nd Half
- In the 1st half the models are fairly new. In this half most of the models are
  not more than 4 years old as of 2019

### Masking-based autoregressive models
- Second major branch of neural AR models
    - Key property: parallelized computation of all conditionals
    - Masked MLP (MADE)
    - Masked convolutions & self-attention
        - Also share parameters across time

- Sometimes adding more layers just does not help. This is because the current
  model misses some statistical dependencies and adding more layers won't solve
  that problem

- One approach is to design statistical models with certain desired properties
  and then use neural networks to model upon the statistical models

- The subsequent approaches are clever ways to augment supervised deep learning.

### Masked Autoencoder for Distribution Estimation (MADE)
- MORE INFO at [youtube around 1:46-1:47](https://youtu.be/zNmvH6OXDpk?t=6430)
- How can we turn an MLP which is deep net into an autoregressive model? The
  answer is MADE
- A autoencoder is essentially a function to reconstruct the input via some
  hidden representation.
- The MADE paper asks how can we use an autoencoder to become a distribution estimator
- The way we do it is to mask out some of the MLP weights so that the output
  contains neurons which are the conditioned probability distribution
- Example we have 3 inputs x1 x2 x3, and we want to obtain p(x1 | x2, x3),
  p(x2), p(x3 | x2). We now need to find a masking of the weights of the neural
  net to obtain the result. To do this we can split the hidden neurons into
  neuron groups.
- Benefits of MADE is that likelihood estimation is quick. We just need to do
  one forward pass to get all the conditionals. Where as in sampling we will
  need to obtain the conditionals sequentially.
- TODO: Read the paper

### Masked Temporal (1d) Convolution
- Output node is essentially p(x<sub>i + 1</sub> | x<sub><= i</sub>)
- This is easy to implement, we just mask part of the convolution kernel
- Const parameter count for variable length distribution!
- Efficient to compute, convolution has hyper-optimized implementation on all
  hardware
- However, limited receptive field, linear in number of layers

### Wavenet
- Introduces a solution the limited receptive field by using dilated
  convolution.

### Masked Spatial (2D) Convolution - PixelCNN
- image scan be flatten into 1D vectors, but they are fundamentally 2D
- We can use a masked variant of ConvNet to exploit this knowledge
- First, we impose an autoregressive ordering on 2D images. One popular ordering
  is the **raster scan ordering**.
    - We impose an autoregressive ordering so that can figure out the
      probability distribution of a given pixel
- Design question: how to design a masking method to obey that ordering?
- One possibility: PixelCNN (2016)
    - PixelCNN-style masking has one problem: blind spot in receptive field

### Gated PixelCNN
- Intorduces a fix to the blind spot by combining two streams of convolution
- There is a vertical stack and a horizontal stack which is dependent on 1d conv
- Finding vertical stack is tricky. This is solved in Gated PixelCNN by padding
- Improved ConvNet architecture: Gates ResNet Block

### PixelCNN++
- Move away from softmax: we know nearby pixel values are likely to co-occur.
  Using softmax might not be the best loss
