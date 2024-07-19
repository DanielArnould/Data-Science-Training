# Naive Bayes Classifier

## What's the Problem?

You've been getting a lot more spam bots texting you lately, and instead of blocking them manually, you've decided to build a classifier which can automatically predict whether a message is spam or not. You've noticed that messages from spam bots tend to have different vocabulary compared to what your friends send you, so you take a sample of messages and plot how many times certain words appear.

![Word Frequency Column Chart](./imgs/mn_naive_bayes_plot1.png)

Clearly, if some words appear more than others, it is more likely that the message is from a spam bot. So, could we build a probabilistic classifier around that?

## What is a Naive Bayes Classifier?

The idea behind Naives Bayes Classifiers is actually quite simple. Let's say we had $K$ classes and $n$ features, then if we assigned probabilities $p(C_k | x_1, ..., x_n)$ to each class $C_k$ given we have some feature vector $\bold{x} = (x_1, ..., x_n)$, then the model's best prediction would just be the class with the highest probablitiy.

For example, let's say we got a text with 4 occurences of the word "food", 2 occurences of the word "girly", 6 occurences of the word "love", and 0 occurrences of the word "money". We could represent that text as a vector $\bold{x} = (4, 2, 6, 0)$. Now, let's say that $p(\text{Spam Message} | \bold{x}) = 0.19$ and $p(\text{Non-spam Message} | \bold{x}) = 0.6$, then we would confidently say that the message is not spam.

The problem is how do we work out $p(C_k | \bold{x})$? Fortunately, [Bayes' Theorem](https://www.wikiwand.com/en/Bayes'_theorem) comes to the rescue, because it tells us the conditional probabilitiy can be expressed as

$
p(C_k | \bold{x}) = \frac{p(C_k) p(\bold{x} | C_k)}{p(\bold{x})}
$

Often in Bayesian probability, the above equation will also be phrased as follows

$
\text{posterior} = \frac{\text{prior} \times \text{likelihood}}{\text{evidence}}
$

We notice that for all $C_k$, the denominator $p(\bold{x})$ doesn't change, so it's effectively a constant. Since we don't really care about the actual probability values but rather the predictions, we can just get rid of it for now to simplify our equations.

$
p(C_k | \bold{x}) \propto p(C_k) p(\bold{x} | C_k)
$

> $\propto$ is the "proportional to" symbol, since strictly speaking the terms aren't equal

Now, by using the [chain rule](https://www.youtube.com/watch?v=v8Uw1TFl2WQ) repeatedly and by assuming that each of the features of $\bold{x}$ are independent (see [here](https://www.youtube.com/watch?v=dNhdefN36E4) for details), we can get the following expression

$
p(C_k | \bold{x}) \propto p(C_k)\prod_{i=1}^{n}p(x_i|C_k)
$

That means, with some feature vector $\bold{x}$, our prediction $\hat{y}$ will be

$
\hat{y} = \underset{k \in \left\{ 0, ..., K \right\}}{\text{argmax}} p(C_k)\prod_{i=1}^{n}p(x_i|C_k)
$

Sometimes, however, some probabilities will be so small that the actual numbers risk underflowing on a computer, causing weird undefined behaviour. For that reason, some models will just take the log probability of everything.

$
\hat{y} = \underset{k \in \left\{ 0, ..., K \right\}}{\text{argmax}} \log{(p(C_k))} + \sum_{i=1}^{n} \log{(p(x_i|C_k))}
$

## Types of Naive Bayes Classifier

There are many different types of Naive Bayes Classifiers depending on what sort of probability distributions the feature variables are sampled from.

- **Bernoulli Naive Bayes**: This is used with vectors with Boolean variables, such as $\bold{x} = (1, 1, 0, 0, 1)$.
- **Multinomial Naive Bayes**: This is used with features from multinomial distributions. This is especially useful for non-negative discrete data, such as frequency counts. The above examples of counting words in a text message are a good example of this. Here, if the probability of seeing the word in a text message given a certain class is $p_k$, the likelihood of it appearing $c$ times would be $p_k^c$.
- **Gaussian Naive Bayes**: This is used with Gaussian/normal distributions. Typically when our feature variables are continuous, we assume that they're sampled from a gaussian distribution, and we thus calculate the likelihood $p(x_i | C_k)$ as follows

$
p(x_i | C_k) = \frac{1}{\sqrt{2 \pi \sigma_k^2}}e^{-\frac{(x_i - \mu_k)^2}{2 \sigma_k^2}}
$

> - $\sigma_k^2$ is the variance of $x_i$ associated with the class $C_k$
> - $\mu_k$ is the mean of $x_i$ associated with the class $C_k$

## Alpha Value

Let's say you've received a text message with 20 occurrences of the word "money" and 1 occurrence of the word "food". Just looking at the column chart above, you'd expect that the text message would be labelled as spam. But since there aren't any spam messages in the training data with the word "food", the likelihood of having 1 occurrence in a spam message is 0, meaning $p(\text{Spam Message} | (1, 0, 0, 20)) = 0$. This is clearly a problem, so what we do is something called [Laplace smoothing](https://datascience.stackexchange.com/questions/30473/how-does-the-mutlinomial-bayess-alpha-parameter-affects-the-text-classificati). Effectively, we inflate the occurences of everything in our training set by $\alpha$ to avoid multiplying anything by 0.

![Laplace Smoothed Multinomial Naive Bayes](./imgs/mn_naive_bayes_plot2.png)

## Exercise

### Multinomial Classifier

### Gaussian Classifier

## Extra Reading: Why is it called naive?
