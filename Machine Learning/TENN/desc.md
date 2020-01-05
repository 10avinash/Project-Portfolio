# Description
This is a part of my Senior thesis (Final Year B.tech project). I wanted to estimate causal relationships using transfer entropy.
The problem with current libraries (there are some in R) is you would need to provide the lag value between two time series to estimate
the causal relationship. However, in the real world, we probably don't know that or can't be sure. Hence, we built a neural network which
not only estimates the transfer entropy but also the underlying lag values. I generated causal time series with known lag values in my native
language-python, utilized an R library to calculate transfer entropy, and trained a simple two layered neural network. It gave a pretty good
result.