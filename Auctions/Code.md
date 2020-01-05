# Question. 
Write a program to estimate nonparametrically the underlying distribution of valuations from data on the number of bidders and the prices
paid in a sample of ascending auctions, assuming symmetric independent private values, no reserve price, and the “button auction” model of
Milgrom and Weber (1982). The data set is in the ascii file ascending_data.dat and contains two columns: the first gives the number of
bidders, the second gives the price paid. The data come from 600 simulated auctions. Run your program on these data. 
Turn in a printout of your code and a graph showing a plot of your estimated CDF of bidders’ private values. 
Be sure to choose the scaling of the plot so that the graph is informative.

# Solution
We needed to use an equation derived based on order statistics. Please find [here](https://github.com/10avinash/Project-Portfolio/tree/master/Auctions), the [equation](https://github.com/10avinash/Project-Portfolio/blob/master/Auctions/Equation.png) and the report with the [code](https://github.com/10avinash/Project-Portfolio/blob/master/Auctions/ps5_1.pdf).
