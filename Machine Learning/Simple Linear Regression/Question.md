# Linear regression with one variable
In this part of this exercise, you will implement linear regression with one variable to predict
profits for a food truck. Suppose you are the CEO of a restaurant franchise and are considering
different cities for opening a new outlet. The chain already has trucks in various cities and you
have data for profits and populations from the cities.
You would like to use this data to help you select which city to expand to next.
The file [ex1data1.txt](https://github.com/emilmont/Artificial-Intelligence-and-Machine-Learning/blob/master/ML/ex1/ex1data1.txt) contains the dataset for our linear regression problem. The first column
is the population of a city and the second column is the profit of a food truck in that city. A
negative value for profit indicates a loss.  
• Plotting the Data
Before starting on any task, it is often useful to understand the data by visualizing it. For this dataset,
you can use a scatter plot to visualize the data, since it has only two properties to plot (profit and
population). (Many other problems that you will encounter in real life are multi-dimensional and can't
be plotted on a 2-d plot.)  
• Gradient Descent
In this part, you will fit the linear regression parameters Ѳ to our dataset using gradient descent.  
       - Update Equations  
       - Computing the cost J(Ѳ)  
• Visualizing J(Ѳ)  
To understand the cost function J(Ѳ) better, you will now plot the cost over a 2 dimensional grid  of Ѳ<sub>0</sub>
and Ѳ<sub>1</sub> values. 
