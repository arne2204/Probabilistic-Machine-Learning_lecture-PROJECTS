[27.Jun.2025] I am happy to see the progress in your report. Some comments:

- You correclty identify the oscilatory behavior of the ACF as indicating seasonality in the series.
- In the plot after: "kNDVI is plotted for each year separately and overlaid. We hope to see an overlap as an indicator for seasonality" indicate that you chose 46 as the period to make this plot, after checking the ACF oscilations below.
- After "We can do a seasonal trend decomposition using Loess (STL) to further investigate the seasonal patterns:" please indicate explicitly how this method works, it is part of the requirements of the project to explicitly indicate the mathematics behind the methods you use. You can start by explaining the general structure of the STL method as
  being a decomposition into **Trend**+**Seasonal**+**Residual** components of the series, and expand further on that...
- Afther explaining the structure and mathematics behind STL, explain the output more clearly. So far you just say that the plots "further confirms the assumption for seasonal patterns," which is correct, but in a report these statemets have to be made more clear for the readers. It is a good practice that will improve the readability of your document.
- It is good that you did a simple FFT analysis to support your findings with the ACF.
- That is all you have for now (do not worry, you have time) What is your intention next? This initial exploration phase is decent. But I suggest you do more checks, such as the cross-correlation between the different series. Some series might be highly correlated, and some might be weakly correlated. I actually would expect all of them to be rather highly correlated, as indicated already when visualizing few points, but it is always a good practive to see it directly. You can use: `from statsmodels.tsa.stattools import ccf`


Requests: 

- To make your project fulfill the main purpose of the lecture, which is to understand the probabilistic framework in ML (Probabilistic Machine Learning), you need to use a probabilistic method for time series analysis in the following parts of your project.
