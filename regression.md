
# Properties of the regression's coefficients

Regression cames from the latin word regressus, which means "to return". In statistics, we are interest in study a variable $y$, one should test if $y$ returns to a model randonly and in an unbiased way.

The simplest mathematical model, the linear relation between the variable $y$ and $X$ via a scalar response parameter $\beta$, can be inspect using the linear regression method.

In other words, it searchs the vector $y$ as a linear combination of $X$ vector columns.

The parameters $\beta$ can be estimated (find $\hat{\mathbf{\beta}}$) using least-squares estimation as follows
\begin{equation}
 \hat{\mathbf{\beta}} = \text{argmin}_{\mathbf{\beta}} \|\mathbf{y}- \mathbf{X}\mathbf{\beta}  \|
\end{equation}
which leads to the so called normal equations. Solving it for $\mathbf{\beta}$ results 
\begin{equation}
 \hat{\beta} = ( \mathbf{X}^{T} \mathbf{X})^{-1} \mathbf{X}^{T} \mathbf{y}.
\end{equation}

This elegantly simple expression is a well known result of the best (minimum variance) linear unbised ($\mathbf{E(\beta) = \hat{\beta}}$) estimator. 

But despite its algebraic elegance, much of the intuition regarding the interplay between $\mathbf{X}$ and $\mathbf{y}$ is concealed within it's hermetic form. 

For one dimension the coefficient reduce to $\mathbf{\hat{\beta}} = \rho_{x,y} \frac{\sigma_y}{\sigma_x}$, but for higher dimentions the relation between correlations is entangled.  

To explore this intuition, we are going to solve $\mathbf{\hat{\beta}}$ when $\mathbf{X}$ has one, two and three dimensions. To then check the interplay between correlations specially for dimension greater then two.

Finally, we are going to compare the result with the parameter output of OLS statsmodels using python.

In this case, the $\mathbf{y}$ will be the return of the net asset value (NAV) of a portfolio composed by different types of assets, so the $\mathbf{X}$ variable are the returns of those assets, and finally the $\mathbf{\beta}$ will be an estimation of the weight invested in each asset class.


# Application in Portfolio Positioning

The value invested in the portfolio at $t$ consist of the amount $P_{t}$ of money distributed among assets, one buy or sell $n_{i}$ units contracts of asset $i$ with unit value $a_{i,t}$, therefore

\begin{equation}
P_{t} = \sum_{i=1}^{N} n_{i} a_{i,t}
\end{equation}

and the return on a $\tau$ scale is the following

\begin{equation}
\tag{1}
\frac{P_{t}}{P_{t-\tau}} = \frac{\sum_{i=1}^{N} n_{i} a_{i,t}}{\sum_{i=1}^{N} n_{i} a_{i,t-\tau}}
\end{equation}

note that one can write the present value of a asset $i$ as its the return 
\begin{equation}
a_{i,t} =  a_{i,t-\tau} (1 + r_{i,t})
\end{equation}

substituting in Eq. (1) and  dividing by $P_{t-\tau}$ on both numerator and denominator in the left-hand side lead us to
\begin{equation}
\frac{P_{t}}{P_{t-\tau}} = \frac{\sum_{i=1}^{N} \frac{ n_{i} a_{i,t-\tau} }{ P_{t-\tau}} (1 + r_{i,t})}{\sum_{i=1}^{N} \frac {n_{i} a_{i,t-\tau}}{P_{t-\tau}} }
\end{equation}

note that $w_{i,t} =  n_{i} a_{i,t} / P_{t}$ are the weight invested in each asset class and $ R_{t} + 1 =  P_{t}/ P_{t-\tau}$


\begin{equation}
R_{t} + 1 = \frac{\sum_{i=1}^{N} w_{i,t-\tau} (1 + r_{i,t}) }{\sum_{i=1}^{N} w_{i,t-\tau} }
\end{equation}
and since $\sum_{i=1}^{N} w_{i,t-\tau} = 1$  


\begin{equation}
R_{t} = \sum_{i=1}^{N} w_{i,t-\tau} r_{i,t} 
\end{equation}

The return of the net asset value is a linear combination of the return of each asset.

Therefore, the goal is to estimate the weights $w_{i}$ invested in each asset knowing just the timeseries $R_{t}$ (this will be our $y$) and the $X$ will be a set of timeseries of asset class.

\begin{equation}
\hat{R}_{t} = \sum_{i \in \mathcal{S}^{*}} \hat{\beta}_{i,t-\tau} r_{i,t} 
\end{equation}
where $\mathcal{S}^{*}$ is the set of factors we choose to regress the net asset return.
This leads us to define the set $\mathcal{S}$ having all asset invested in our portfolio.
If we get luck the matching asset live in $\mathcal{S} \cap \mathcal{S}^{*}$.

\begin{equation}
R_{t}  - \hat{R}_{t} = \sum_{i \in \mathcal{S} \cap \mathcal{S}^{*} } (w_{i,t-\tau} - \hat{\beta}_{i,t-\tau} ) r_{i,t}  + 
\sum_{i \in \mathcal{S} / \mathcal{S}^{*} } w_{i,t-\tau} r_{i,t}  -
\sum_{i \in \mathcal{S}^{*} / \mathcal{S} } \hat{\beta}_{i,t-\tau} r_{i,t} 
\end{equation}

Calling $R_{t}  - \hat{R}_{t} = \epsilon$ as the residual observable error, then it follows

\begin{align}
\epsilon  &= \sum_{\mathcal{S} \cap \mathcal{S}^{*} } (w - \hat{\beta} )   + 
\sum_{\mathcal{S} / \mathcal{S}^{*} } w   - \sum_{\mathcal{S}^{*} / \mathcal{S} } \hat{\beta}  \\
\epsilon &= \sum_{\mathcal{S} \cap \mathcal{S}^{*} } (w - \hat{\beta} )   + 
\sum_{\mathcal{S} / \mathcal{S}^{*} } w - \sum_{\mathcal{S}^{*} / \mathcal{S} } w + \sum_{\mathcal{S}^{*} / \mathcal{S} } w  - \sum_{\mathcal{S}^{*} / \mathcal{S} } \hat{\beta} \\
\epsilon  &= \sum_{\mathcal{S} \cap \mathcal{S}^{*} } (w - \hat{\beta} )   + 
\sum_{\mathcal{S}^{*} / \mathcal{S} } (w - \hat{\beta} ) + \sum_{\mathcal{S} / \mathcal{S}^{*} } w - \sum_{\mathcal{S}^{*} / \mathcal{S} } w   \\
\epsilon  - \sum_{\mathcal{S} / \mathcal{S}^{*} } w + \sum_{\mathcal{S}^{*} / \mathcal{S} } w   &= \sum_{ \mathcal{S}^{*} } (w - \hat{\beta} ) \\
\end{align}

The LHS is the error of the estimation $\epsilon_{\mathcal{S} \cap \mathcal{S}^{*}}$ due to join factors. 
The second term in the RHS is not observed and is the error related to unknown variable $\epsilon_{\mathcal{S} / \mathcal{S}^{*}} $.
The last term can be rewrite to as zero weighted 

\begin{equation}
\overbrace{
\underbrace{\epsilon }_\text{obs. error} - 
\underbrace{\epsilon_{\mathcal{S} / \mathcal{S}^{*}} }_\text{unknow factor}}^\text{pair error}   = \sum_{i \in \mathcal{S}^{*} } (w_{i,t-\tau} - \hat{\beta}_{i,t-\tau} ) r_{i,t} 
\end{equation}

In this execise, let us consider a portfolio investing and keeping the weights as follows: 

* 20\% of Brazilian Real currency, 
* 5\% of Bitcoin, 
* 60\% of US 10 year bonds and 
* 15\% of S&P ETFs.