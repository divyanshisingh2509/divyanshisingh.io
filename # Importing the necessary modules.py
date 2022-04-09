# Importing the necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


plt.style.use('seaborn-dark')
plt.rcParams['figure.figsize']=14,6

# Initializing the random seed
random_seed=1000

# List containing the variance
# covariance values
cov_val = [-0.8, 0, 0.8]

# Setting mean of the distributino to
# be at (0,0)
mean = np.array([0,0])

# Iterating over different covariance
# values
for idx, val in enumerate(cov_val):
	plt.subplot(1,3,idx+1)
	
	# Initializing the covariance matrix
	cov = np.array([[1, val], [val, 1]])
	
	# Generating a Gaussian bivariate distribution
	# with given mean and covariance matrix
	distr = multivariate_normal(cov = cov, mean = mean,
								seed = random_seed)
	
	# Generating 5000 samples out of the
	# distribution
	data = distr.rvs(size = 5000)
	
	# Plotting the generated samples
	plt.plot(data[:,0],data[:,1], 'o', c='lime',
			markeredgewidth = 0.5,
			markeredgecolor = 'black')
	plt.title(f'Covariance between x1 and x2 = {val}')
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.axis('equal')
	
plt.show()
μ = np.array([.5, 1.])
Σ = np.array([[1., .5], [.5 ,1.]])

# construction of the multivariate normal instance
multi_normal = MultivariateNormal(μ, Σ)
k = 1 # choose partition

# partition and compute regression coefficients
multi_normal.partition(k)
multi_normal.βs[0],multi_normal.βs[1]
(array([[0.5]]), array([[0.5]]))
Let’s illustrate the fact that you can regress anything on anything else.

We have computed everything we need to compute two regression lines, one of  on , the other of  on .

We’ll represent these regressions as

and

where we have the population least squares orthogonality conditions

and

Let’s compute .

beta = multi_normal.βs 

a1 = μ[0] - beta[0]*μ[1]
b1 = beta[0]

a2 = μ[1] - beta[1]*μ[0]
b2 = beta[1]
Let’s print out the intercepts and slopes.

For the regression of  on  we have

print ("a1 = ", a1)
print ("b1 = ", b1)
a1 =  [[0.]]
b1 =  [[0.5]]
For the regression of  on  we have

print ("a2 = ", a2)
print ("b2 = ", b2)
a2 =  [[0.75]]
b2 =  [[0.5]]
Now let’s plot the two regression lines and stare at them.

z2 = np.linspace(-4,4,100)


a1 = np.squeeze(a1)
b1 = np.squeeze(b1)

a2 = np.squeeze(a2)
b2 = np.squeeze(b2)

z1  = b1*z2 + a1


z1h = z2/b2 - a2/b2


fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(1, 1, 1)
ax.set(xlim=(-4, 4), ylim=(-4, 4))
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.ylabel('$z_1$', loc = 'top')
plt.xlabel('$z_2$,', loc = 'right')
plt.title('two regressions')
plt.plot(z2,z1, 'r', label = "$z_1$ on $z_2$")
plt.plot(z2,z1h, 'b', label = "$z_2$ on $z_1$")
plt.legend()
plt.show()
_images/multivariate_normal_16_0.png
The red line is the expectation of  conditional on .

The intercept and slope of the red line are

print("a1 = ", a1)
print("b1 = ", b1)
a1 =  0.0
b1 =  0.5
The blue line is the expectation of  conditional on .

The intercept and slope of the blue line are

print("-a2/b2 = ", - a2/b2)
print("1/b2 = ", 1/b2)
-a2/b2 =  -1.5
1/b2 =  2.0
We can use these regression lines or our code to compute conditional expectations.

Let’s compute the mean and variance of the distribution of  conditional on .

After that we’ll reverse what are on the left and right sides of the regression.

# compute the cond. dist. of z1
ind = 1
z1 = np.array([5.]) # given z1

μ2_hat, Σ2_hat = multi_normal.cond_dist(ind, z1)
print('μ2_hat, Σ2_hat = ', μ2_hat, Σ2_hat)
μ2_hat, Σ2_hat =  [3.25] [[0.75]]
Now let’s compute the mean and variance of the distribution of  conditional on .

# compute the cond. dist. of z1
ind = 0
z2 = np.array([5.]) # given z2

μ1_hat, Σ1_hat = multi_normal.cond_dist(ind, z2)
print('μ1_hat, Σ1_hat = ', μ1_hat, Σ1_hat)
μ1_hat, Σ1_hat =  [2.5] [[0.75]]
Let’s compare the preceding population mean and variance with outcomes from drawing a large sample and then regressing  on .

We know that

which can be arranged to

We anticipate that for larger and larger sample sizes, estimated OLS coefficients will converge to  and the estimated variance of  will converge to .

n = 1_000_000 # sample size

# simulate multivariate normal random vectors
data = np.random.multivariate_normal(μ, Σ, size=n)
z1_data = data[:, 0]
z2_data = data[:, 1]

# OLS regression
μ1, μ2 = multi_normal.μs
results = sm.OLS(z1_data - μ1, z2_data - μ2).fit()
Let’s compare the preceding population  with the OLS sample estimate on 

multi_normal.βs[0], results.params
(array([[0.5]]), array([0.50082709]))
Let’s compare our population  with the degrees-of-freedom adjusted estimate of the variance of 

Σ1_hat, results.resid @ results.resid.T / (n - 1)
(array([[0.75]]), 0.7514957868891082)
Lastly, let’s compute the estimate of  and compare it with 

μ1_hat, results.predict(z2 - μ2) + μ1
(array([2.5]), array([2.50330838]))
Thus, in each case, for our very large sample size, the sample analogues closely approximate their population counterparts.

A Law of Large Numbers explains why sample analogues approximate population objects.