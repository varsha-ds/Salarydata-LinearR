# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline


# %%
df = pd.read_csv('Salary_dataset.csv')

# %%
df.head()

# %%
df.shape

# %% [markdown]
# There are 30 records and 3 columns

# %%
#printing the first column
df.iloc[:,0]

# %% [markdown]
# The first column represents the index values --> number of values, so we can drop the column

# %%
#del df['Unnamed: 0'] #delets the column entirely from the dataset
df = df.drop('Unnamed: 0',axis=1)

# %%
df.nunique()

# %% [markdown]
# We have 2 columns with numeric features 

# %%
df.describe()

# %% [markdown]
# The data set contains 30 observations with 
# mean of years of experience is nearly 5 and std 2.8 indicating some variability
# min exp = 1.2
# max experience = 10.6
# 
# Mean of salary is nearly 76004 and std of 27414 indicating considerable variability
# min salary = 37732
# max salary = 122392
# 
# 50% represents the median(The median represents the middle value of the dataset when it is sorted in ascending order)  - for YearsExperience, it is 4.8 :  This indicates that half of the individuals in the dataset have less than or equal to 4.8 years of experience, while the other half have more than or equal to 4.8 years of experience. 
# We can say that the data is evenly spread without any skewness.
# 
# If the distribution has significant skewness or imbalance, we may expect the median to be pushed to one end of the range, indicating that a large proportion of people have either fewer or more years of experience than the rest of the data. However, in this example, the median being around the middle of the range indicates a more equally distributed distribution of years of experience.
# 
# As the years of experience increases salary is increasing, but further analysis is needed,
# Possibility of outliers

# %%
df.isnull().sum()


# %%
#creating scatter plot to analyse the relationship

# Create a scatter plot 
plt.figure(figsize=(8, 6))  # Set the size of the plot
plt.scatter(df['YearsExperience'], df['Salary'], color='blue', alpha=0.5)  # Create the scatter plot, alpha specifies transparency of data points
#sns.scatterplot(x='YearsExperience', y='Salary', data=df, color='blue', alpha=0.5)  # Create the scatter plot using Seaborn
plt.title('Scatter Plot of Years of Experience vs. Salary')  # Set the title of the plot
plt.xlabel('Years of Experience')  # Set the label for the x-axis
plt.ylabel('Salary')  # Set the label for the y-axis
plt.grid(True)  # Add gridlines to the plot
plt.show()  # Show the plot


# %%
# Create a histogram for YearsExperience using Matplotlib
plt.figure(figsize=(8, 6))  # Set the size of the plot
#plt.hist(df['YearsExperience'], bins=10, color='skyblue', edgecolor='black')  # Create the histogram
sns.histplot(df['YearsExperience'],bins = 10, color='skyblue', edgecolor='black',kde=True)
plt.title('Histogram of Years of Experience')  # Set the title of the plot
plt.xlabel('Years of Experience')  # Set the label for the x-axis
plt.ylabel('Frequency')  # Set the label for the y-axis
plt.grid(True)  # Add gridlines to the plot
plt.show()  # Show the plot

# %%
# Create a histogram for Salary using Matplotlib
plt.figure(figsize=(8, 6))  # Set the size of the plot
#plt.hist(df['Salary'], bins=10, color='skyblue', edgecolor='black')  # Create the histogram
sns.histplot(df['Salary'],bins = 10, color='skyblue', edgecolor='black',kde=True)
plt.title('Histogram of Years of Experience')  # Set the title of the plot
plt.xlabel('Years of Experience')  # Set the label for the x-axis
plt.ylabel('Frequency')  # Set the label for the y-axis
plt.grid(True)  # Add gridlines to the plot
plt.show()  # Show the plot

# %%
# Create a box plot for both YearsExperience and Salary using Seaborn
plt.figure(figsize=(8, 6))  # Set the size of the plot
sns.boxplot(data=df[['YearsExperience']], palette='pastel')  # Create the box plot using Seaborn
plt.title('Box Plot of Years of Experience')  # Set the title of the plot
plt.ylabel('Value')  # Set the label for the y-axis
plt.grid(True)  # Add gridlines to the plot
plt.show()  # Show the plot


plt.figure(figsize=(8, 6))  # Set the size of the plot
sns.boxplot(data=df[['Salary']], palette='pastel')  # Create the box plot using Seaborn
plt.title('Box Plot of Salary')  # Set the title of the plot
plt.ylabel('Value')  # Set the label for the y-axis
plt.grid(True)  # Add gridlines to the plot
plt.show()  # Show the plot

# %% [markdown]
# -->Rightly/positively skewed
# 
# For the "YearsExperience" column, positive skewness may indicate that there are more persons with more years of experience than the median.
# For the "Salary" column, positive skewness may indicate that there are more persons with greater earnings than the median.

# %%
plt.figure(figsize=(8, 6))  # Set the size of the plot
sns.violinplot(y=df['YearsExperience'], color='skyblue')  # Create the violin plot for 'YearsExperience'
plt.title('Violin Plot of Years of Experience')  # Set the title of the plot
plt.ylabel('Years of Experience')  # Set the label for the y-axis
plt.grid(True)  # Add gridlines to the plot
plt.show()  # Show the plot

# %%
# Create a violin plot for 'Salary'
plt.figure(figsize=(8, 6))  # Set the size of the plot
sns.violinplot(y=df['Salary'], color='lightgreen')  # Create the violin plot for 'Salary'
plt.title('Violin Plot of Salary')  # Set the title of the plot
plt.ylabel('Salary')  # Set the label for the y-axis
plt.grid(True)  # Add gridlines to the plot
plt.show()  # Show the plot


# %%
# Create a heatmap for the correlation matrix
plt.figure(figsize=(8, 6))  # Set the size of the plot
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")  # Create the heatmap for the correlation matrix
plt.title('Heatmap of Correlation Matrix')  # Set the title of the plot
plt.show()  # Show the plot

# %% [markdown]
# correlation coefficient close to 1 shows a strong positive relationship between the 2 variables
# If the correlation coefficient of 0.98 pertains to the relationship between 'YearsExperience' and 'Salary', it suggests that there is a strong positive association between the number of years of experience and salary.
# Specifically, individuals with higher years of experience tend to have higher salaries, and individuals with lower years of experience tend to have lower salaries.
# 
# The strong correlation coefficient of 0.98 implies that the relationship between the two variables can be adequately described by a linear equation.
# Changes in one variable are directly proportional to changes in the other variable, following a straight-line pattern.
# 
# --> knowing a person's years of experience could potentially allow you to make accurate predictions about their salary, and vice versa.

# %%
# Create a regression plot with custom line color
plt.figure(figsize=(8, 6))  # Set the size of the plot
sns.regplot(x="YearsExperience", y="Salary", data=df, line_kws={"color": "red"})  # Create the regression plot with custom line color
plt.title('Regression Plot of Salary vs Years of Experience')  # Set the title of the plot
plt.xlabel('Years of Experience')  # Set the label for the x-axis
plt.ylabel('Salary')  # Set the label for the y-axis
plt.grid(True)  # Add gridlines to the plot
plt.show()  # Show the plot

# %%


# Calculate Z-scores for the data column
z_scores = (df['YearsExperience'] - df['YearsExperience'].mean()) / df['YearsExperience'].std()

# Set the threshold for outlier detection (typically |Z| > 3)
threshold = 2

# Find the indices of outliers based on Z-scores
outliers_indices = np.where(np.abs(z_scores) > threshold)[0]

# Print the indices of the outliers
print("Indices of outliers detected by Z-score analysis:", outliers_indices)


# %%
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df' is your pandas DataFrame containing the data
# 'YearsExperience' and 'Salary' are the columns you want to analyze

# Perform independent samples t-test
t_stat, p_value = stats.ttest_ind(df['YearsExperience'], df['Salary'])
print("Independent Samples t-test:")
print("t-statistic:", t_stat)
print("p-value:", p_value)



# %% [markdown]
# Null Hypothesis (H0): There is no significant difference between the mean of 'YearsExperience' and the mean of 'Salary'.
# 
# Alternative Hypothesis (H1): There is a significant difference between the mean of 'YearsExperience' and the mean of 'Salary'.
# 
# The aim of the t-test is to assess whether the observed difference in means between the two groups (YearsExperience and Salary) is statistically significant or if it could have occurred due to random sampling variability. The p-value obtained from the test helps determine the likelihood of observing the data if the null hypothesis were true. A low p-value (typically below a chosen significance level, such as 0.05) suggests that the null hypothesis should be rejected in favor of the alternative hypothesis, indicating a statistically significant difference between the means.
# 
# Interpretation:
# 
# The t-statistic measures the difference between the means of the two groups relative to the variation in the data. In this case, the negative t-statistic indicates that the mean of 'YearsExperience' is significantly lower than the mean of 'Salary'.
# The p-value is extremely small (7.21e-22), indicating strong evidence against the null hypothesis. This suggests that the difference in means between 'YearsExperience' and 'Salary' is statistically significant.
# 
# Conclusion:
# 
# Based on the low p-value, we reject the null hypothesis and conclude that there is a statistically significant difference between the means of 'YearsExperience' and 'Salary'.

# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 'YearsExperience' is the independent variable and 'Salary' is the dependent variable

# Split the dataset into training and testing sets
X = df[['YearsExperience']]  # Independent variable
y = df['Salary']  # Dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the metrics
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)
print("Intercept:", model.intercept_)
print("coefficient:", model.coef_)

# %%
# Calculate residuals
residuals = y_test - y_pred

# Residual vs. Fitted Values Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual vs. Fitted Values Plot')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# Residual Histogram
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.title('Residual Histogram')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Q-Q Plot
import scipy.stats as stats
plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.grid(True)
plt.show()


# %%
import statsmodels.api as sm


# Add a constant term for the intercept
X = sm.add_constant(df['YearsExperience'])

# Fit the linear regression model
model = sm.OLS(df['Salary'], X).fit()

# Print the summary of the regression model
print(model.summary())


# %% [markdown]
# Overall, both models produce consistent results, indicating a significant positive association between YearsExperience and Salary. The models account for a considerable percentage of the variation in Salary and have high prediction accuracy.


