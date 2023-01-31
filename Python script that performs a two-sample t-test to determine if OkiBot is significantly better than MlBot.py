import numpy as np
import scipy.stats as stats

# Make a txt doc to keep results
txt = open("Chi2Resuts.txt","w+")

# Sample data
okibot_wins = [10, 10, 10]
okibot_losses = [5, 5, 5]
mlbot_wins = [12, 11, 9]
mlbot_losses = [3, 4, 6]

# Calculate average win rates
okibot_mean = np.mean(okibot_wins) / (np.mean(okibot_wins) + np.mean(okibot_losses))
mlbot_mean = np.mean(mlbot_wins) / (np.mean(mlbot_wins) + np.mean(mlbot_losses))

# Calculate the standard error of the difference in means
se_diff = np.sqrt((np.var(okibot_wins) / len(okibot_wins)) + (np.var(mlbot_wins) / len(mlbot_wins)))

# Calculate the t-statistic
t_stat = (okibot_mean - mlbot_mean) / se_diff

# Calculate the degrees of freedom
dof = len(okibot_wins) + len(mlbot_wins) - 2

# Calculate the p-value
p_value = stats.t.sf(np.abs(t_stat), dof) * 2

# Significance level
alpha = 0.05

# Make a decision and wirte it in txt file
if p_value < alpha:
    txt.write("Reject the null hypothesis: There is a significant difference in the win rates of OkiBot and MlBot.")
else:
    txt.write("Fail to reject the null hypothesis: There is not a significant difference in the win rates of OkiBot and MlBot.")

# Report results
print("t-statistic: ", t_stat)
print("p-value: ", p_value)

# Write eport results in txt file
tstatistic = "t-statistic: " + str(t_stat)
pvalue = "p-value: " + str(p_value)

txt.write(tstatistic)
txt.write(pvalue)
