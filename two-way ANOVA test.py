import pandas as pd
from scipy.stats import f_oneway

data = {'Player': ['Leen', 'Leen', 'Tobias', 'Tobias', 'Jasper', 'Jasper'],
        'Bots': ['OkiBot', 'MlBot', 'OkiBot', 'MlBot', 'OkiBot', 'MlBot'],
        'Wins': [10, 12, 10, 11, 10, 9]}
df = pd.DataFrame(data)

# Calculate wins for each player and bot combination
df_melted = df.melt(id_vars=['Player', 'Bots'], value_vars='Wins')

# Perform two-way ANOVA test
f_statistic, p_value = f_oneway(df_melted[df_melted['Bots'] == 'OkiBot']['value'],
                                df_melted[df_melted['Bots'] == 'MlBot']['value'])

# Print results
x = ("F-statistic:" + str(f_statistic))
y = ("P-value:" + str(p_value))

txt = open("ANOVA_result.txt",'w+')

txt.write(x)

txt.write(y)