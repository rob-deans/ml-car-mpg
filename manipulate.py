import pandas as pd

# Read the csv
df = pd.read_csv('auto-mpg.txt', delim_whitespace=True, header=None)

columns = ['mpg', 'cylinders', 'displacement', 'horsepower',
           'weight', 'acceleration', 'model year', 'origin', 'car name']
df.columns = columns


def normalise(val):
    f_val = float(val)
    return 1/f_val


for col in columns:
    if col != 'car name':
        df[col] = df[col].apply(normalise)

# Drop the irrelevant column
df = df.drop('car name', axis=1)

# Save the dataframe to a file
df.to_csv('norm_auto_mpg.csv', sep=',', index=False)
