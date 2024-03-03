import pandas as pd
from matplotlib import pyplot as plt

df=pd.read_csv("C:/Users/paula/Documents/UK LAPTOP/17 GITHUB/workspace/timeseries/customers.csv")
product = df['product'].value_counts().sort_values(ascending=True)

# Use plot.bar() to create a bar chart
#product.plot.bar()
#plt.show()

plt.figure(figsize=(12,6))
plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.1)
ax= product.plot(kind='barh')

plt.xlabel('Number of clients')
plt.ylabel("Product Type")
plt.title("Customer classification")
plt.show()

from tabulate import tabulate
print(df.head())
# Print the first row using tabulate
print(tabulate(df.head(1), headers='keys', tablefmt='pretty'))

# Split the 'departments' column into a list of departments
df['stages'] = df['stages'].str.split(', ')

# Create a new column 'sequence' to represent the sequence of departments
df['sequence'] = df['stages'].apply(lambda x: ' -> '.join(x))

# Count the occurrences of each unique sequence
sequence_counts = df['sequence'].value_counts()

# Plot the sequences and their counts
plt.figure(figsize=(12,6))
bar_plot = sequence_counts.plot(kind='barh', color='skyblue')
plt.xlabel('Number of clients')
plt.ylabel("Shopping Phases")
plt.title("Customers Journey")
plt.show()

# Get the top 5 sequences and sort them in descending order
top_sequence = sequence_counts.head(5).sort_values(ascending=False)

# Plot the top sequences as horizontal bars with custom y-axis labels
plt.figure(figsize=(10, 6))
bars = plt.barh(range(1, 6), top_sequence.values, color='skyblue')

# Set custom y-axis labels
plt.yticks(range(1, 6), ['Top Sequence 1', 'Top Sequence 2', 'Top Sequence 3', 'Top Sequence 4', 'Top Sequence 5'])

# Create dummy lines for each legend entry
legend_handles = [plt.Line2D([0], [0], color='white', label=f'{i}. {sequence}') for i, sequence in enumerate(top_sequence.index, start=1)]

# Create a legend using the dummy lines, centered
plt.legend(handles=legend_handles, title='Top Sequences', loc='upper center', bbox_to_anchor=(0.6, 0.75))

plt.xlabel('Number of Customers')
plt.ylabel('Top Sequences')
plt.title('Top 5 Customer Journeys Through Departments')
plt.show()
