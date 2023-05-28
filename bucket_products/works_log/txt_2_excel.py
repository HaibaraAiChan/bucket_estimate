import pandas as pd

# Read the TXT file
with open('degree_nids.txt', 'r') as file:
    lines = file.readlines()
numbers = [int(line.strip()) for line in lines]
# Create a Pandas DataFrame
df = pd.DataFrame({'Num': numbers})

# Export to Excel
df.to_excel('output_file.xlsx', index=False)
