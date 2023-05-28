# Open the file for reading
# with open("1-24.txt", "r") as file:
# with open("redundant_ratio_list.txt", "r") as file:
# with open("redundant_ratio_list_coeff.txt", "r") as file:
with open("redundant_file.txt", "r") as file:
    # Read the content of the file and split it into a list of lines
    lines = file.readlines()

# Remove newline characters from the lines
# lines = [float(line.strip())*0.411 for line in lines]
lines = [float(line.strip()) for line in lines]

# Print the list
print(lines)
print(len(lines))
# print(sum(lines))

