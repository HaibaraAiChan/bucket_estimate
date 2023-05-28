# Open the file for reading

with open("input.txt", "r") as file:
    # Read the content of the file and split it into a list of lines
    lines = file.readlines()

# Remove newline characters from the lines

lines = [float(line.strip()) for line in lines]
mid = int(len(lines)/2)
input_ =  lines[:mid]
output =  lines[mid:]
# print(input_)
# print(output)
# for i in range(mid):
#     print(input_[i]/output[i])
# Print the list
# print(lines)
# print(len(lines))
# print(sum(lines))

# for i in range(mid):
#     print(input_[i]/output[i]/25/0.411)
res =0
res_list = []
for i in range(mid):
    # rr = 15.05/(input_[i]/output[i]/25/0.411)
    # rr = 15.05/(input_[i]/output[i])
    rr = 15.05/(input_[i]/output[i]/25)
    print(rr)
    res += rr 
    res_list.append(rr)
print()
print(res-res_list[0])