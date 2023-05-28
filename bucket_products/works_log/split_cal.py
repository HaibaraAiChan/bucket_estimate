# Open the file for reading

# with open("input_seeds.txt", "r") as file:
#     lines = file.readlines()

# lines = [float(line.strip()) for line in lines]
# mid = int(len(lines)/2)
# input_ =  lines[:mid]
# output =  lines[mid:]
# # print(mid)
# res =0
# res_list = []
# print('input/output')
# for i in range(mid):
#     # print(i )
#     rr = (input_[i]/output[i]/50/0.411*2.12)
#     error = 100*(16.80-rr)/16.80
#     # print(rr)
#     print(error)
#     # res += rr 
#     # res_list.append(rr)
# # print()
# # print(res-res_list[0])


with open("input_seeds-38.txt", "r") as file:
    lines = file.readlines()

lines = [float(line.strip()) for line in lines]
mid = int(len(lines)/2)
input_ =  lines[:mid]
output =  lines[mid:]
print('mid')
print(mid)
res =0
error_list = []
print('input/output/100/0.411')
for i in range(mid):
    
    rr = (input_[i]/output[i]/100/0.411*1.965)
    error = 100*(16-rr)/16
    print(str(i+1)+ ' \t'+str(rr) + '\t' + str(error)+'\t %')
    error_list.append(error)
print()
print(sum(error_list[11:])/len(error_list[11:]))


# with open("input_seeds-34.txt", "r") as file:
#     lines = file.readlines()

# lines = [float(line.strip()) for line in lines]
# mid = int(len(lines)/2)
# input_ =  lines[:mid]
# output =  lines[mid:]
# print('mid')
# print(mid)
# res =0
# error_list = []
# print('input/output/100/0.411')
# for i in range(mid):
    
#     rr = (input_[i]/output[i]/100/0.411*2.19625)
#     # print(str(i+1)+ ' \t'+str(rr))
#     error = 100*(17.5-rr)/17.5
#     print(str(i+1)+ ' \t'+str(rr) + '\t' + str(error)+'\t %')
#     # print(error)
#     # res += rr 
#     error_list.append(error)
# print()
# print(sum(error_list[11:])/len(error_list[11:]))


# with open("input_seeds-40.txt", "r") as file:
#     lines = file.readlines()

# lines = [float(line.strip()) for line in lines]
# mid = int(len(lines)/2)
# input_ =  lines[:mid]
# output =  lines[mid:]
# print('mid')
# print(mid)
# res =0
# error_list = []
# print('input/output/100/0.411')
# for i in range(mid):
    
#     rr = (input_[i]/output[i]/100/0.411*1.867)
#     # print(str(i+1)+ ' \t'+str(rr))
#     error = 100*(15.3-rr)/15.3
#     print(str(i+1)+ ' \t'+str(rr) + '\t' + str(error)+'\t %')
#     # print(error)
#     # res += rr 
#     error_list.append(error)
# print()
# print(sum(error_list[11:])/len(error_list[11:]))

