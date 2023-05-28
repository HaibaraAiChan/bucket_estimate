import numpy as np

def backpack_split(weights, values, capacity):
    n = len(weights)
    
    # Create a table to store the maximum values for each pack and weight combination
    table = [[0] * (capacity + 1) for _ in range(n + 1)]
    # table [i-1] means contains the from 1 to i-1 items.
    for i in range(1, n + 1):
        for j in range(1, capacity + 1):
            if weights[i - 1] <= j: # weigths idx range[0, n-1], i range [1,n]'i-1' here means the range[0,n-1]
                # If the current weight can fit in the pack
                # Take the maximum of either including the item or excluding it
                table[i][j] = max(values[i - 1] + table[i - 1][j - weights[i - 1]], table[i - 1][j])
            else:
                # If the current weight cannot fit in the pack, exclude the item
                table[i][j] = table[i - 1][j]

    # Determine the items included in each pack
    packs = []
    i, j = n, capacity
    while i > 0 and j > 0:
        if table[i][j] != table[i - 1][j]:
            pack = []
            while i > 0 and j > 0 and table[i][j] != 0:
                if table[i][j] != table[i - 1][j]:
                    pack.append(i - 1)
                    j -= weights[i - 1]
                i -= 1
            packs.append(pack[::-1])
        i -= 1

    # Calculate the maximum value for each pack
    max_values = [table[n][capacity] for _ in range(len(packs))]

    return max_values, packs

def remove_items_by_indices(lst, indices):
    # print('indices ',indices)
    rst = [item for index, item in enumerate(lst) if index not in indices]
    # print('rst ', rst)
    return rst




def split_all(weights, values, capacity):
    
    weights.sort(reverse=True)
    print('weights', weights)
    values = weights
    GROUPS =[]

    while len(weights)>1:
        if sum(weights)<= capacity:
            # print('the last batch total value is ', sum(weights))
            print("Maximum values:", sum(weights))
            # print('last batch items values ', weights)
            GROUPS.append(weights)
            break
        else:    # sum(weights)> capacity
            max_values, packs = backpack_split(weights, values, capacity)
            
            res_tmp = np.array(weights)[packs[0]]
            GROUPS.append(list(res_tmp))
            print("Maximum values:", max_values[0])
            # print("Packs:")
            for pack in packs:
                print("remove weights:", res_tmp)
            print()
            print('before remove weights, ',weights)
            weights = remove_items_by_indices(weights, packs[0])
            print('after remove pre pack weights, ', weights)
            values = weights
                
    if len(weights)==1 :
        if sum(weights)<= capacity:
            print('the last batch value is ', weights[0])
            GROUPS.append([weights[0]])
        else:
            print('error, OOM!')
            
    return GROUPS

def main():

    weights = [2, 3, 4, 5, 2, 1]
    values = [2, 3, 4, 5, 2, 1]
    capacity = 9
    
    GROUPS = split_all(weights, values, capacity)
    print()
    print('GROUPS ', GROUPS)

    
if __name__=='__main__':
	main()