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

def sort_with_original_index(lst):
    indexed_dict = {index: value for index, value in enumerate(lst)}
    sorted_dict = dict(sorted(indexed_dict.items(), key=lambda x: -x[1]))
    sorted_indices = list(sorted_dict.keys())
    sorted_values = list(sorted_dict.values())
    return sorted_indices, sorted_values, indexed_dict, sorted_dict

def get_index_by_value(my_dict, value):
    reverse_dict = {v: k for k, v in my_dict.items()}
    print('reverse_dict ', reverse_dict)
    print('value ', value)
    index = []
    for v in value:
        tmp = reverse_dict.get(v)
        if tmp not in index:
            index.append(tmp )
        else:
            indices = [ind for ind, val in reverse_dict.items() if val == v]
            for i in indices:
                if i not in index:
                    index.append(tmp )
                    break
    return index



def split_all(weights, values, capacity):
    
    # weights.sort(reverse=True)
    sorted_indices, sorted_values, my_dict, sorted_dict = sort_with_original_index(weights)
    print('my_dict ', sorted_dict)
    print()
    print('weights after sort', sorted_values)
    weights = sorted_values
    values = sorted_values
    GROUPS_weight =[]
    GROUPS_bucket_idx =[]
    while len(weights)>1:
        if sum(weights)<= capacity:
            # print('the last batch total value is ', sum(weights))
            # print("Maximum values:", sum(weights))
            # print('last batch items values ', weights)
            original_index = get_index_by_value(sorted_dict, weights)
            GROUPS_weight.append(weights)
            GROUPS_bucket_idx.append(original_index)
            break
        else:    # sum(weights)> capacity
            max_values, packs = backpack_split(weights, values, capacity)
            
            res_tmp = np.array(weights)[packs[0]]
            GROUPS_weight.append(list(res_tmp))
            
            print("Maximum values:", max_values[0])
            original_index = get_index_by_value(sorted_dict, res_tmp)
            GROUPS_bucket_idx.append(original_index)
            print()
            print("remove bucket_id: ",packs[0])
            print('original bucket_id :, ', original_index)
            print("remove weights:  ", res_tmp)
            print()
            print('before remove weights, ',weights)
            weights = remove_items_by_indices(weights, packs[0])
            print('after remove pre pack weights, ', weights)
            values = weights
                
    if len(weights)==1 :
        if sum(weights)<= capacity:
            print('the last batch value is ', weights[0])
            GROUPS_weight.append([weights[0]])
        else:
            print('error, OOM!')
            
    return GROUPS_weight, GROUPS_bucket_idx

def main():
    # weights = [2, 3, 4, 5, 2, 1]
    # values = [2, 3, 4, 5, 2, 1]
    adjust = 1000
    weights = [0.004248619079589844, 0.010951995849609375, 0.024433135986328125, 0.040657997131347656, 0.054093360900878906, 0.0782003402709961, 0.10260200500488281, 0.1095743179321289, 0.1300048828125, 0.1599140167236328, 0.17887401580810547, 0.17557525634765625, 0.21084308624267578, 0.23604297637939453, 0.24538421630859375, 0.2514839172363281, 0.26112842559814453,0.27727317810058594,0.29793834686279297,0.29139232635498047,0.3255643844604492,0.3578939437866211,0.38632965087890625,0.3936424255371094]
    weights = [int(item * adjust) for item in weights]
    # print('weights after preprocess ')
    # print(weights)
    print()
    values =  weights 
    capacity = 2 * adjust
    
    GROUPS_weight, GROUPS_bucket_idx = split_all(weights, values, capacity)
    print()
    print('GROUPS_weight ')
    for itm in GROUPS_weight:
        tmp = [it/adjust for it in itm]
        print(str(tmp)+ ', sum memory of current group: '+ str(sum(tmp)))
    print()
    print('bucket id ')
    for itm in GROUPS_bucket_idx:
        print(itm)
        
if __name__=='__main__':
	main()


