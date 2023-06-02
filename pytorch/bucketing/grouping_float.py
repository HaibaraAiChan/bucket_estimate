import numpy as np
import itertools

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

def get_index_by_value(dictionary, values_list):
    n = len(values_list)
    values_list = list(values_list)
    keys = []
    # print('values_list ', values_list)
    # print('keys ------')
    
    for va in values_list:
        m = values_list.count(va)
        idx = [key for key, value in dictionary.items() if value == va]
        if m == len(idx):
            if idx[0] not in keys:
                keys += idx
        elif m < len(idx):
            if idx[0] not in keys:
                keys += idx[:m]
        
        # print(keys)
    # print('keys ------end')
    return keys



def split_all(weights, values, capacity):
    
    # weights.sort(reverse=True)
    sorted_indices, sorted_values, my_dict, sorted_dict = sort_with_original_index(weights)
    print('sorted_dict ', sorted_dict)
    print()
    print('weights after sort', sorted_values)
    weights = sorted_values
    values = sorted_values
    GROUPS_weight =[]
    GROUPS_bucket_idx =[]
    while len(weights)>=1:
        if sum(weights)<= capacity:

            original_index = get_index_by_value(sorted_dict, weights)
            GROUPS_weight.append(weights)
            GROUPS_bucket_idx.append(original_index)
            break
        else:    # sum(weights)> capacity
            max_values, packs = backpack_split(weights, values, capacity)
            
            res_tmp = np.array(weights)[packs[0]]
            GROUPS_weight.append(list(res_tmp))
            

            original_index = get_index_by_value(sorted_dict, res_tmp)
            GROUPS_bucket_idx.append(original_index)
            print()
            print("remove bucket_id: ",packs[0])
            print('original bucket_id :, ', original_index)
            print("remove weights:  "+ str(res_tmp) + ", \t\t------------sum "+ str(sum(res_tmp)))
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


def grouping_fanout_1(adjust, weights, capacity):

	if len(weights) == 49:
		weights = [0.031600323026579925, 0.053446445057834434, 0.04691033726707499, 0.07212925883696267, 0.0954132446010461, 0.13250813817436047, 0.16562827234049787, 0.18126462923828512, 0.21130672298992675, 0.25300076929852366, 0.2809490893635299, 0.28129312471449885, 0.33190986587898375, 0.36230173630435075, 0.3834405979819673, 0.38852240658495635, 0.4104866247767621, 0.427057239492208, 0.45594087203866557, 0.4482479429953582, 0.494359802184077, 0.5455698065359045, 0.5838345744003708, 0.5952225418284881, 0.6416539241286929, 0.6823511784373357, 0.666389745486164, 0.7496792492248849, 0.7371837931190246, 0.7577242599083827, 0.7889046908693763, 0.8683255342292655, 0.9311795745279405, 0.8477295250909833, 0.9436967117287708, 0.9945587138174034, 1.0309573992937635, 1.0749793136129961, 1.0747561831684673, 1.1274098691910925,1.2304586825034851, 1.1488268197006972, 1.3300050600793791, 1.2305013597063668, 1.339544299635952, 1.363191539881995, 1.501307503974184, 1.4590092047286807, 1.473764838436366]
	if len(weights) == 24:
		weights = [0.004248619079589844, 0.010951995849609375, 0.024433135986328125, 0.040657997131347656, 0.054093360900878906, 0.0782003402709961, 0.10260200500488281, 0.1095743179321289, 0.1300048828125, 0.1599140167236328, 0.17887401580810547, 0.17557525634765625, 0.21084308624267578, 0.23604297637939453, 0.24538421630859375, 0.2514839172363281, 0.26112842559814453,0.27727317810058594,0.29793834686279297,0.29139232635498047,0.3255643844604492,0.3578939437866211,0.38632965087890625,0.3936424255371094]
	weights = [int(item * adjust) for item in weights]

	values =  weights 
	capacity = int(capacity * adjust)
	print('capacity ', capacity)


	GROUPS_weight, GROUPS_bucket_idx = split_all(weights, values, capacity)

	# for itm in GROUPS_weight:
	# 	tmp = [it/adjust for it in itm]
	# print()
	# print('bucket ids in each group ')
	# for itm in GROUPS_bucket_idx:
	# 	print(itm)
	# print()
	# total = [ len(GROUPS_bucket_idx[i]) for i in range(len(GROUPS_bucket_idx)) ]
	# print('each group length ')
	# print(total)
		
	# print(sum(total))
	return GROUPS_weight, GROUPS_bucket_idx





# def main():

#     # weights = [2, 3, 4, 5, 2, 1]
#     # values = [2, 3, 4, 5, 2, 1]
#     adjust = 1000
#     weights = [0.004248619079589844, 0.010951995849609375, 0.024433135986328125, 0.040657997131347656, 0.054093360900878906, 0.0782003402709961, 0.10260200500488281, 0.1095743179321289, 0.1300048828125, 0.1599140167236328, 0.17887401580810547, 0.17557525634765625, 0.21084308624267578, 0.23604297637939453, 0.24538421630859375, 0.2514839172363281, 0.26112842559814453,0.27727317810058594,0.29793834686279297,0.29139232635498047,0.3255643844604492,0.3578939437866211,0.38632965087890625,0.3936424255371094]
	
#     # weights = [0.031600323026579925, 0.053446445057834434, 0.04691033726707499, 0.07212925883696267, 0.0954132446010461, 0.13250813817436047, 0.16562827234049787, 0.18126462923828512, 0.21130672298992675, 0.25300076929852366, 0.2809490893635299, 0.28129312471449885, 0.33190986587898375, 0.36230173630435075, 0.3834405979819673, 0.38852240658495635, 0.4104866247767621, 0.427057239492208, 0.45594087203866557, 0.4482479429953582, 0.494359802184077, 0.5455698065359045, 0.5838345744003708, 0.5952225418284881, 0.6416539241286929, 0.6823511784373357, 0.666389745486164, 0.7496792492248849, 0.7371837931190246, 0.7577242599083827, 0.7889046908693763, 0.8683255342292655, 0.9311795745279405, 0.8477295250909833, 0.9436967117287708, 0.9945587138174034, 1.0309573992937635, 1.0749793136129961, 1.0747561831684673, 1.1274098691910925,1.2304586825034851, 1.1488268197006972, 1.3300050600793791, 1.2305013597063668, 1.339544299635952, 1.363191539881995, 1.501307503974184, 1.4590092047286807, 1.473764838436366]
#     weights = [int(item * adjust) for item in weights]

#     print()
#     values =  weights 
#     capacity = 7 * adjust
#     # capacity = 9 * adjust
    
#     GROUPS_weight, GROUPS_bucket_idx = split_all(weights, values, capacity)
    
#     for itm in GROUPS_weight:
#         tmp = [it/adjust for it in itm]
#     print()
#     print('bucket ids in each group ')
#     for itm in GROUPS_bucket_idx:
#         print(itm)
#     print()
#     total = [ len(GROUPS_bucket_idx[i]) for i in range(len(GROUPS_bucket_idx)) ]
#     print('each group length ')
#     print(total)
    
#     print(sum(total))
    
if __name__=='__main__':
	# main()
	adjust = 1000
	weights = [0.031600323026579925, 0.053446445057834434, 0.04691033726707499, 0.07212925883696267, 0.0954132446010461, 0.13250813817436047, 0.16562827234049787, 0.18126462923828512, 0.21130672298992675, 0.25300076929852366, 0.2809490893635299, 0.28129312471449885, 0.33190986587898375, 0.36230173630435075, 0.3834405979819673, 0.38852240658495635, 0.4104866247767621, 0.427057239492208, 0.45594087203866557, 0.4482479429953582, 0.494359802184077, 0.5455698065359045, 0.5838345744003708, 0.5952225418284881, 0.6416539241286929, 0.6823511784373357, 0.666389745486164, 0.7496792492248849, 0.7371837931190246, 0.7577242599083827, 0.7889046908693763, 0.8683255342292655, 0.9311795745279405, 0.8477295250909833, 0.9436967117287708, 0.9945587138174034, 1.0309573992937635, 1.0749793136129961, 1.0747561831684673, 1.1274098691910925,1.2304586825034851, 1.1488268197006972, 1.3300050600793791, 1.2305013597063668, 1.339544299635952, 1.363191539881995, 1.501307503974184, 1.4590092047286807, 1.473764838436366]
	# weights = [0.004248619079589844, 0.010951995849609375, 0.024433135986328125, 0.040657997131347656, 0.054093360900878906, 0.0782003402709961, 0.10260200500488281, 0.1095743179321289, 0.1300048828125, 0.1599140167236328, 0.17887401580810547, 0.17557525634765625, 0.21084308624267578, 0.23604297637939453, 0.24538421630859375, 0.2514839172363281, 0.26112842559814453,0.27727317810058594,0.29793834686279297,0.29139232635498047,0.3255643844604492,0.3578939437866211,0.38632965087890625,0.3936424255371094]
	
	print(len(weights))
	capacity = 1.7
	G_WEIGHTS, G_BUCKET_ID = grouping_fanout_1(adjust, weights, capacity)
	if G_WEIGHTS:
		for itm in G_WEIGHTS:
				tmp = [it/adjust for it in itm]
		print()
		print('bucket ids in each group ')
		for itm in G_BUCKET_ID:
			print(itm)
		print()
		total = [ len(G_BUCKET_ID[i]) for i in range(len(G_BUCKET_ID)) ]
		print('each group length ')
		print(total)
            
		print(sum(total))
		print(len(total))
	# oo =range(24)
	


	# flattened_list = list(itertools.chain(*G_BUCKET_ID))
	# print(flattened_list)

	# print(set(oo)-set(flattened_list))