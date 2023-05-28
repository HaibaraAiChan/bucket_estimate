import numpy

data_dict=[[{1: 181, 2: 3, 3: 6, 4: 3, 5: 5, 6: 2, 7: 3, 9: 3, 10: 101}, {1: 173}], [{1: 11, 2: 248, 3: 13, 4: 12, 5: 11, 6: 9, 7: 4, 8: 4, 9: 7, 10: 256}, {2: 221}], [{1: 12, 2: 20, 3: 408, 4: 19, 5: 22, 6: 17, 7: 13, 8: 5, 9: 7, 10: 566}, {3: 374}], [{1: 7, 2: 17, 3: 30, 4: 499, 5: 39, 6: 33, 7: 32, 8: 17, 9: 13, 10: 939}, {4: 456}], [{1: 9, 2: 27, 3: 28, 4: 52, 5: 491, 6: 37, 7: 29, 8: 32, 9: 19, 10: 1302}, {5: 445}], [{1: 9, 2: 26, 3: 35, 4: 32, 5: 61, 6: 616, 7: 44, 8: 28, 9: 34, 10: 1890}, {6: 550}], [{1: 7, 2: 28, 3: 41, 4: 51, 5: 53, 6: 55, 7: 717, 8: 47, 9: 53, 10: 2442}, {7: 655}], [{1: 12, 2: 18, 3: 20, 4: 45, 5: 68, 6: 50, 7: 53, 8: 580, 9: 46, 10: 2785}, {8: 512}], [{1: 9, 2: 16, 3: 28, 4: 37, 5: 43, 6: 50, 7: 54, 8: 56, 9: 628, 10: 3299}, {9: 576}], [{1: 6, 2: 22, 3: 52, 4: 36, 5: 57, 6: 57, 7: 84, 8: 69, 9: 53, 10: 4690}, {10: 640}], [{1: 18, 2: 23, 3: 39, 4: 46, 5: 59, 6: 70, 7: 76, 8: 89, 9: 97, 10: 5246}, {11: 624}], [{1: 21, 2: 35, 3: 38, 4: 49, 5: 59, 6: 61, 7: 74, 8: 83, 9: 72, 10: 5148}, {12: 583}], [{1: 12, 2: 12, 3: 35, 4: 51, 5: 63, 6: 59, 7: 93, 8: 75, 9: 95, 10: 6258}, {13: 615}], [{1: 12, 2: 29, 3: 39, 4: 43, 5: 66, 6: 68, 7: 61, 8: 93, 9: 85, 10: 6974}, {14: 695}], [{1: 13, 2: 22, 3: 28, 4: 46, 5: 87, 6: 84, 7: 83, 8: 88, 9: 96, 10: 7226}, {15: 673}], [{1: 12, 2: 20, 3: 46, 4: 56, 5: 69, 6: 83, 7: 107, 8: 87, 9: 106, 10: 7442}, {16: 614}], [{1: 17, 2: 29, 3: 39, 4: 63, 5: 60, 6: 79, 7: 85, 8: 108, 9: 88, 10: 7835}, {17: 556}], [{1: 11, 2: 33, 3: 43, 4: 65, 5: 73, 6: 72, 7: 79, 8: 104, 9: 96, 10: 8253}, {18: 604}], [{1: 15, 2: 31, 3: 33, 4: 63, 5: 87, 6: 94, 7: 99, 8: 120, 9: 109, 10: 8898}, {19: 584}], [{1: 14, 2: 43, 3: 50, 4: 60, 5: 83, 6: 75, 7: 92, 8: 73, 9: 104, 10: 8689}, {20: 572}], [{1: 13, 2: 31, 3: 40, 4: 51, 5: 83, 6: 78, 7: 100, 8: 107, 9: 111, 10: 9765}, {21: 586}], [{1: 17, 2: 31, 3: 40, 4: 61, 5: 76, 6: 99, 7: 105, 8: 104, 9: 125, 10: 10796}, {22: 592}], [{1: 21, 2: 25, 3: 66, 4: 58, 5: 86, 6: 84, 7: 83, 8: 113, 9: 102, 10: 11629}, {23: 648}], [{1: 15, 2: 39, 3: 54, 4: 66, 5: 94, 6: 116, 7: 110, 8: 133, 9: 111, 10: 11759}, {24: 645}], [{1: 2709, 2: 4376, 3: 5796, 4: 7894, 5: 9452, 6: 10556, 7: 11720, 8: 12212, 9: 12614, 10: 1103303}, {25: 183378}]]
in_feat=100
hidden_size=128
SUM_mem =0
estimated_mem_list = []
modified_estimated_mem_list = []
weight =1
redundant_ratio = [7.976878612716763, 7.248868778280543, 5.85204991087344, 5.722039473684211, 6.154157303370786, 5.834545454545455, 5.287677208287896, 6.475830078125, 6.010030864197531, 5.838125, 6.01923076923077, 5.901943967981704, 6.146466541588493, 5.490133607399794, 5.587518573551263, 5.859324104234528, 6.388806601777402, 5.797001471670345, 6.045331651045422, 5.7749125874125875, 5.94677393141557, 6.1750614250614255, 5.81568706387547, 5.67984496124031, 0.395462923578619]

# redundant_ratio = [7.976878612716763, 7.248868778280543, 5.85204991087344, 5.722039473684211, 6.154157303370786, 5.834545454545455, 5.287677208287896, 6.475830078125, 6.010030864197531, 5.838125, 6.01923076923077, 5.901943967981704, 6.146466541588493, 5.490133607399794, 5.587518573551263, 5.859324104234528, 6.388806601777402, 5.797001471670345, 6.045331651045422, 5.7749125874125875, 5.94677393141557, 6.1750614250614255, 5.81568706387547, 5.67984496124031, 0.395462923578619]
# redundant_ratio = [7.976878612716763, 7.248868778280543, 5.85204991087344, 5.722039473684211, 6.154157303370786, 5.834545454545455, 5.287677208287896, 6.475830078125, 6.010030864197531, 5.838125, 6.01923076923077, 5.901943967981704, 6.146466541588493, 5.490133607399794, 5.587518573551263, 5.859324104234528, 6.388806601777402, 5.797001471670345, 6.045331651045422, 5.7749125874125875, 5.94677393141557, 6.1750614250614255, 5.81568706387547, 5.67984496124031, 0.395462923578619]
# redundant_ratio = [7.976878612716763, 7.248868778280543, 5.85204991087344, 5.722039473684211, 6.154157303370786, 5.834545454545455, 5.287677208287896, 6.475830078125, 6.010030864197531, 5.838125, 6.01923076923077, 5.901943967981704, 6.146466541588493, 5.490133607399794, 5.587518573551263, 5.859324104234528, 6.388806601777402, 5.797001471670345, 6.045331651045422, 5.7749125874125875, 5.94677393141557, 6.1750614250614255, 5.81568706387547, 5.67984496124031, 0.395462923578619]
				
for i, data in enumerate(data_dict):
    estimated_mem = 0
    for i in range (len(data)):
        sum_b = 0
        for idx, (key, val) in enumerate(data[i].items()):
            sum_b = sum_b + key*val
            if idx ==0: # the input layer, in_feat 100
                estimated_mem  +=  sum_b*in_feat*18*4/1024/1024/1024
            if idx ==1: # the output layer
                estimated_mem  +=  sum_b*hidden_size*18*4/1024/1024/1024
            

    estimated_mem_list.append([estimated_mem, weight])
    modified_estimated_mem_list.append([estimated_mem/redundant_ratio[i], weight]) # redundant_ratio[i] is a variable depends on graph characteristic
    SUM_mem += estimated_mem
    # print('  estimated memory /GB degree '+str(ii)+': '+str(estimated_mem) )  
    
print(SUM_mem)