## AA_indegree_prodcuts.py 
collect the information of 
1. degree of dst distribution of full batch
2. fanout degree partition 
    (a) number of input nids of each bucket batch
    (b) redundancy ratio of each bucket batch

## dict_to_excel.py
copy your degree of dst distribution of full batch to "data"
then you can get and excel table, which can copy to online excel to draw bar chart.


## cd redundant_ratio/
#### python gen_redundant_list.py > res_.txt 
the redundant ratio list is saved in res_.txt

#### 0.411 ogbn-products cluster coeff
python gen_redundant_list.py  > res_coeff.txt 
the redundant ratio list is saved in res_coeff.txt

## AA_mem_estimate_products.py
1. estimate the memory consumption in each bucket batch. There are two blocks in each bucket batch.
#### ogbn-products 
input feature size 100
hidden size 128 
redundant_ratio_list is in "redundant_ratio/res.txt"
"def estimate(blocks_info, in_feat, hidden_size, redundant_ratio_list ):"

#### works_log
split_cal.py
calculate ratio and estimated memory in slides