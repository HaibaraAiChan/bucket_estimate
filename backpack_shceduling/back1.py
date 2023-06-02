[0.031600323026579925, 
     0.053446445057834434, 
     0.04691033726707499, 
     0.07212925883696267, 
     0.0954132446010461, 
     0.13250813817436047, 
     0.16562827234049787, 
     0.18126462923828512, 
     0.21130672298992675, 
     0.25300076929852366, 
     0.2809490893635299, 
     0.28129312471449885, 
     0.33190986587898375, 
     0.36230173630435075, 
     0.3834405979819673, 
     0.38852240658495635, 
     0.4104866247767621, 
     0.427057239492208, 
     0.45594087203866557, 
     0.4482479429953582, 
     0.494359802184077, 
     0.5455698065359045, 
     0.5838345744003708, 
     0.5952225418284881, 
     0.6416539241286929, 
     0.6823511784373357, 
     0.666389745486164, 
     0.7496792492248849, 
     0.7371837931190246, 
     0.7577242599083827, 
     0.7889046908693763, 
     0.8683255342292655, 
     0.9311795745279405, 
     0.8477295250909833, 
     0.9436967117287708, 
     0.9945587138174034, 
     1.0309573992937635, 
     1.0749793136129961, 
     1.0747561831684673, 
     1.1274098691910925,
     1.2304586825034851, 
     1.1488268197006972, 
     1.3300050600793791, 
     1.2305013597063668, 
     1.339544299635952, 
     1.363191539881995, 
     1.501307503974184, 
     1.4590092047286807, 
     1.473764838436366]

def multiple_backpack(weights, values, capacities):
    n = len(weights)
    m = len(capacities)

    dp = [[[0, []] for _ in range(n + 1)] for _ in range(m + 1)]
    print(dp)
    
    for i in range(1, n + 1):
        print('i=', i)
        for j in range(1, m + 1):
            for k in range(1, max(capacities) + 1):
                print('i-1=', i-1)
                print('weights[i - 1]=', weights[i - 1])
                print('k=', k)
                
                if weights[i - 1] <= k:
                    print(weights[i - 1]-k)
                    print('values[i - 1]=', values[i - 1])
                    print('dp[j - 1][i - 1][k - weights[i - 1]][0]=', dp[j - 1][i - 1][k - weights[i - 1]][0])
                    print('dp[j][i - 1][k][0]=', dp[j][i - 1][k][0])
                    included_value = values[i - 1] + dp[j - 1][i - 1][k - weights[i - 1]][0]
                    excluded_value = dp[j][i - 1][k][0]

                    if included_value > excluded_value:
                        included_items = dp[j - 1][i - 1][k - weights[i - 1]][1] + [i]
                        dp[j][i][k] = [included_value, included_items]
                    else:
                        dp[j][i][k] = [excluded_value, dp[j][i - 1][k][1]]
                else:
                    dp[j][i][k] = dp[j][i - 1][k]

    results = []
    for j in range(1, m + 1):
        max_value, included_items = dp[j][n][capacities[j - 1]]
        items = [(weights[i - 1], values[i - 1]) for i in included_items]
        results.append((max_value, items))

    return results


# Example usage
weights = [2, 3, 4, 5, 1]
values = [2, 3, 4, 5, 1]
capacities = [6, 6, 6]

weights.sort(reverse=True)
print('weights', weights)
values = weights

max_values_with_items = multiple_backpack(weights, values, capacities)
for max_value, items in max_values_with_items:
    print("Max Value:", max_value)
    print("Items included:")
    for weight, value in items:
        print("Weight:", weight, "Value:", value)
    print()
    
    
    
    
