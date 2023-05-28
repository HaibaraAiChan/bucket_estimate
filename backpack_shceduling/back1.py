
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