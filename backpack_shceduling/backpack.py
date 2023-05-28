
def multiple_backpack(weights, values, capacities):
    n = len(weights)
    m = len(capacities)

    # Create a table to store the maximum values and included items for each backpack and weight combination
    table = [[(0, []) for _ in range(max(capacities) + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, max(capacities) + 1):
            if weights[i - 1] <= j:
                # If the current weight can fit in the backpack
                # Take the maximum of either including the item or excluding it
                included_value = values[i - 1] + table[i - 1][j - weights[i - 1]][0]
                excluded_value = table[i - 1][j][0]

                if included_value > excluded_value:
                    included_items = table[i - 1][j - weights[i - 1]][1] + [i]
                    table[i][j] = (included_value, included_items)
                else:
                    table[i][j] = (excluded_value, table[i - 1][j][1])
            else:
                # If the current weight cannot fit in the backpack, exclude the item
                table[i][j] = table[i - 1][j]

    results = []
    for backpack in capacities:
        max_value, included_items = table[n][backpack]
        items = [(weights[i - 1], values[i - 1]) for i in included_items]
        results.append((max_value, items))

    return results


# Example usage
weights = [1, 3, 4, 2]
# values = [4, 5, 6, 3]
values = [1, 3, 4, 2]
capacities = [5, 7, 8]

max_values_with_items = multiple_backpack(weights, values, capacities)
for max_value, items in max_values_with_items:
    print("Max Value:", max_value)
    print("Items included:")
    for weight, value in items:
        print("Weight:", weight, "Value:", value)
    print()