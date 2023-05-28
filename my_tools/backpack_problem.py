def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[n][capacity]

def main():
    weights = [1, 3, 4, 5]
    values = [10, 40, 50, 70]
    capacity = 8
    max_value = knapsack(weights, values, capacity)
    print("The maximum value of items that can be packed in the knapsack is:", max_value)

if __name__ == "__main__":
    main()
