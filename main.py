import matplotlib.pyplot as plt
import random

def get_user_input():
    choice = input("Do you want to provide your own data points or generate random ones? (type 'provide' or 'generate'): ").strip().lower()
    if choice == 'provide':
        n = int(input("How many data points do you want to provide? "))
        x = []
        y = []
        for i in range(n):
            xi = float(input(f"Enter x value for point {i+1}: "))
            yi = float(input(f"Enter y value for point {i+1}: "))
            x.append(xi)
            y.append(yi)
    elif choice == 'generate':
        n = int(input("How many random data points do you want to generate? "))
        random.seed(0)  # Seed for reproducibility
        x = [random.uniform(0, 100) for _ in range(n)]
        y = [2.5 * xi + random.normalvariate(0, 2) for xi in x]
    else:
        print("Invalid input. Please type 'provide' or 'generate'.")
        return get_user_input()
    return x, y


def dot_product(v1, v2):
    """ Calculate the dot product of two vectors. """
    return sum(x * y for x, y in zip(v1, v2))

def vector_subtract(v1, v2):
    """ Subtract vector v2 from v1. """
    return [x - y for x, y in zip(v1, v2)]

def vector_add(v1, v2):
    """ Add vector v2 to v1. """
    return [x + y for x, y in zip(v1, v2)]

def scalar_multiply(c, v):
    """ Multiply a vector v by a scalar c. """
    return [c * x for x in v]

def gram_schmidt(X):
    """ Orthogonalize a list of vectors. """
    Y = []
    for i in range(len(X)):
        gs = X[i]
        for j in range(i):
            gs = vector_subtract(gs, scalar_multiply(dot_product(Y[j], X[i]) / dot_product(Y[j], Y[j]), Y[j]))
        Y.append(gs)
    return Y

def project(V, y):
    """ Calculate the projection of y in the column space of X """
    Y = scalar_multiply(dot_product(y, V[0])/dot_product(V[0], V[0]), V[0])
    for i in range(len(V) - 1):
        proj = scalar_multiply(dot_product(y, V[i+1])/dot_product(V[i+1], V[i+1]), V[i+1])
        Y = vector_add(Y, proj)
    return Y

def fit_line(x, y):
    """ Fit a line using the Gram-Schmidt process. """
    n = len(x)
    X = [[1] * n, x]  # Design matrix with intercept and x values
    V = gram_schmidt(X)
    y = project(V, y)
    # Compute coefficients
    print(V)
    print(y)
    print([row + [y[i]] for i, row in enumerate([[V[row][col] for row in range(len(V))] for col in range(len(V[0]))])])
    b = gaussian_elimination([row + [y[i]] for i, row in enumerate([[V[row][col] for row in range(len(V))] for col in range(len(V[0]))])])
    return b

def gaussian_elimination(aug_matrix):
    # Step 1: Get the number of rows and columns
    num_rows = len(aug_matrix)
    num_cols = len(aug_matrix[0])
    
    # Step 2: Perform Gaussian elimination
    for i in range(min(num_rows, num_cols - 1)):  # Use min to handle matrices with more rows than variables
        # Partial pivoting
        max_row = max((abs(aug_matrix[j][i]), j) for j in range(i, num_rows))[1]
        if i != max_row:
            # Swap the current row with the row having the maximum element
            aug_matrix[i], aug_matrix[max_row] = aug_matrix[max_row], aug_matrix[i]

        # Eliminate entries below the current pivot
        for j in range(i + 1, num_rows):
            if aug_matrix[i][i] == 0:
                continue  # This column is zeroed, which shouldn't happen after pivoting
            factor = aug_matrix[j][i] / aug_matrix[i][i]
            for k in range(i, num_cols):
                aug_matrix[j][k] -= factor * aug_matrix[i][k]

    # Step 3: Back substitution
    solution = [0] * (num_cols - 1)
    for i in range(min(num_rows, num_cols - 1) - 1, -1, -1):
        sum_ax = 0
        for j in range(i + 1, num_cols - 1):
            sum_ax += aug_matrix[i][j] * solution[j]
        if aug_matrix[i][i] == 0:
            if abs(aug_matrix[i][-1] - sum_ax) < 1e-12:
                solution[i] = 0  # This might indicate multiple solutions
            else:
                return None  # No solution
        else:
            solution[i] = (aug_matrix[i][-1] - sum_ax) / aug_matrix[i][i]

    return solution



#Get vectors for the x and y values.
x, y = get_user_input()


# Fit the line using Gram-Schmidt
coefficients = fit_line(x, y)
intercept, slope = coefficients

# Calculate the fitted values
fitted_y = [intercept + slope * xi for xi in x]

# Plotting the data and the line of best fit
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, fitted_y, color='red', label='Line of Best Fit')
plt.title('Line of Best Fit Using Gram-Schmidt Orthogonalization')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
