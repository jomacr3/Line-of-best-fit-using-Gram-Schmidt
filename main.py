import matplotlib.pyplot as plt

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
        import random
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

def scalar_multiply(c, v):
    """ Multiply a vector v by a scalar c. """
    return [c * x for x in v]

def gram_schmidt(X):
    """ Orthogonalize a list of vectors. """
    Y = []
    for i in range(len(X)):
        proj = X[i]
        for j in range(i):
            proj = vector_subtract(proj, scalar_multiply(dot_product(Y[j], X[i]) / dot_product(Y[j], Y[j]), Y[j]))
        Y.append(proj)
    return Y

def fit_line_gram_schmidt(x, y):
    """ Fit a line using the Gram-Schmidt process. """
    n = len(x)
    X = [[1] * n, x]  # Design matrix with intercept and x values
    Y = gram_schmidt(X)
    # Normalize Y
    Y = [scalar_multiply(1 / vector_norm(vec), vec) for vec in Y]
    # Compute coefficients
    b = [dot_product(vec, y) for vec in Y]
    return b


#Get vectors for the x and y values.
x, y = get_user_input()

# Fit the line using Gram-Schmidt
coefficients = fit_line_gram_schmidt(x, y)
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