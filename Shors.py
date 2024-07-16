import numpy as np
import matplotlib.pyplot as plt
import random
import math

def get_number_to_factorize():
    while True:
        try:
            N = int(input("Enter the number to factorize (greater than 1): "))
            if N > 1:
                return N
            else:
                print("Please enter a number greater than 1.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def find_order(a, N):
    r = 1
    while pow(a, r, N) != 1:
        r += 1
    return r

def shors_algorithm(N):
    if N % 2 == 0:
        return 2
    sqrt_N = int(math.sqrt(N))
    for i in range(2, sqrt_N + 1):
        if N % i == 0:
            return i

    while True:
        a = random.randint(2, N - 1)
        g = gcd(a, N)
        if g > 1:
            return g
        r = find_order(a, N)
        if r % 2 == 0:
            x = pow(a, r // 2, N)
            if x != N - 1:
                factor1 = gcd(x + 1, N)
                factor2 = gcd(x - 1, N)
                if factor1 > 1 and factor2 > 1:
                    return factor1, factor2

def main():
    N = get_number_to_factorize()
    factors = shors_algorithm(N)
    
    if isinstance(factors, tuple):
        print(f"Factors found: {factors}")
    else:
        print(f"Factor found: {factors}")

if __name__ == "__main__":
    main()