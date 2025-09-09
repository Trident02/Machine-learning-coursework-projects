## 1 ANS :
# # Define a function that checks if there are two numbers in a sequence that add up to a target sum
# def has_pair_with_sum(sequence, target_sum):
#     seen_numbers = set()  # Initialize an empty set to store numbers we have seen so far
#     for number in sequence:  # Iterate through each number in the sequence
#         if target_sum - number in seen_numbers:  # Check if the complement of the current number exists in the set
#             return True  # If so, we have found a pair that adds up to the target sum
#         seen_numbers.add(number)  # Otherwise, add the current number to the set
#     return False  # If we reach the end of the sequence without finding a pair, return False


# sequence = [int(x) for x in input("Enter a sequence of numbers separated by spaces: ").split()]
# target_sum = int(input("Enter the target sum: "))

# if has_pair_with_sum(sequence, target_sum):
#     print("There exists two elements in the sequence whose sum is exactly %d" %target_sum)
# else:
#     print("There are no two elements in the sequence whose sum is exactly %d" %target_sum)



## 2 ANS :

# Recursion
# def subSets_recursive(string, index, current):
#     if index == len(string):
#         print(current)
#         return
#     subSets_recursive(string, index + 1, current + string[index])
#     subSets_recursive(string, index + 1, current)

# # Iterative version
# def subSets_iterative(string):
#     result = [""]
#     for char in string:
#         # Store the length of result before we start adding new subsets
#         n = len(result)
#         for i in range(n):
#             # For each existing subset, add a new subset by concatenating the current character
#             subset = result[i] + char
#             result.append(subset)
#     print("\n".join(result))

# s1 = input("Enter the String : ")
# index = 0
# current = ""
# print("Subsets of the string using recursion:")
# subSets_recursive(s1, index, current)

# print("Subsets of the string using iteration:")
# subSets_iterative(s1)



## 3 ANS : 
# # Define a function to calculate the distance from the starting point after a sequence of movements
# def distance_from_start(movements):
#     x, y = 0, 0  # Initialize coordinates of the starting point
#     for direction, steps in movements:  # Iterate through each movement
#         if direction == 'N':  # Update coordinates based on the direction of movement
#             y += steps
#         elif direction == 'S':
#             y -= steps
#         elif direction == 'E':
#             x += steps
#         elif direction == 'W':
#             x -= steps
#     # Calculate the Euclidean distance from the starting point and return the result
#     return ((x ** 2) + (y ** 2)) ** 0.5

# # Initialize an empty list to store the movements
# movements = []
# while True:
#     # Take input for the direction and steps of each movement
#     direction = input("Enter direction (N/S/E/W) or 'done' to finish: ")
#     if direction == 'done':  # Break the loop if the user enters 'done'
#         break
#     steps = int(input("Enter steps: "))  # Convert steps to an integer
#     movements.append((direction, steps))  # Add the movement to the list

# # Call the function to calculate the distance from the starting point
# distance = distance_from_start(movements)
# # Print the result, rounded to two decimal places
# print("Distance from starting point: %.2f "%distance)



## 4 - ANS : 
# def binary_search(arr, x):
#     low = 0
#     high = len(arr) - 1
#     mid = 0
 
#     while low <= high:
#         mid = (high + low) // 2
 
#         # If x is greater, ignore the left half
#         if arr[mid] < x:
#             low = mid + 1
 
#         # If x is smaller, ignore the right half
#         elif arr[mid] > x:
#             high = mid - 1
 
#         # x is present at mid
#         else:
#             return mid
 
#     # If we reach here, the element was not present
#     return -1

# arr = list(map(int, input("Enter a sorted list of integers separated by spaces: ").split()))
# x = int(input("Enter the element to be searched: "))
# result = binary_search(arr, x)

# if result != -1:
#     print("Element is present at index %d" %result)
# else:
#     print("Element is not present in the list")


## 5 ANS : 
# Define a recursive function to find the length of the Longest Common Subsequence (LCS) of two strings
# def lcs(s1, s2, m, n):
#     if m == 0 or n == 0:  # Base case: if either string is empty, LCS length is 0
#         return 0
#     elif s1[m-1] == s2[n-1]:  # If last characters of both strings match, add 1 to LCS of remaining strings
#         return 1 + lcs(s1, s2, m-1, n-1)
#     else:  # If last characters do not match, take maximum of LCS without last character of first string or second string
#         return max(lcs(s1, s2, m, n-1), lcs(s1, s2, m-1, n))

# # Input two strings from the user
# s1 = input("Enter the first string: ")
# s2 = input("Enter the second string: ")

# # Call the lcs function and print the length of the LCS
# print ("Length of Longest Common Subsequence (LCS) is %d" %lcs(s1, s2, len(s1), len(s2)))


## 6 ANS : 
# def take_matrix_input():
#     # Initialize an empty list to store the matrix
#     matrix = []
#     while True:
#         # Prompt the user to enter a row of numbers or 'done' to finish
#         row = input("Enter a row of the matrix (or 'done' to finish): ")
#         # Check if the input is 'done'
#         if row.lower() == 'done':
#             # Break out of the loop if 'done' is entered
#             break
#         else:
#             # Split the entered row into individual strings, convert each string to an integer,
#             # convert the map object to a list, and append the list to the matrix
#             matrix.append(list(map(int, row.split())))
#     # Return the complete matrix
#     return matrix


# def print_spiral_form(matrix):
#     result = []   # Empty list for string the matrix elements in spiral order
#     while matrix:
#         result.extend(matrix.pop(0))  # Add the first row and remove it from the matrix
#         if matrix and matrix[0]:      # If there are remaining rows and columns
#             for row in matrix:
#                 result.append(row.pop())  # Add the last element of each remaining row
#             if matrix and matrix[-1]:     # If there is a remaining last row
#                 result.extend(matrix.pop()[::-1])  # Add it in reverse order
#             for row in matrix[::-1]:        # For each remaining row in reverse order
#                 if row:                     # If there are remaining elements in the row
#                     result.append(row.pop(0))  # Add the first element
#     print(result)



# matrix = take_matrix_input()
# print("Spiral order of the matrix:")
# print_spiral_form(matrix)



