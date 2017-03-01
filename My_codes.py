# O(n^2), in-place
def bubble_sort(L):
    N = len(L)
    for i in range(N-1): # N-1 ops
        for j in range(1, N): # N-1 ops
            # swap consecutive elems(bubble up)
            if L[j]<=L[j-1]:
                L[j], L[j-1] = L[j-1], L[j]
    print(L)


# O(n^2), in-place
def selection_sort(L):
    N = len(L)
    for i in range(N-1): # 0 ... N-2 ==> N-1 ops
        max_val = L[0]
        pos = 0
        # find max
        for j in range(1, N-i): # 1 ... N-i-1 ==> N-i-1 ops
            if L[j]>=max_val:
                max_val = L[j]
                pos = j
        # move the max to the list end
        L[N-1-i], L[pos] = L[pos], L[N-1-i]
    print(L)


# O(n^2), in-place
def insertion_sort(L):
    N = len(L)
    for i in range(1, N):
        for j in range(N):
            if L[j] > L[i]:
                L[i], L[j] = L[j], L[i]
    print(L)

                
# O(nlogn), a lot of copying
def merge_sort(L):
    N = len(L)
    if N==1:
        return L
    else:
        left = merge_sort(L[:int(N/2)])
        right = merge_sort(L[int(N/2):])

    # merging
    L_sorted = []    
    while left and right:
        if left[0] < right[0]:
            L_sorted.append(left.pop(0))
        else:
            L_sorted.append(right.pop(0))

    # leftovers
    if left:
        L_sorted.extend(left)
    if right:
        L_sorted.extend(right)
        
    return L_sorted


# O(logn)
def max_heapify(A, i):
    N = len(A)
    if 2*i+1 < N:
        tmp = A[i]        
        if A[2*i+1] > A[i]:
            A[i] = A[2*i+1]
            A[2*i+1] = tmp
            max_heapify(A, 2*i+1)
    if 2*i+2 < N:
        tmp = A[i]        
        if A[2*i+2] > A[i]:
            A[i] = A[2*i+2]
            A[2*i+2] = tmp
            max_heapify(A, 2*i+2)            


# O(n)
def build_max_heap(A):
    N = len(A)
    for i in reversed(range(N//2)):
        max_heapify(A, i)
        

# O(nlogn)
def heap_sort(A):
    build_max_heap(A)
    while A:
        # after building the max is at 0
        mx = A[0]
        # swap the max with the last element
        A[0] = A[-1]
        A[-1] = mx
        # new array of size n-1
        print(A.pop())
        # retore the heapness
        max_heapify(A, 0)


def radix_sort(x_input, b=10, max_d=None):
    """ 
    b: base, default 10 
    max_d: maximum number of digits
    """
    
    from math import log
    from copy import deepcopy
    #from collections import deque # use dible queue

    def get_digit(num, i):
        """
        example: num=1234 ==> get_digit(num, 1) = 4 and get_digit(num, 4) = 1
        """
        return (num%b**i)//(b**(i-1))     
    
    x = deepcopy(x_input)
    if not max_d:
        max_val = max(x)
        max_d = int(log(max_val)/log(b))+ 1
        print("max:{}, digits:{}".format(max_val, max_d))

    #for i in reversed(range(1, max_d+1)):
    # LSD sort: least significant digit radix sort
    for i in range(1, max_d+1): # loop over the digits for the given base
        L = []
        for j in range(b):
            L.append([])
        for num in x: # loop over the input list
            dig = get_digit(num, i)
            L[dig].append(num)
        x = []
        for l in L:
            x.extend(l)
        #print(x)
    print("Sorted: {}".format(x))

    
# Lomuto scheme, didnt work    
def partition_1(X, lo, hi):
    print(X)
    pivot = X[hi]
    print("pivot: ", pivot)
    new_pid = lo
    for i in range(lo, hi-1):
        if X[i] <= pivot:
            X[new_pid], X[i] = X[i], X[new_pid]
            new_pid += 1
    X[new_pid], X[hi] = X[hi], X[new_pid]
    print("new: ", new_pid)
    print(X)
    return new_pid


# Hoare scheme, right pivot(the right most data point
# is used as the pivot)
def partition_2(X, lo, hi):
    #print(X, lo, hi)
    pivot = X[hi]    
    left, right = lo, hi-1

    while True:
        while left<=right and X[left] <= pivot:
            left += 1
        while left<=right and X[right] >= pivot:
            right -= 1
        if left > right:
            X[hi], X[left] = X[left], X[hi]
            #print(X, left)
            return left
        # flip and repeat
        X[right], X[left] = X[left], X[right]
        #print(X,":", left,":", right)

           
# Hoare scheme, left pivot, copied from web
def partition_3(alist,first,last):
   pivotvalue = alist[first]

   leftmark = first+1
   rightmark = last

   done = False
   while not done:

       while leftmark <= rightmark and alist[leftmark] <= pivotvalue:
           leftmark = leftmark + 1

       while alist[rightmark] >= pivotvalue and rightmark >= leftmark:
           rightmark = rightmark -1

       if rightmark < leftmark:
           done = True
       else:
           temp = alist[leftmark]
           alist[leftmark] = alist[rightmark]
           alist[rightmark] = temp

   temp = alist[first]
   alist[first] = alist[rightmark]
   alist[rightmark] = temp

   return rightmark


def quick_sort(A, lo, hi):
    # lo=0, hi=len(A)-1
    if lo<hi:
        p = partition_2(A, lo, hi)
        print(p)
        quick_sort(A, lo, p-1) # 0, p-1
        quick_sort(A, p+1, hi) # p+1, hi
