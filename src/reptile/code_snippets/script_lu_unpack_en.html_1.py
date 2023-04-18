.. code-block:: text
    ones = eye(rows) #eye matrix of rank rows
    for i in range(cols):
        swap(ones[i], ones[pivots[i]])