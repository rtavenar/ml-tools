# Copyright © 2023 Paul Viallard <paul.viallard@gmail.com>
# This work is free. You can redistribute it and/or modify it under the
# terms of the Do What The Fuck You Want To Public License, Version 2,
# as published by Sam Hocevar. See http://www.wtfpl.net/ for more details.

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nd_data import NDData

# We create a csv file (from NDData)
data = NDData("nd_data.csv")

# We create new data
data.set({"col1": "val1", "col2": "val2"}, {"i1": "ind1", "i2": "ind2"})
data.set({"col1": 1, "col2": 2}, {"i1": 3, "i2": 4})

# We print the pandas dataframe
print(data)

# We get the column "col1" when "i1=3"
print(data.get("col1", i1=3))

# We print the keys in the indices
print(data.index_keys())

# We print the values associated with the key "i1" in the indices
print(data.index_keys("i1"))

# We print the column names
print(data.col_keys())

# We remove the csv file
os.remove("nd_data.csv")
