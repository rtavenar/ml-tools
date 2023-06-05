# Copyright © 2023 Paul Viallard <paul.viallard@gmail.com>
# This work is free. You can redistribute it and/or modify it under the
# terms of the Do What The Fuck You Want To Public License, Version 2,
# as published by Sam Hocevar. See http://www.wtfpl.net/ for more details.

import os
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from writer import Writer

# We create a writer folder and save the file with the npz format
writer = Writer("writer_folder", mode="pkl")
# We open the file associated with the key/value pairs a=1 and b="c"
writer.open(a=1, b="c")

# We remove the list associated with the keys "test1" and "test2"
writer.remove(["test1", "test2"])

# We write in the file the values
# [2.3, 3.0, 4.0] for the key "test1"
# and [[1, 2], [3, 4], [5, 6]] for the key "test2"
writer.write(test1=2.3, test2=[1, 2])
writer.write(test1=3.0, test2=[3, 4])
writer.write(test1=4.0, test2=[5, 6])

# We print the keys
print(writer.keys())

# We check that the key "test3" is in the writer file
print("test3" in writer)

# We print the values associated with "test1" and "test2"
print(writer["test1"])
print(writer["test2"])

# We remove the writer folder
shutil.rmtree("writer_folder")
