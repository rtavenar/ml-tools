This repository contains some tools that were useful for my machine learning experiments.

### Description of the tools

* **A system for running different scripts at once (in _run.py_)**

The file *run.py* allows you to run different commands described in a .ini file.
The script is able to (1) print the commands to execute, (2) to run them on the computer, and (3) to launch them on a Slurm cluster.

* **A generic class for file locking (in _lock.py_)**

The file *lock.py* contains a generic class for handling the (concurrent) modifications of a unique file on different computers. For instance, the class handles the concurrent modifications in a Slurm cluster.  
* **A class to save a csv file**

The file *nd_data.py* contains a class handling the saving of csv files (that is heavily based on Pandas).

* **A class to save Python objects**

The file *writer.py* contains a class handling the saving of Python objects.

### Running the examples 

##### writer.py
To run the example, you need to execute the following command in your bash shell.
```bash
python example/example_writer.py
```

##### nd_data.py
To run the example, you need to execute the following command in your bash shell.
```bash
python example/example_nd_data.py
```

##### run.py
To run the example, you need to execute the following command in your bash shell.
```bash
python run.py print example/example_run.ini
python run.py local example/example_run.ini
python run.py slurm example/example_run.ini
```
