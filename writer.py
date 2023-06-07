# Copyright © 2023 Paul Viallard <paul.viallard@gmail.com>
# This work is free. You can redistribute it and/or modify it under the
# terms of the Do What The Fuck You Want To Public License, Version 2,
# as published by Sam Hocevar. See http://www.wtfpl.net/ for more details.

import os
import torch
import pickle
import hashlib
import numpy as np
from lock import Lock

###############################################################################


class WriterFile(Lock):

    def __init__(self, file_):
        self.erase = True
        super().__init__(file_)

    def _save(self):
        raise NotImplementedError

    def _load(self):
        raise NotImplementedError

    # ----------------------------------------------------------------------- #
    # Functions

    def write(self, **kwargs):
        return self.do(self.__write, **kwargs)

    def __write(self, **kwargs):
        # We load the writer
        self._load()

        # For each key/value
        for key, val in kwargs.items():
            # We create the key if it does not exist in the writer
            if(key not in self.file_dict):
                self.file_dict[key] = []
            # We add the value in the writer
            self.file_dict[key].append(val)

        # We save the writer
        self._save()

    def remove(self, key_list):
        return self.do(self.__remove, key_list)

    def __remove(self, key_list):
        # We load the writer
        self._load()

        # For each key
        for key in key_list:
            # We remove it in the writer
            if(key in self.file_dict):
                del self.file_dict[key]

        # We save the writer
        self._save()

    def __contains__(self, key):
        return self.do(self.__contains, key)

    def __contains(self, key):
        # We load the writer
        self._load()
        # We return if the key in in the dictionary
        return key in self.file_dict

    def __getitem__(self, key):
        return self.do(self.__getitem, key)

    def __getitem(self, key):
        # We load the writer
        self._load()

        # We return the item associated with the key
        if(isinstance(self.file_dict[key], list)
           and len(self.file_dict[key]) == 1):
            return self.file_dict[key][0]
        return self.file_dict[key]

    def keys(self):
        return self.do(self.__keys)

    def __keys(self):
        # We load the writer
        self._load()
        # We return the keys
        return self.file_dict.keys()

###############################################################################


class NpzWriterFile(WriterFile):

    def _save(self):

        # We remove the lists in the dictionary
        new_file_dict = {}
        for key, val_list in self.file_dict.items():
            i = 0
            for val in val_list:
                new_file_dict[key+"_"+str(i)] = np.array(val)
                i += 1

        # We save the writer file as a npz file
        np.savez_compressed(self._lock_file, **new_file_dict)
        os.rename(self._lock_file+".npz", self._lock_file)

    def _load(self):

        # We load the writer file
        old_file_dict = {}
        if(os.path.getsize(self._lock_file) > 0):
            old_file_dict = dict(np.load(self._lock_file))

        # We retrieve the original list of keys
        key_list = []
        for key in old_file_dict.keys():
            new_key = key.split("_")[:-1]
            new_key = "_".join(new_key)
            if(new_key not in key_list):
                key_list.append(new_key)

        # We retrieve the lists
        self.file_dict = {}
        for key in key_list:
            i = 0
            while(key+"_"+str(i) in old_file_dict):
                if(key not in self.file_dict):
                    self.file_dict[key] = []
                self.file_dict[key].append(old_file_dict[key+"_"+str(i)])
                i += 1

###############################################################################


class PklWriterFile(WriterFile):

    def _save(self):
        # We save the writer file as a pickle file
        with open(self._lock_file, "wb") as f:
            pickle.dump(self.file_dict, f)

    def _load(self):
        # We load the writer file
        self.file_dict = {}
        if(os.path.getsize(self._lock_file) > 0):
            with open(self._lock_file, "rb") as f:
                self.file_dict = pickle.load(f)

###############################################################################


class PthWriterFile(WriterFile):

    def _save(self):
        # We save the writer file as a PyTorch file
        torch.save(self.file_dict, self._lock_file)

    def _load(self):
        # We load the writer file
        self.file_dict = {}
        if(os.path.getsize(self._lock_file) > 0):
            self.file_dict = torch.load(self._lock_file)

###############################################################################


class Writer():

    def __init__(self, path, mode="pkl"):
        # We save the folder path
        self._folder_path = os.path.abspath(path)
        # and we create it
        os.makedirs("{}".format(self._folder_path), exist_ok=True)
        # We save the mode of saving the writer files
        self.mode = mode
        if(self.mode not in ["npz", "pkl", "pth"]):
            raise RuntimeError(
                "mode must be either npz, pkl, pth")

    def open(self, **kwargs):
        # We get the name of the writer file
        path = self.__get_path(kwargs)
        # and we open it

        if(self.mode == "npz"):
            self.writer_file = NpzWriterFile(path)
        elif(self.mode == "pkl"):
            self.writer_file = PklWriterFile(path)
        elif(self.mode == "pth"):
            self.writer_file = PthWriterFile(path)
        else:
            raise RuntimeError(
                "mode must be either npz, pkl, pth")

    def __get_path(self, key_val_dict):

        # We create the string with the keys and values
        file_path = ""
        for key in sorted(list(key_val_dict.keys())):
            file_path += "{}={}/".format(key, key_val_dict[key])

        # We hash it to obtain the name of the writer file
        h = hashlib.new('sha256')
        h.update(str.encode(file_path[:-1]))
        file_path = h.hexdigest()

        # We get the path of the file
        path = os.path.join(self._folder_path, file_path)
        return path

    # ----------------------------------------------------------------------- #
    # Functions

    def write(self, **kwargs):
        return self.writer_file.write(**kwargs)

    def remove(self, key_list):
        return self.writer_file.remove(key_list)

    def __contains__(self, key):
        return self.writer_file.__contains__(key)

    def __getitem__(self, key):
        return self.writer_file.__getitem__(key)

    def keys(self):
        return self.writer_file.keys()
