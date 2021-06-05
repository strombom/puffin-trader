import base64
import hashlib
import pathlib
import pickle

import pandas as pd


def make_hashable(o):
    if isinstance(o, (tuple, list)):
        return tuple((make_hashable(e) for e in o))

    if isinstance(o, dict):
        return tuple(sorted((k,make_hashable(v)) for k,v in o.items()))

    if isinstance(o, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in o))

    return o


def make_hash_sha256(o):
    hasher = hashlib.sha256()
    hasher.update(repr(make_hashable(o)).encode())
    return base64.b32encode(hasher.digest()).decode()


def cache_it(fun):
    def cache_wrapper(*args, **kwargs):
        new_kwargs = {}
        for kwarg in kwargs:
            if type(kwargs[kwarg]) == pd.DataFrame:
                new_kwargs[kwarg] = kwargs[kwarg].shape
            else:
                new_kwargs[kwarg] = kwargs[kwarg]

        hash_sum = make_hash_sha256(new_kwargs.items())[0:16]
        path = f"cache/{fun.__name__}"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        file_path = f"{path}/{hash_sum}.pickle"

        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            pass

        data = fun(*args, **kwargs)

        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        return data
    return cache_wrapper
