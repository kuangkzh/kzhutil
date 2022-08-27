import json
import csv
import sys
import math
from kzhutil.functools import try_until
from typing import List, Dict, AnyStr


maxInt = sys.maxsize
try_until(csv.field_size_limit, [maxInt//(2**i) for i in range(int(math.log2(maxInt)))])


def read_from_file(fn, binary=False, encoding=None):
    """
    get all data of a file

    :param fn: file name or path
    :param binary: if true open in binary mode and return bytes
    :param encoding:
    :return: all data of the file
    """
    with open(fn, "r" if not binary else "rb", encoding=encoding) as f:
        return f.read()


def write_over_file(fn, s: AnyStr, binary=False, encoding=None):
    """
    overwrite a file

    :param fn: file name or path
    :param s: content to write
    :param binary: if true open in binary mode and s should be bytes
    :param encoding:
    """
    with open(fn, "w" if not binary else "wb", encoding=encoding) as f:
        return f.write(s)


def append_to_file(fn, s: AnyStr, binary=False, encoding=None):
    """
    append content to the end of a file

    :param fn: file name or path
    :param s: content to write
    :param binary: if true open in binary mode and s should be bytes
    :param encoding:
    """
    with open(fn, "a" if not binary else "ab", encoding=encoding) as f:
        return f.write(s)


def write_jsonl(fn, data: List[object], encoding=None, **json_kwargs):
    """
    write data to a jsonl file, where a long list is split into lines of json

    :param fn: file name or path
    :param data: a list data
    :param encoding:
    """
    with open(fn, 'w', encoding=encoding) as f:
        return f.write("\n".join([json.dumps(dl, **json_kwargs) for dl in data]))


def read_jsonl(fn, encoding=None, **json_kwargs) -> List[object]:
    """
    read from a jsonl file, where a long list is split into lines of json

    :param fn: file name or path
    :return: a list data
    :param encoding:
    """
    with open(fn, 'r', encoding=encoding) as f:
        return [json.loads(dl, **json_kwargs) for dl in f.readlines()]


def read_csv(fn, encoding=None, **csv_kwargs):
    """
    read a csv(comma-separated values) file

    :param fn: file name or path
    :return: list containing all data in .csv file
    :param encoding:
    """
    with open(fn, 'r', newline='', encoding=encoding) as f:
        return [*csv.reader(f, **csv_kwargs)]


def write_csv(fn, data: List[object], encoding=None, **csv_kwargs):
    """
    write to a file

    :param fn: file name or path
    :param data: data needed to be written to a csv file
    :param encoding:
    """
    with open(fn, 'w', newline='', encoding=encoding) as f:
        writer = csv.writer(f, **csv_kwargs)
        for row in data:
            writer.writerow(row)


def read_csv_dict(fn, encoding=None, **csv_kwargs):
    """
    read a csv(comma-separated values) file

    :param fn: file name or path
    :return: list containing all data in .csv file
    :param encoding:
    """
    with open(fn, 'r', newline='', encoding=encoding) as f:
        return [*csv.DictReader(f, **csv_kwargs)]


def write_csv_dict(fn, fieldnames, data: List[Dict], encoding=None, **csv_kwargs):
    """
    write to a file

    :param fn: file name or path
    :param fieldnames: fieldnames of the csv header
    :param data: data needed to be written to a csv file
    :param encoding:
    """
    with open(fn, 'w', newline='', encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, **csv_kwargs)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
