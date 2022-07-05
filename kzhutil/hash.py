import hashlib
from typing import AnyStr


def md5(s: AnyStr, encoding="utf-8"):
    """
    return md5 hex string

    :param s: a string or bytes
    :param encoding: only used to encode s while s is a string
    """
    if isinstance(s, str):
        s = s.encode(encoding)
    return hashlib.md5(s).hexdigest()


def sha1(s: AnyStr, encoding="utf-8"):
    """
    return sha1 hex string

    :param s: a string or bytes
    :param encoding: only used to encode s while s is a string
    """
    if isinstance(s, str):
        s = s.encode(encoding)
    return hashlib.sha1(s).hexdigest()


def sha224(s: AnyStr, encoding="utf-8"):
    """
    return sha224 hex string

    :param s: a string or bytes
    :param encoding: only used to encode s while s is a string
    """
    if isinstance(s, str):
        s = s.encode(encoding)
    return hashlib.sha224(s).hexdigest()


def sha256(s: AnyStr, encoding="utf-8"):
    """
    return sha256 hex string

    :param s: a string or bytes
    :param encoding: only used to encode s while s is a string
    """
    if isinstance(s, str):
        s = s.encode(encoding)
    return hashlib.sha256(s).hexdigest()


def sha384(s: AnyStr, encoding="utf-8"):
    """
    return sha384 hex string

    :param s: a string or bytes
    :param encoding: only used to encode s while s is a string
    """
    if isinstance(s, str):
        s = s.encode(encoding)
    return hashlib.sha384(s).hexdigest()


def sha512(s: AnyStr, encoding="utf-8"):
    """
    return sha512 hex string

    :param s: a string or bytes
    :param encoding: only used to encode s while s is a string
    """
    if isinstance(s, str):
        s = s.encode(encoding)
    return hashlib.sha512(s).hexdigest()
