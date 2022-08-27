# My utils for python，让常用功能可以一行搞定

Bisect
-----------
### MappingSpace (arr: Sequence, key: Callable)
Bisect in python < 3.9 have no key parameter. Use MappingSpace for map f: arr->f(arr) for bisect model.

python < 3.9 的 bisect 不带key参数，可以用这个类作为映射：

```python
import bisect
from kzhutil.bisect import MappingSpace

bisect.bisect(MappingSpace(range(10**8), lambda x: x**2), 2e14)
# 14142136
```

File
-----------
一些将读写json、csv和普通文件浓缩成一行的方法

read write or append a file in one line: 
### read_from_file(fn, binary=False, encoding=None)
### write_over_file(fn, s: AnyStr, binary=False, encoding=None)
### append_to_file(fn, s: AnyStr, binary=False, encoding=None)

-----------
jsonl file: list of jsons separated into lines
### write_jsonl(fn, data: List[object], encoding=None)
### read_jsonl(fn, encoding=None)

-----------
csv file
### read_csv(fn, encoding=None)
### write_csv(fn, data: List[object], encoding=None)
### read_csv_dict(fn, encoding=None)
### write_csv_dict(fn, fieldnames, data: List[Dict], encoding=None)


Functools
-----------
### try_for(n)
A decorator to try a function up to n times till success. Useful for function with failure probability. e.g. web crawler.

一个函数装饰器，可以让后面的函数最多尝试n次，直到某次运行成功再返回。适合用于爬虫这类有成果概率的场景。

```python
import random
from kzhutil.functools import try_for

random.seed(42)

@try_for(10)
def func():
    n = random.randint(0, 10)       # 10, 1, 0, 4, 3, 3, 2, 1, 10, 8
    print(n, end=" ")
    assert n == 3
    return 'success'

print(func())   # 10 1 0 4 3 success
```
### try_for_(n, f: Callable, *args, **kwargs)
The kernel of try_for(n)

### try_until(f: Callable, args: Iterable)
try f(*args[0]), f(*args[1]), f(*args[2]), ... until success then return

### repeat_for(n)
repeat a function for n times
### repeat_for_(n, f: Callable, *args, **kwargs)



Hash
-----------
Hash string/bytes in one line and return hex string. All hash functions:

比起hashlib，这里提供的hash函数既能接受string又能接受bytes，并且直接返回十六进制string，所有hash函数：

**md5, sha1, sha224, sha256, sha384, sha512**



io_util
-----------
### get_clipboard()
Windows Only. 获得剪切板内容，只支持windows平台。
### set_clipboard(s: AnyStr)
Windows Only. 设置剪切板内容，只支持windows平台。


Math
----------
### normalize(array, **kwargs)
Normalize the array to standard normal distribution. 正态分布归一化

array: numpy.ndarray or torch.Tensor.

**kwargs: (axis, keepdims) for numpy or (dim, keepdim) for pytorch.

### uniform_normalize(array, **kwargs)
Normalize the array to uniform distribution [0, 1]. 归一化

array: numpy.ndarray or torch.Tensor.

**kwargs: (axis, keepdims) for numpy or (dim, keepdim) for pytorch.

### primes(n)
Get all primes not large than n.
```python
from kzhutil.math import primes
primes(19)  # [2, 3, 5, 7, 11, 13, 17, 19]
```

### exgcd(a, b)
Get solution x, y for ax+by=gcd(a,b).

return: x, y, gcd(a, b)
```python
from kzhutil.math import exgcd
exgcd(18, 44)   # 5, -2, 2
```

### inverse(x, n)
Get the inverse element ![](https://latex.codecogs.com/gif.latex?x^{-1}) of ![](https://latex.codecogs.com/gif.latex?x) in ![](https://latex.codecogs.com/gif.latex?Z_n) where ![](https://latex.codecogs.com/svg.image?x%5E%7B-1%7Dx%5Cequiv%201%5C%20(mod%5C%20n))

### continued_fraction(d, eps=1e-12, max_denominator=100000000)
Calculate continued fraction coefficients [a0, a1, a2, ...] of ![](https://latex.codecogs.com/svg.image?d=a_0&plus;\frac{1}{a_1&plus;\frac{1}{a_2&plus;\frac{1}{a_3&plus;...}}})

e.g. π -> [3, 7, 15, 1, 292, 1, 1, 1, 2, 1, 3]

### convergent_fraction(a)
Get the convergent fraction of a cotinued fraction. e.g. [3, 7, 15, 1] -> 355/113


### miller_rabin(n, k=16)
Miller-Rabin prime test with time complexity in ![](https://latex.codecogs.com/svg.image?O(klog^3(n))).
The error rate is approximate to ![](https://latex.codecogs.com/svg.image?(1%2F4)%5Ek)

If k is 0, use deterministic miller test(deterministic if generalized Riemann hypothesis proved).

### next_prime(n)
Get the next prime of n (n>2)

### random_prime(n)
Get a random prime smaller than ![](https://latex.codecogs.com/svg.image?10^n)

### factorization(n, deterministic=False)
return factors and exponents of n. (may run for real long time)
```python
import kzhutil
n = 357686312646216567629136   # 2*2*2*2*3*41*307*367*1061*1520398399903
kzhutil.math.factorization(n)   # {2: 4, 3: 1, 41: 1, 307: 1, 367: 1, 1061: 1, 1520398399903: 1}
```

### lucas_test(n)
A deterministic primality test (may run for real long time).
The running speed determined by whether n-1 is well factorized.

### euler_phi(n)
Euler's totient function.

torch
----------
utils for pyroch
### set_seed(seed)
set seed for torch, numpy, random

transformers
-----------
### load_pretrained(model_class, model_name, model_path, **kwargs)
Load pretrained transformers. If a saved model provided in `model_path`, then load directly. Else load from hub and save to `model_path`.
```python
from kzhutil.transformers import load_pretrained
from transformers import AutoModelForMaskedLM
load_pretrained(AutoModelForMaskedLM, "bert-base-uncased", "/home/bert")
```
