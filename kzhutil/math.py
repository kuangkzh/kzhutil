import math
import random
import fractions
import _kzhutil


def normalize(array, **kwargs):
    """
    normalize the array to N(0, 1) by mean and std
    """
    mu, std = array.mean(**kwargs), array.std(**kwargs)
    return (array - mu) / std


def uniform_normalize(array, **kwargs):
    """
    normalize the array to U(0, 1)
    """
    ma, mi = array.max(**kwargs), array.min(**kwargs)
    return (array - mi) / (ma - mi)


primes = _kzhutil.primes


def exgcd(a, b):
    """
    get solution x, y for ax+by=gcd(a,b)

    :return: x, y, gcd(a, b)
    """
    def _exgcd(_a, _b):
        if _b == 0:
            return 1, 0, _a
        else:
            x, y, gcd = _exgcd(_b, _a % _b)
            x, y = y, (x - (_a // _b) * y)
            return x, y, gcd
    return _exgcd(a, b)


def inverse(x, n):
    """
    get the inverse element x^{-1} of x in Z_n where x^{-1} * x = 1 (mod n).
    gcd(x, n) must equal to 1 for the correct inverse, else return 0.

    :return: x^{-1}
    """
    inv, _, gcd = exgcd(x, n)
    return (inv + n) % n if gcd == 1 else 0


def continued_fraction(d, eps=1e-12, max_denominator=100000000):
    """
    calculate continued fraction coefficients [a0, a1, a2, ...] of d = a0 + 1/(a1 + 1/(a2 + 1/(a3 + ...)))
    """
    res, d = [int(d//1)], d % 1
    p_, q_, p, q = 1, 0, res[0], 1      # p/q is the convergent of continued fraction
    while True:
        a = int(1/d)
        p_, q_, p, q = p, q, a*p + p_, a*q + q_
        if q*(d*q + q_) > 1/eps or q > max_denominator: break     # error evaluate
        res.append(a)
        d = 1 / d % 1
    return res


def convergent_fraction(a):
    """
    get the convergent fraction of a cotinued fraction. e.g. [3, 7, 15, 1] -> 355/113
    """
    p_, q_, p, q = 1, 0, a[0], 1  # p/q is the convergent of continued fraction
    for ai in a[1:]:
        p_, q_, p, q = p, q, ai*p + p_, ai*q + q_
    return fractions.Fraction(p, q)


primes_1000 = set(primes(1000))


def miller_rabin(n, test_time=16):
    """
    Miller-Rabin prime test
    :param n: the number
    :param test_time: how many miller-rabin test to try. the error rate is approximate to (1/4)^t
    :return: if n is a prime
    """
    if n <= 1000:
        return n in primes_1000
    if any(n % p == 0 for p in primes_1000):
        return False
    u, t = n-1, 0
    while u % 2 == 0:   # divide n-1 = u * 2^t
        u, t = u//2, t+1
    for i in range(test_time):
        a = random.randint(0, 2**31) % (n - 2) + 2
        v = pow(a, u, n)    # a^u mod n
        if v == 1: continue
        for _ in range(t):
            if v == n - 1: break
            v = v * v % n
        else:
            return False
    return True


def next_prime(n):
    """
    the next prime of n (n>2)
    """
    n += 1 if n % 2 == 0 else 2
    while not miller_rabin(n):
        n += 2
    return n


def random_prime(n):
    """
    get a random prime contains n digits
    """
    num = 0
    for _ in range(n):
        num = num*10 + random.randint(0, 9)
    return next_prime(num)


def euler_phi(n):
    ans = n
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            ans = ans // i * (i - 1)
            while n % i == 0:
                n = n // i
    if n > 1:
        ans = ans // n * (n - 1)
    return ans


def linear_interp(x, x0, x1):
    # TODO
    pass


def linear_interp2d(x, y, x00, x01, x10, x11):
    # TODO
    pass


class Triangle:
    def __init__(self, a=None, b=None, c=None, A=None, B=None, C=None, mode='rad'):
        assert mode in ('rad', 'deg'), "angle mode must be 'rad' or 'deg'"
        self.a, self.b, self.c = a, b, c
        self.A, self.B, self.C = (A % math.pi, B % math.pi, C % math.pi) if mode == 'rad' \
            else ((A % 180)*math.pi/180, (B % 180)*math.pi/180, (C % 180)*math.pi/180)
        self.mode = mode

    def __repr__(self):
        if self.mode == 'rad':
            return f"Triangle(a={self.a}, b={self.b}, c={self.c}, A={self.A/math.pi}*pi, B={self.B/math.pi}*pi, C={self.C/math.pi}*pi, mode={self.mode})"
        else:
            return f"Triangle(a={self.a}, b={self.b}, c={self.c}, A={self.A/math.pi*180}, B={self.B/math.pi*180}, C={self.C/math.pi*180}, mode={self.mode})"
    __str__ = __repr__

    def __eq__(self, other):
        return self.a == other.a and self.b == other.a and self.c == other.c

    @staticmethod
    def sss(a, b, c, mode='rad'):
        A = math.acos((b*b + c*c - a*a) / (2*b*c))
        B = math.acos((a*a + c*c - b*b) / (2*a*c))
        C = math.acos((b*b + a*a - c*c) / (2*b*a))
        A, B, C = (A, B, C) if mode == 'rad' else (A/math.pi*180, B/math.pi*180, C/math.pi*180)
        return Triangle(a, b, c, A, B, C, mode)

    @staticmethod
    def ssa(a, b, C, mode='rad'):
        C = C if mode == 'rad' else C*math.pi/180
        c = a*a + b*b - 2*a*b*math.cos(C)
        A = math.acos((b*b + c*c - a*a) / (2*b*c))
        B = math.acos((a*a + c*c - b*b) / (2*a*c))
        return Triangle(a, b, c, A, B, C, mode)

    @staticmethod
    def saa(a, B, C, mode='rad'):
        B, C = (B, C) if mode == 'rad' else (B*math.pi/180, C*math.pi/180)
        A = math.pi - B - C
        b = math.sin(B) / math.sin(A) * a
        c = math.sin(C) / math.sin(A) * a
        return Triangle(a, b, c, A, B, C, mode)
