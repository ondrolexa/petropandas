"""Parser/evaluator for the compiled polynomial notation used by THERMOCALC axfiles for
`p(end-member)` and site-fraction (`sf`) blocks.

Grammar: a named polynomial is `name numTerms term1 term2 ... termN` (the first term's
tokens continue on the header line; later terms are separate, indented continuation
lines). Each term is `numFactors factor1 ... factorK`; a term's value is the *product* of
its factor values. Each factor is `constant numPairs (coeff1 var1) ... (coeffN varN)`; a
factor's value is `constant + sum(coeff * var)`. A polynomial's value is the *sum* of its
term values. Coefficients may be fractions written as a single "num/den" token.
"""

from __future__ import annotations

import re

import pandas as pd

Pair = tuple[float, str]
Factor = tuple[float, tuple[Pair, ...]]
Term = tuple[Factor, ...]
Polynomial = tuple[Term, ...]

_P_NAME = re.compile(r"^p\((\w+)\)$")


def _parse_number(token: str) -> float | None:
    if "/" in token:
        numerator, _, denominator = token.partition("/")
        try:
            return float(numerator) / float(denominator)
        except ValueError:
            return None
    try:
        return float(token)
    except ValueError:
        return None


def _tokenize(text: str) -> list[str]:
    tokens = []
    for line in text.splitlines():
        line = line.partition("%")[0].strip()
        if line:
            tokens.extend(line.split())
    return tokens


def parse_polynomials(text: str) -> dict[str, Polynomial]:
    """Parse every named polynomial in `text` into a {name: Polynomial} mapping."""
    tokens = _tokenize(text)
    polynomials: dict[str, Polynomial] = {}
    pos = 0

    def next_int() -> int:
        nonlocal pos
        value = int(tokens[pos])
        pos += 1
        return value

    def next_number() -> float:
        nonlocal pos
        value = _parse_number(tokens[pos])
        pos += 1
        return value

    def next_name() -> str:
        nonlocal pos
        value = tokens[pos]
        pos += 1
        return value

    while pos < len(tokens):
        name = next_name()
        num_terms = next_int()
        terms = []
        for _ in range(num_terms):
            num_factors = next_int()
            factors = []
            for _ in range(num_factors):
                constant = next_number()
                num_pairs = next_int()
                pairs = tuple((next_number(), next_name()) for _ in range(num_pairs))
                factors.append((constant, pairs))
            terms.append(tuple(factors))
        polynomials[name] = tuple(terms)
    return polynomials


def evaluate_polynomial(polynomial: Polynomial, variables: pd.DataFrame) -> pd.Series:
    """Vectorized evaluation of a single parsed `Polynomial` over `variables`."""
    total = 0.0
    for factors in polynomial:
        term_value = 1.0
        for constant, pairs in factors:
            factor_value = constant
            for coeff, var in pairs:
                factor_value = factor_value + coeff * variables[var]
            term_value = term_value * factor_value
        total = total + term_value
    return (
        pd.Series(total, index=variables.index)
        if not isinstance(total, pd.Series)
        else total
    )


def evaluate_polynomials(text: str, variables: pd.DataFrame) -> pd.DataFrame:
    """Parse and evaluate every named polynomial in `text`, as a DataFrame aligned to
    `variables.index`. Names of the form "p(name)" are unwrapped to just "name"."""
    polynomials = parse_polynomials(text)

    def column_name(name: str) -> str:
        match = _P_NAME.match(name)
        return match.group(1) if match else name

    return pd.DataFrame(
        {
            column_name(name): evaluate_polynomial(poly, variables)
            for name, poly in polynomials.items()
        },
        index=variables.index,
    )
