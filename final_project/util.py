#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numbers
import fractions
import numpy as np


def symfrac_ticks(
    start: numbers.Rational,
    stop: numbers.Rational,
    step: numbers.Rational,
    sym_numeric: numbers.Number,
    sym_repr: str,
):
    def _str_symfrac(frac: numbers.Rational, sym: str) -> str:
        if frac == 0:
            return str("0")

        def _elide_ones_str(x, before: str = "", after: str = "") -> str:
            if x == 1:
                return ""
            return before + str(x) + after

        frac = fractions.Fraction(frac)
        return (
            _elide_ones_str(frac.numerator)
            + sym
            + _elide_ones_str(frac.denominator, before="/")
        )

    nprange = sym_numeric * np.arange(start, stop, step=step)
    start, step = fractions.Fraction(start), fractions.Fraction(step)
    labels = [_str_symfrac(start + i * step, sym_repr) for i in range(len(nprange))]
    return nprange, labels
