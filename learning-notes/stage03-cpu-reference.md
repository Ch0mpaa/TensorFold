# Stage 3 retrospective — the Python CPU reference

**For:** the project owner (you)
**Length:** ~10 min read
**Prerequisites:** none; this explains the "why" from scratch

---

## Why did we build this at all?

Here's the honest version. By the end of Stage 2 we had a `.tfold` file
that could be written and read back without corruption. Great — but
that only proves *data in, data out*. It doesn't prove the file is
**useful**: that we can actually do math with the weights inside.

Stage 3's one job was to prove that. We wrote a deliberately-simple
Python program that:

1. Opens `examples/demo.tfold` (built in Stage 2)
2. Pulls the three tensors out of it (weight, bias, input)
3. Does the math `y = x @ W + b` by hand — **no numpy, no BLAS, no library magic**
4. Compares the answer to an independent reference that numpy computed
5. Reports how close the two answers are

If our file is any good at all, the answer must match the reference. If
it doesn't, something is wrong — either the file corrupted numbers, or
our math is wrong, and we need to know which.

This is the equivalent of a mechanic hooking up a brand-new engine to
a known-good dynometer. The engine either spins the drum at the right
RPM or it doesn't. No sales pitch, no benchmarks — just: *does it
move*.

---

## How the matrix calculation actually works

The demo computes `y = x @ W + b` where:

| Symbol | Meaning | Shape |
|---|---|---|
| `x` | the input vector | `[8]` — eight floats |
| `W` | the weight matrix | `[8, 4]` — eight rows of four columns |
| `b` | the bias vector | `[4]` — four floats |
| `y` | the output vector | `[4]` — four floats |

The `@` symbol in Python is matrix multiplication. For our shapes it's
really just "for each of the four output slots, add up the eight
`x[k] * W[k][n]` products". Picture:

```
x = [x0 x1 x2 x3 x4 x5 x6 x7]                        (length 8)

W = ⎡ w00 w01 w02 w03 ⎤                               (8 rows x 4 cols)
    ⎢ w10 w11 w12 w13 ⎥
    ⎢ w20 w21 w22 w23 ⎥
    ⎢  …   …   …   …  ⎥
    ⎣ w70 w71 w72 w73 ⎦

y[0] = x0*w00 + x1*w10 + x2*w20 + x3*w30 + x4*w40 + x5*w50 + x6*w60 + x7*w70
y[1] = x0*w01 + x1*w11 + x2*w21 + ...
y[2] = ...
y[3] = ...

then: y[i] += b[i]
```

Our Python version in `tools/tfold_cpu/ops.py` does exactly this with a
pair of nested `for` loops. Deliberately. No clever tricks. Every
multiply and add is visible on the page.

---

## How the numerical results are verified

The verification is a three-way comparison:

1. **tfold_cpu answer** — our Python loops, reading from `demo.tfold`
2. **numpy reference** — numpy's own `@` operator, reading from `demo.reference.npz`
3. **expected tolerance** — documented in `tools/tfold_cpu/policy.py`:
   ```python
   FP32_TOL = {"atol": 1e-5, "rtol": 1e-5}
   ```

The test (`tests/test_cpu_executor.py::test_executor_matches_numpy_reference_within_tolerance`)
checks every element of our output against the numpy element. For our
tiny 4-element output, we got:

| index | tfold_cpu | numpy | difference |
|---|---|---|---|
| 0 | -3.5695247650146484 | -3.5695247650146484 | **0.0** (exact) |
| 1 | -1.8333334922790527 | -1.8333334922790527 | **0.0** (exact) |
| 2 | 0.1698397994041443 | 0.1698397994041443 | **0.0** (exact) |
| 3 | 0.9130314588546753 | 0.9130312204360962 | 2.38e-7 (1 ULP) |

**Three out of four are bit-exact. The fourth differs by one unit in
the last place.** ULP means "unit in the last place" — it's the
smallest nonzero amount an FP32 number can change by. For FP32, that's
roughly 1 part in 8 million. Nothing to worry about, but worth
explaining.

### Why isn't it bit-exact?

Floating-point addition is **not associative**: `(a + b) + c` is not
always exactly `a + (b + c)` when the numbers are represented in FP32.
Our Python loops add numbers strictly left to right. numpy's `@`
operator may use *pairwise summation* or a vectorised reduction that
adds them in a different order. The final result differs by a tiny
amount because of the different order of addition.

This isn't a bug in either implementation. It's a fundamental property
of floating-point arithmetic. The honest thing to do is to **document a
tolerance** and test to it, which is what `policy.py` does.

> Philosophy that carries through the whole project: *fast is useful;
> fast and wrong is not*. Here, "wrong" means "silently different by an
> unknown amount". Our tolerance says "different by at most 1e-5
> relative, and we promise it". If we ever exceed that, the test
> fails loudly.

---

## What else the reference enforces

The CPU reference refuses to produce a number when something smells
wrong:

- **NaN or infinity on input**: `FiniteError` — the op refuses to
  propagate garbage down the pipeline
- **Shape mismatch**: `ShapeError` — if you hand it a 5-element input
  and an 8-element weight matrix, it says no instead of silently
  reshaping
- **Wrong dtype**: `CpuError` — if you store integers where fp32 was
  declared, the reference refuses

These rejections mirror what the Stage 2 reader does at the file
level, and what the Stage 4 Zig loader will do in native code. The
pattern is consistent: validate at every boundary, reject explicitly,
never guess.

---

## What Stage 3 is NOT

It's worth being clear about what we explicitly did **not** build so
you can calibrate expectations:

- Not fast. 3.4 ms to compute 32 multiplies + 4 adds. That's ~10 μs
  per multiply, roughly 10,000× slower than a real BLAS kernel.
- Not GPU. Pure Python on the CPU.
- Not a serving backend. Stage 8 is where we plug this into the
  "dyno" runner so people can compare models.
- Not a real model. 32 weights, not 2 billion.

What it **is**: proof that the Stage 2 file format carries data
faithfully, with a repeatable, documented numerical behavior we can
hold Stage 4's native Zig loader to.

---

## Why this unlocked Stage 4

Stage 4 is the native loader. We needed the Python reference first
because Stage 4 has nothing to compare itself against otherwise. Now,
when the Zig loader produces an output, we can diff it byte-for-byte
against the Python answer (or against the tolerance-bounded version).
The reference becomes the oracle.

Think of it like having a reliable calculator next to you when
you're testing a new scientific instrument. The instrument tells you
something; you check the calculator; if they agree within the stated
error, the instrument probably works.

---

## Files to look at

| File | What's in it | Why read it |
|---|---|---|
| `tools/tfold_cpu/ops.py` | the math (matvec, bias_add) | see the explicit `for` loops |
| `tools/tfold_cpu/policy.py` | the FP32 tolerance + non-finite policy | understand what "close enough" means |
| `tools/tfold_cpu/executor.py` | glue between Stage 2 reader and the ops | see how the data flows |
| `tests/test_cpu_executor.py` | the end-to-end verification | this is the proof |
| `examples/demo.tfold` | the 1,192-byte test file | this is what gets loaded |
| `docs/tfold-stage3-conformance.md` | the formal report | one-page summary of what passed and what didn't |
