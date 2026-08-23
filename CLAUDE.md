# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Educational from-scratch implementation of an artificial neural network in a single Python file (`Network_classes.py`). No external dependencies beyond the standard library; a `venv/` is present but currently contains only the Python standard library (no third-party packages installed). The README is just a title.

## Commands

- **Run the demo:** `python Network_classes.py` (or `venv/bin/python Network_classes.py` to use the project venv). The script executes a training run on a single `Neuron` at import time (lines 69–70), so simply running the file is both the demo and the only "test."
- **Activate the venv:** `source venv/bin/activate` — useful if you want to install packages later.
- There is no build step, no test suite, no linter, and no `requirements.txt`. Add any of these before introducing non-stdlib dependencies.

## Architecture

Three classes live in `Network_classes.py`, in this order:

- **`Neuron`** — A single artificial neuron with weights and a bias. The only fully implemented class.
  - `output(inputs)` — Linear forward pass: `bias + Σ input_i * weight_i`. Stores the result on `self.prediction`.
  - `train(inputs, exp_out, learning_rate, epochs)` — Iterates the dataset with a pointer (`input_i`) that wraps back to `0` at the end. For each sample it: computes the prediction, compares to expected output, and (if not matching) updates weights via `adjust_weights`. Three early-exit conditions: exact match, repeated error value (stagnation), or epoch limit reached.
  - `adjust_weights(loss, learning_rate)` — Plain gradient descent on a quadratic loss surrogate: `grad = 2 * loss * w`, then `w -= grad * learning_rate`. Note this is not the true MSE gradient (it ignores the input), so it is a teaching scaffold rather than a correct optimizer.
- **`Layer`** — Constructor-only stub. Holds `neurons`, `inputs`, `parameters`, `bias`, `size`, but the body is a `pass`-like loop (`for id in self.size: pass`). No forward/backward logic yet.
- **`Network`** — `pass`. Reserved for the multi-layer composition that will wire `Layer`s together.

## Things to know before editing

- **Execution side effects:** Lines 69–70 instantiate and train a `Neuron` at module import. Any import of this file (including from tests) will run the training demo and print to stdout. Wrap the demo in `if __name__ == "__main__":` if you add tests or import this module elsewhere.
- **Builtin shadowing:** `Neuron.train` uses `for input in inputs:` (line 17), shadowing the `input` builtin. Be careful when refactoring this loop.
- **Sparse error handling:** `train` raises a bare `Exception` after printing when input lengths are inconsistent or weight count mismatches input feature count.
- **Verbose logging:** `train` and `adjust_weights` print every iteration (`Grad:`, `Pred:`, `Iteration N: Weights..., Error: ...`). This is intentional for the teaching purpose — silence it before using this code as a library.
- **Gradient is approximate:** `adjust_weights` computes `grad = 2 * loss * w` rather than the true `d_loss/d_w = -2 * loss * input`. Treat the math as illustrative, not numerically correct.

## Extending the code

The intended direction is clearly to flesh out `Layer` (vectorized forward pass across `len(self.size)` neurons) and `Network` (stacking `Layer`s). When doing so, decide whether to keep the per-iteration `print` style or move to a logger, and whether to keep weights as Python lists or switch to NumPy (the venv currently has no NumPy installed).
