# pyoptsparsewrapper

A simplified pyOptSparse interface.  Mainly for students in my optimization class.

The dictionary-based format in pyoptsparse is very convenient when dealing with large problems where the gradients or linear constraints have significant sparsity.  But for smaller problems putting arrays in and out of dictionaries can be cumbersome.

Other additions include:

- combining obj/con and gradient functions without extra function calls.
- making the availability of xStar and fStar clear (as return objects)
- fixing a bug where fStar isn't updated for some optimizers
- fixing a bug in NLPQLP xStar output
- returning number of function calls correctly
- automatically check if gradients are supplied or not
- automatically determining size of design vars, constraints, etc.
- convenient syntax for defining linear constraints.

## Install

Open up terminal (or command prompt) and type:
`python setup.py install` 

(or for a one-off case, just stick the script pyoptwrapper.py in your current directory).

## Syntax and Example

Syntax is similar to Matlab's fmincon, except the linear constraints are optional arguments:

```python
from pyoptwrapper import optimize

xopt, fopt, info = optimize(func, x0, lb, ub, optimizer, A=[], b=[], Aeq=[], beq=[], args=[])
```

Inputs:

- `func`: function handle of function to minimize (see signature below)
- `x0`: starting point (array)
- `lb`: lower bound constraints (array)
- `ub`: upper bound constraints (array)
- `optimizer`: optimizer to use from pyoptsparse
- `A` and `b`: linear inequality constaints of form: A x <= b
- `Aeq` and `beq`: linear equality constaints of form: Aeq x = beq
- `args`: a tuple of extra arguments to pass into func (optional)


Outputs:

- `xopt`: x solution array
- `fopt`: corresponding function value
- `info`: dictionary containing
    - `max-c-vio`: maximum constraint violation
    - `fcalls`: number of calls to func
    - `time`: time spent in optimization
    - `code` an output code, if any, returned by the optimizer

The function func should be of one of the following two forms:
```
[f, c] = func(x)

[f, c, gf, gc] = func(x)
```

- `f` function value (or a function array if a multiobjective problem)
- `c` is an array of constraints of the form c(x) <= 0
- `gf` is an array of the gradients of f (not used in multiobj case)
- `gc` is the gradients of the constraints and is of size (len(c), len(x)).  

The script will automatically detect if gradients are provided or not.  func can accept extra arguments, you pass those in with the args option.

See example.py for examples.
