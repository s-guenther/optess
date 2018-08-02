OPTESS
============

_The development of this toolbox is still in early stage. It is not intended
for productive use at the moment._

OPTESS is a framework around a (MI)LP Optimization to dimension an energy
storage system or hybrid energy storage system. It builds the model with the
help of pyomo and provides routines and classes for pre- and postprocessing.
Exemplarily, the toolbox can take a load profile of a factory and calculate
minimal storage (regarding energy, for a fixed power) to perform a specified
power cut. Or, it can take the load profile of a photo voltaik plant with some
consumers in a microgrid and calculate minimal storage to achieve a certain
throughput and self consumption rate. The model assumes an idealised storage
subject to efficiency and self discharge losses and not specific storage
technologies with a fixed power to energy ratio.

Requirements
------------

Python 3.4 or later

Additional Python Packages:
- pyomo
- numpy
- scipy
- matplotlib
- overload


Installation
------------

_Information below is outdated._

Download the source code and add it to the python search path. Install the
missing packages (via pip). Install and sign up for optional solvers (e.g.
gurobi or glpk)


Getting Started
---------------

_Information below is outdated._

Program flow is centered around two main classes, `OptimizeSingleEES` and
`OptimizeHybridEES`, depending on whether a single energy storage shall be
optimized or a hybrid energy storage (The latter one does not work at the
moment). These classes gather inputs, delegate calculation and gather outputs,
or in other words, administrate the whole calculation process with pre- and
postprocessing.

An optimization setting is defined by:
- a load profile (`class Signal`)
- one or two storages (`class Storage`)
- an optimization aim (`class Objective`)
- in case of HESS: a strategy or boundary for control (`class Strategy`)
- A solver (`class Solver`)

For all inputs, factories are prepared to easily get started.

```python
    import factories

    signal = factories.datafactory('alt')
    storage = factories.storagefactory('2.low')
    objective = factories.objectivefactory('std0-3')
    solver = 'glpk'
```

Then, the optimization object can be initialized:

```python
    from optimize_ess import OptimizeSingleESS

    optim = OptimizeSingleESS(signal, storage, objective, solver)
```

To get the results, simply call it as a property

```python
    res = optim.results
```

An alternative way to set up an optimization is the usage of a factory for the
complete setting:

```python
    opt_setup = factories.singlesetupfactory('alt.low', '2')
```

Which returns a tuple which can be directly unpacked into `OptimizeSingleESS`:

```python
    optim = OptimizeSingleESS(*opt_setup)
```

All objects provide `pprint()` and `pplot()` functions to easily analzye,
visualize and show the different objects, e.g.

```python
    optim.pplot()
    optim.signal.pprint()
    optim.results.pplot()
```


Known Issues
------------

_Information below is outdated._

Hybrid EES Optimization is virtually useless at the moment as the model is
erroneous.


Todo
----

- Add plotting and printing capabilities to:
    - abstractoptimees
    - storage
    - signal
    - objective
    - results (single, hybrid)
- Add Docstrings
- Rework Factories
- Fill Error Classes with code
- Debug HybridBuilder
- Validity Checking for PyomoResult
- Objective.validate(Signal)

License
-------

This software is licensed under GPLv3, excluding later versions.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

For details see the license file [\$OPTMIZE-EES/LICENSE](LICENSE).

GPLv3 explicitely allows a commercial usage without any royalty or further
implications. However, any contact to discuss possible cooperations is
appreciated.


Author
------

optimize-ees - MILP optimization to find minimal (hybrid) energy storage\
Copyright (C) 2018\
Sebastian Günther\
sebastian.guenther@ifes.uni-hannover.de

Institut für Elektrische Energiesysteme\
Fachgebiet für Elektrische Energiespeichersysteme

Institute of Electric Power Systems\
Electric Energy Storage Systems Section

https://www.ifes.uni-hannover.de/ees.html
