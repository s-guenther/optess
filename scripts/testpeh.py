import optess as oe

npoints = 4096
signal = oe.DataFactory.rand(npoints, seed=npoints)

peakcut = 10.5
storpower = max(signal.vals) - peakcut
objective = oe.Target('power', peakcut)
storage = oe.Storage(storpower, 0.95, 1e50)

opt = oe.OptimizeSingleESS(signal, storage, objective)
opt.solve_pyomo_model()

opt.pplot()

power = opt.results.power
energy = opt.results.energy

peh = oe.PEHMap(power, energy)
peh.pplot()
peh2 = peh.rebin((10, 4))
peh2.pplot()

dummybreakpoint = True
