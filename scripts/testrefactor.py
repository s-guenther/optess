import optess as oe

sig = oe.DataFactory.rand(npoints=32, seed=123)
obj = oe.Objective('power', 12)
stor = oe.Storage(max(sig.vals) - 12, 0.95, 1e6)
hyb = oe.HybridDia(sig, stor, obj)
hyb.compute_parallel(workers=4)
hyb.pplot()
dummybreakpoint = True
