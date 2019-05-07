import optess as oe

sig = oe.DataFactory.rand(npoints=128, seed=123)
obj = oe.Target('power', 12)
stor = oe.Storage(max(sig.vals) - 12, 0.95, 1e6)
hyb = oe.HybridDia(sig, stor, obj)
hyb.compute_serial(cuts=[0.4, 0.6], curves=2)
hyb.pplot()
dummybreakpoint = True
