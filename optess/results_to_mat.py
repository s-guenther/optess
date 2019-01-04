#!/usr/bin/env python3

import numpy as np
import os
import scipy.io as scio
import sys

import optess as oe


def meta_to_dict(hyb):
    meta = dict()
    meta['objective_type'] = hyb.objective.type.name
    meta['objective_val'] = np.array(hyb.objective.val)
    meta['storage_power'] = np.array(hyb.storage.power)
    meta['storage_energy'] = np.array(hyb.energycapacity)
    meta['storage_efficiency'] = np.array(hyb.storage.efficiency)
    meta['storage_selfdischarge'] = hyb.storage.selfdischarge
    meta['time'] = hyb.signal.times
    meta['signal'] = hyb.signal.vals
    return meta


def single_to_dict(hyb):
    singledata = hyb.single.load_all_results()
    single = dict()
    single['time'] = singledata.power.times
    single['power'] = singledata.power.vals
    single['inner'] = singledata.inner.vals
    single['energy'] = singledata.energy.vals
    single['energycap'] = singledata.energycapacity
    single['energyinit'] = singledata.energyinit
    return single


def hybrid_to_dict(hybres):
    hybdata = hybres.load_all_results()
    hybrid = dict()
    hybrid['pcut'] = hybres.basenorm.power
    hybrid['ecut'] = hybres.basenorm.energy
    hybrid['time'] = hybdata.base.times
    hybrid['base_power'] = hybdata.base.vals
    hybrid['peak_power'] = hybdata.peak.vals
    hybrid['base_inner'] = hybdata.baseinner.vals
    hybrid['peak_inner'] = hybdata.peakinner.vals
    hybrid['base_energy'] = hybdata.baseenergy.vals
    hybrid['peak_energy'] = hybdata.peakenergy.vals
    hybrid['inter'] = hybdata.inter.vals
    hybrid['base_energycap'] = hybdata.baseenergycapacity
    hybrid['peak_energycap'] = hybdata.peakenergycapacity
    hybrid['base_energyinit'] = hybdata.baseenergyinit
    hybrid['peak_energyinit'] = hybdata.peakenergyinit
    return hybrid


def results_to_mat(hyb, outpath):
    if not outpath:
        outpath = os.path.join(os.getcwd(), 'to_mat')
    try:
        os.mkdir(outpath)
    except FileExistsError:
        pass

    meta = meta_to_dict(hyb)
    scio.savemat(os.path.join(outpath, 'meta'), meta)

    single = single_to_dict(hyb)
    scio.savemat(os.path.join(outpath, 'single'), single)

    for cut, data in hyb.inter.items():
        hybrid = hybrid_to_dict(data)
        outfilepath = os.path.join(outpath, 'inter_{}'.format(cut))
        scio.savemat(outfilepath, hybrid)
    for cut, data in hyb.nointer.items():
        hybrid = hybrid_to_dict(data)
        outfilepath = os.path.join(outpath, 'nointer_{}'.format(cut))
        scio.savemat(outfilepath, hybrid)
    for point, data in hyb.area.items():
        hybrid = hybrid_to_dict(data)
        outfilepath = os.path.join(outpath, 'area_{}_[}'.format(*point))
        scio.savemat(outfilepath, hybrid)


if __name__ == '__main__':
    ALLARGS = sys.argv
    try:
        ARGS = sys.argv[1:]
        HYB = oe.HybridDia.load(ARGS[0])
        ARGS[0] = HYB
    except IndexError:
        print('Not enough parameters passed')
    results_to_mat(*ARGS)
