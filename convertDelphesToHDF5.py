#!/usr/bin/env python

import sys
import uproot
import numpy as np
import h5py
import progressbar
import os

widgets=[
    progressbar.SimpleProgress(), ' - ', progressbar.Timer(), ' - ', progressbar.Bar(), ' - ', progressbar.AbsoluteETA()
]

def deltaR(eta1, phi1, eta2, phi2):
    """ calculate deltaR """
    dphi = (phi1-phi2)
    while dphi >  np.pi: dphi -= 2*np.pi
    while dphi < -np.pi: dphi += 2*np.pi
    deta = eta1-eta2
    return np.hypot(deta, dphi)


import optparse

#configuration
usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option('-i', '--input', dest='input', help='input file', default='', type='string')
parser.add_option('-o', '--output', dest='output', help='output file', default='', type='string')
parser.add_option("-N", "--maxevents", dest='maxevents', help='max number of events', default=-1, type='int')
parser.add_option("--data", dest="data", action="store_true", default=False, help="input is data. The default is MC")
(opt, args) = parser.parse_args()

if opt.input == '' or opt.output == '':
    sys.exit('Need to specify input and output files!')

varList = [
    'PuppiParticles_size', 'PuppiParticles.PT', 'PuppiParticles.Eta', 'PuppiParticles.Phi', 'PuppiParticles.PID',
    'PuppiParticles.Charge', 'PuppiParticles.Mass',
    'PuppiParticles.D0', 'PuppiParticles.DZ', 'PuppiParticles.VertexIndex', 'PuppiParticles.PuppiWeight',
]

# event-level variables
varList_evt = [
    'Rho.Rho', 'RecoVertex_size',
]

varList_mc = [
    'GenMissingET.MET', 'GenMissingET.Phi',
]

d_encoding = {
    b'PuppiParticles.Charge':{-1: 0, 0: 1, 1: 2},
    b'PuppiParticles.PID':{
        # -211: 0,
        -13: 1,
        -11: 2,
        0: 3,
        11: 4,
        13: 5,
        22: 6,
        130: 7,
        # 211: 8
    }, # Delphes has more PIDs than PF
    # b'PuppiParticles.VertexIndex':{0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3} # Delphes has associated vertex ID
}
# add positive PF "pions"
d_encoding[b'PuppiParticles.PID'].update(dict.fromkeys([211, 321, 2212, 3222, -3112, -3312, -3334], 8))
# add negative PF "pions" from previous list
antiParticles = {}
for key,val in d_encoding[b'PuppiParticles.PID'].items():
    if val == 8:
        antiParticles[-key] = 0
d_encoding[b'PuppiParticles.PID'].update(antiParticles)

if not opt.data:
    varList = varList + varList_mc
varList = varList + varList_evt

upfile = uproot.open(opt.input)
tree = upfile['Delphes'].arrays( varList )

# general setup
maxNPF = 4500
nFeatures = 14

maxEntries = len(tree[b'PuppiParticles_size']) if opt.maxevents==-1 else opt.maxevents
# input PF candidates
X = np.zeros(shape=(maxEntries,maxNPF,nFeatures), dtype=float, order='F')
# recoil estimators
Y = np.zeros(shape=(maxEntries,2), dtype=float, order='F')
# leptons 
XLep = np.zeros(shape=(maxEntries, 2, nFeatures), dtype=float, order='F')
# event-level information
EVT = np.zeros(shape=(maxEntries,len(varList_evt)+2), dtype=float, order='F')

print(X.shape)

# loop over events
for e in progressbar.progressbar(range(maxEntries), widgets=widgets):
    # get momenta
    ipf = 0
    ilep = 0
    for j in range(tree[b'PuppiParticles_size'][e]):
        if ipf == maxNPF:
            break

        pt = tree[b'PuppiParticles.PT'][e][j]
        #if pt < 0.5:
        #    continue
        eta = tree[b'PuppiParticles.Eta'][e][j]
        phi = tree[b'PuppiParticles.Phi'][e][j]

        pf = X[e][ipf]

        ipf += 1

        # 4-momentum
        pf[0] = pt
        pf[1] = pt * np.cos(phi)
        pf[2] = pt * np.sin(phi)
        pf[3] = eta
        pf[4] = phi
        pf[5] = tree[b'PuppiParticles.D0'][e][j]
        pf[6] = tree[b'PuppiParticles.DZ'][e][j]
        pf[7] = tree[b'PuppiParticles.PuppiWeight'][e][j]
        pf[8] = tree[b'PuppiParticles.Mass'][e][j]
        pf[9] = 0
        pf[10] = 0
        # encoding
        pf[11]  = d_encoding[b'PuppiParticles.PID'][tree[b'PuppiParticles.PID'][e][j]]
        pf[12] = d_encoding[b'PuppiParticles.Charge'][tree[b'PuppiParticles.Charge'][e][j]]
        pf[13] = 1. if tree[b'PuppiParticles.VertexIndex'][e][j] == 0 else 0.
        # set pion and kaon masses for charged/neutral hadrons
        if pf[11] in [0, 8]:
            pf[8] = 0.140
        elif pf[11] == 0:
            pf[8] = 0.498

    # truth info
    Y[e][0] += tree[b'GenMissingET.MET'][e] * np.cos(tree[b'GenMissingET.Phi'][e])
    Y[e][1] += tree[b'GenMissingET.MET'][e] * np.sin(tree[b'GenMissingET.Phi'][e])

    EVT[e][0] = tree[b'Rho.Rho'][e][0] # rho eta [-5, -2.5]
    EVT[e][1] = tree[b'Rho.Rho'][e][1] # rho eta [-2.5, 2.5]
    EVT[e][2] = tree[b'Rho.Rho'][e][2] # rho eta [2.5, 5]
    EVT[e][3] = tree[b'RecoVertex_size'][e]

with h5py.File(opt.output, 'w') as h5f:
    h5f.create_dataset('X',    data=X,   compression='lzf')
    h5f.create_dataset('Y',    data=Y,   compression='lzf')
    h5f.create_dataset('EVT',  data=EVT, compression='lzf')
