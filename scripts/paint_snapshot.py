import numpy as np
import tools
import readgadget
from pmesh import ParticleMesh as pmnew
from nbodykit.lab import FFTPower
import sys, os


##Setup Mesh 
bs = 1000 #BoxSize
nc = 256 #Nmesh
kvec = tools.fftk([nc]*3, bs)
mesh = pmnew(Nmesh=[nc]*3, BoxSize=bs)
cic_kwts = [np.sinc(kvec[i] * bs / (2 * np.pi * nc)) for i in range(3)]
cic_kwts = (cic_kwts[0] * cic_kwts[1] * cic_kwts[2])**(-2)


def cic_compensation(field, kernel=cic_kwts):
      """
      Does cic compensation with kernel for the field.
      Adapted from https://github.com/bccp/nbodykit/blob/a387cf429d8cb4a07bb19e3b4325ffdf279a131e/nbodykit/source/mesh/catalog.py#L499
      Itself based on equation 18 (with p=2) of
      `Jing et al 2005 <https://arxiv.org/abs/astro-ph/0409240>`_
      Args:
      kvec: array of k values in Fourier space  
      Returns:
      field_comp: compensated field
      """
      cfield = field.r2c() #np.fft.rfftn(field)
      cfield *= kernel
      field_comp = cfield.c2r() #np.fft.irfftn(cfield)
      return field_comp


#Setup Quijote
savefolder = "/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N%04d/"%nc

path = "/mnt/home/fvillaescusa/ceph/Quijote/Snapshots/latin_hypercube_HR/%d/snapdir_004/snap_004"
ptype = [1]
col = "POS "

idd = 0
for idd in range(2000):
      print(idd)
      savepath = savefolder + '%04d/'%idd 
      os.makedirs(savepath, exist_ok=True)
      snapshot = path%idd
      pos = readgadget.read_block(snapshot, col, ptype)/1e3
      dm = mesh.paint(pos)
      dm_comp = cic_compensation(dm)
      np.save(savepath + 'field', dm_comp)
      
      ps = FFTPower(dm_comp, mode='1d').power.data
      k, p = ps['k'], ps['power'].real
      np.save(savepath + 'power', np.stack([k, p]).T)
      del dm, dm_comp
