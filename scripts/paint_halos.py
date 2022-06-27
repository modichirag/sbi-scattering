import numpy as np
import tools
import readgadget, readfof
from pmesh import ParticleMesh as pmnew
from nbodykit.lab import FFTPower
import sys, os

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--id0', type=int, help='sim number to start painting from')
parser.add_argument('--id1', type=int, default=2000, help='sim number to paint upto')
parser.add_argument('--z', type=float, help='redshift')
args = parser.parse_args()

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


###Setup Quijote
#snapnum = 2 #4
redshift = float("%0.1f"%args.z)
snapnum = int(np.where(np.array([3.0, 2.0, 1.0, 0.5, 0.0]) == redshift)[0])
print(redshift, snapnum)
savefolder = "/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/FoF/N%04d/z%s/"%(nc, str(redshift))
path = '/mnt/ceph/users/fvillaescusa/Quijote/Halos/FoF/latin_hypercube/HR_%d//' #folder hosting the catalogue
#savefolder = "/mnt/ceph/users/cmodi/Quijote/latin_hypercube_nwLH/FoF/N%04d/"%nc
#path = '/mnt/ceph/users/fvillaescusa/Quijote/Halos/FoF/latin_hypercube_nwLH/%d//' #folder hosting the catalogue


idd = 0

for idd in range(args.id0, args.id1):
      print(idd)
      savepath = savefolder + '%04d/'%idd 
      os.makedirs(savepath, exist_ok=True)
      catalog = path%idd
      try:
            halo_comp = np.load(savepath + 'field.npy')
            halo_comp = mesh.create(mode='real', value=halo_comp)
            pk = np.load(savepath + 'power.npy')
            print("%d exists"%idd)
            raise Exception
      except Exception as e:
            print(e)
            FoF = readfof.FoF_catalog(catalog, snapnum, long_ids=False,
                                swap=False, SFR=False, read_IDs=False)
            pos = FoF.GroupPos/1e3            #Halo positions in Mpc/h
            mass  = FoF.GroupMass*1e10          #Halo masses in Msun/h
            Npart = FoF.GroupLen    
            print("Max and min masses : %0.2e, %0.2e"%(mass.max(), mass.min()))
            print("Total number of halos is : %0.2e"%(mass.size))
            print("Number density : %0.2e"%(mass.size/bs**3))

            #idsort = np.argsort(Npart)[::-1]
            diff = Npart[:-1] - Npart[1:]
            print("negatives : ", (diff < 0).sum(), (diff < 0).sum()/diff.size)
            #
            
            halo = mesh.paint(pos)
            halo_comp = cic_compensation(halo)
            np.save(savepath + 'field', halo_comp)

            halo_comp = halo_comp / halo_comp.cmean() - 1
            ps = FFTPower(halo_comp, mode='1d').power.data
            k, p = ps['k'], ps['power'].real
            np.save(savepath + 'power', np.stack([k, p]).T)
            del halo, halo_comp
                  
            for numd in [1e-3, 5e-4, 1e-4]:
                  num = int(numd * bs**3)
                  print("for number density %0.3e, number of halos is %0.3e"%(numd, num))
                  halo = mesh.paint(pos[:num])
                  halo_comp = cic_compensation(halo)
                  np.save(savepath + 'field_n%0.0e'%numd, halo_comp)

                  halo_comp = halo_comp / halo_comp.cmean() - 1
                  ps = FFTPower(halo_comp, mode='1d').power.data
                  k, p = ps['k'], ps['power'].real
                  np.save(savepath + 'power_n%0.0e'%numd, np.stack([k, p]).T)
                  del halo, halo_comp
            
