import numpy as np
import numpy


####################################################################
def paint(pos, mesh, weights=1.0, mode="raise", period=None, transform=None):
    """ CIC approximation (trilinear), painting points to Nmesh,
        each point has a weight given by weights.
        This does not give density.
        pos is supposed to be row vectors. aka for 3d input
        pos.shape is (?, 3).

        pos[:, i] should have been normalized in the range of [ 0,  mesh.shape[i] )

        thus z is the fast moving index

        mode can be :
            "raise" : raise exceptions if a particle is painted
             outside the mesh
            "ignore": ignore particle contribution outside of the mesh
        period can be a scalar or of length len(mesh.shape). if period is given
        the particles are wrapped by the period.

        transform is a function that transforms pos to mesh units:
        transform(pos[:, 3]) -> meshpos[:, 3]
    """
    pos = numpy.array(pos)
    chunksize = 1024 * 16 * 4
    Ndim = pos.shape[-1]
    Np = pos.shape[0]

    if transform is None:
        transform = lambda x:x
    neighbours = ((numpy.arange(2 ** Ndim)[:, None] >> \
            numpy.arange(Ndim)[None, :]) & 1)
    for start in range(0, Np, chunksize):
        chunk = slice(start, start+chunksize)
        if numpy.isscalar(weights):
          wchunk = weights
        else:
          wchunk = weights[chunk]
        gridpos = transform(pos[chunk])
        rmi_mode = 'raise'
        intpos = numpy.intp(numpy.floor(gridpos))

        for i, neighbour in enumerate(neighbours):
            neighbour = neighbour[None, :]
            targetpos = intpos + neighbour

            kernel = (1.0 - numpy.abs(gridpos - targetpos)).prod(axis=-1)
            add = wchunk * kernel

            if period is not None:
                period = numpy.int32(period)
                numpy.remainder(targetpos, period, targetpos)

            if len(targetpos) > 0:
                targetindex = numpy.ravel_multi_index(
                        targetpos.T, mesh.shape, mode=rmi_mode)
                u, label = numpy.unique(targetindex, return_inverse=True)
                mesh.flat[u] += numpy.bincount(label, add, minlength=len(u))

    return mesh

def paintcic(pos, bs, nc, mass=1.0, period=True):
    mesh = np.zeros((nc, nc, nc))
    transform = lambda x: x/bs*nc
    if period: period = int(nc)
    else: period = None
    return paint(pos, mesh, weights=mass, transform=transform, period=period)

def paintnn(pos, bs, nc, mass=1.0, period=True, shift=True):
    if type(mass) !=  np.ndarray : mass = np.ones(pos.shape[0])
    bins = np.arange(0, bs+bs/nc, bs/nc)
    if shift: 
        posshift = pos + 0.5*bs/nc
        np.remainder(posshift, bs, posshift)
        mesh = np.histogramdd(posshift, bins = (bins, bins, bins) , weights=mass)
    else: mesh = np.histogramdd(pos, bins = (bins, bins, bins) , weights=mass)
    return mesh[0]


#########################################################################################


def readhead(path):
    shape = None
    with open(path +'attr-v2') as  f:
        for line in f.readlines():
            if 'ndarray.shape' in line: 
                shape = tuple(int(i) for i in line.split('[')[1].split()[:-1])
    with open(path +'header') as  f:
        for line in f.readlines():
            if 'DTYPE' in line: dtype = line.split()[-1]
            if 'NFILE' in line: nf = int(line.split()[-1])
            if 'NMEMB' in line: 
                if shape is None: shape = tuple([-1, int(line.split()[-1])])
    return dtype, nf, shape


def readbigfile(path):
    dtype, nf, shape = readhead(path)
    data = []
    for i in range(nf): data.append(np.fromfile(path + '%06d'%i, dtype=dtype))
    data = np.concatenate(data)
    data = np.reshape(data, shape)
    return data




#########################################################################################
def fftk(shape, boxsize, symmetric=True, finite=False, dtype=np.float64):
    """ return kvector given a shape (nc, nc, nc) and boxsize 
    """
    k = []
    for d in range(len(shape)):
        kd = numpy.fft.fftfreq(shape[d])
        kd *= 2 * numpy.pi / boxsize * shape[d]
        kdshape = numpy.ones(len(shape), dtype='int')
        if symmetric and d == len(shape) -1:
            kd = kd[:shape[d]//2 + 1]
        kdshape[d] = len(kd)
        kd = kd.reshape(kdshape)

        k.append(kd.astype(dtype))
    del kd, kdshape
    return k




def laplace(bs=None, nc=None, kvec=None, symmetric=True):
    if kvec is None:
        if nc is None or bs is None:
            print('Need either a k vector or bs & nc')
            return None
        else: kvec = fftk((nc, nc, nc), bs, symmetric=symmetric)

    kk = sum(ki**2 for ki in kvec)
    mask = (kk == 0).nonzero()
    kk[mask] = 1
    wts = 1/kk
    imask = (~(kk==0)).astype(int)
    wts *= imask
    return wts



def gradient(dir, bs, nc, kvec=None, finite=True, symmetric=True):
    if kvec is None:
        kvec = fftk((nc, nc, nc), bs, symmetric=symmetric)

    cellsize = bs/nc
    w = kvec[dir] * cellsize
    if finite:
        a = 1 / (6.0 * cellsize) * (8 * numpy.sin(w) - numpy.sin(2 * w))
        wts = a*1j
    else: wts = 1j*kvec[dir]
    return wts


def potential(mesh, k, symmetric=True):
    if abs(mesh.mean()) > 1e-3:
        ovd = (mesh-mesh.mean())/mesh.mean()
    else: ovd = mesh.copy()
    lap = laplace(kvec = k, symmetric=symmetric)

    if not symmetric:
        ovdc = np.fft.fftn(ovd)/np.prod(mesh.shape)
        potc = ovdc*lap
        pot =  np.fft.ifftn(potc)*np.prod(mesh.shape)
    else:
        ovdc = np.fft.rfftn(ovd)/np.prod(mesh.shape)
        potc = ovdc*lap
        pot =  np.fft.irfftn(potc)*np.prod(mesh.shape)
    return pot
    



def gauss(mesh, k, R):
    kmesh = sum([i ** 2 for i in k])**0.5
    meshc = np.fft.rfftn(mesh)/np.prod(mesh.shape)
    wts = np.exp(-0.5*kmesh**2*(R**2))
    meshc = meshc*wts
    return np.fft.irfftn(meshc)*np.prod(mesh.shape)


def fingauss(mesh, k, R, kny):
    kmesh = sum(((2*kny/np.pi)*np.sin(ki*np.pi/(2*kny)))**2  for ki in k)**0.5
    meshc = np.fft.rfftn(mesh)/np.prod(mesh.shape)
    wts = np.exp(-0.5*kmesh**2*(R**2))
    meshc = meshc*wts
    return np.fft.irfftn(meshc)*np.prod(mesh.shape)



def tophat(mesh, k, R):
    kmesh = sum([i ** 2 for i in k])**0.5
    meshc = np.fft.rfftn(mesh)/np.prod(mesh.shape)
    kr = R * kmesh
    kr[kr==0] = 1
    wt = 3 * (np.sin(kr)/kr - np.cos(kr))/kr**2
    wt[kr==0] = 1        
    meshc = meshc*wt
    return np.fft.irfftn(meshc)*np.prod(mesh.shape)



def decic(mesh, k, kny, n=2):
    kmesh = [np.sinc(k[i]/(2*kny)) for i in range(3)]
    wts = (kmesh[0]*kmesh[1]*kmesh[2])**(-1*n)
        
    meshc = np.fft.rfftn(mesh)/np.prod(mesh.shape)
    meshc = meshc*wts
    return np.fft.irfftn(meshc)*np.prod(mesh.shape)



def tophatfunction(k, R):
    '''Takes in k, R scalar to return tophat window for the tuple'''
    kr = k*R
    wt = 3 * (np.sin(kr)/kr - np.cos(kr))/kr**2
    if wt is 0:
        wt = 1
    return wt

def gaussfunction(k, R):
    '''Takes in k, R scalar to return gauss window for the tuple'''
    kr = k*R
    wt = np.exp(-0.5*(kr**2))
    return wt


def fingaussfunction(k, kny, R):
    '''Takes in k, R and kny to do Gaussian smoothing corresponding to finite grid with kny'''
    kf = np.sin(k*np.pi/kny/2.)*kny*2/np.pi
    return np.exp(-(kf**2 * R**2) /2.)

def guassdiff(pm, R1, R2):
    pass


def diracdelta(i, j):
    if i == j: return 1
    else: return 0


def shear(mesh, k):
    '''Takes in a PMesh object in real space. Returns am array of shear'''          
    k2 = sum([i ** 2 for i in k])
    k2[0, 0, 0] = 1
    meshc = np.fft.rfftn(mesh)/np.prod(mesh.shape)
    s2 = np.zeros_like(mesh)

    for i in range(3):
        for j in range(i, 3):                                                       
            basec = meshc.copy()
            basec *= (k[i]*k[j] / k2 - diracdelta(i, j)/3.)              
            baser = np.fft.irfftn(basec)*np.prod(mesh.shape)                                                                
            s2[...] += baser**2                                                        
            if i != j:                                                              
                s2[...] += baser**2                                                
    return s2  




#################################################################################


def power(f1, f2=None, boxsize=1.0, k = None, symmetric=True, demean=True, eps=1e-9):
    """
    Calculate power spectrum given density field in real space & boxsize.
    Divide by mean, so mean should be non-zero
    """
    if demean and abs(f1.mean()) < 1e-3:
        print('Add 1 to get nonzero mean of %0.3e'%f1.mean())
        f1 = f1*1 + 1
    if demean and f2 is not None:
        if abs(f2.mean()) < 1e-3:
            print('Add 1 to get nonzero mean of %0.3e'%f2.mean())
            f2 =f2*1 + 1
    
    if symmetric: c1 = numpy.fft.rfftn(f1)
    else: c1 = numpy.fft.fftn(f1)
    if demean : c1 /= c1[0, 0, 0].real
    c1[0, 0, 0] = 0
    if f2 is not None:
        if symmetric: c2 = numpy.fft.rfftn(f2)
        else: c2 = numpy.fft.fftn(f2)
        if demean : c2 /= c2[0, 0, 0].real
        c2[0, 0, 0] = 0
    else:
        c2 = c1
    #x = (c1 * c2.conjugate()).real
    x = c1.real* c2.real + c1.imag*c2.imag
    del c1
    del c2
    if k is None:
        k = fftk(f1.shape, boxsize, symmetric=symmetric)
        k = sum(kk**2 for kk in k)**0.5
    H, edges = numpy.histogram(k.flat, weights=x.flat, bins=f1.shape[0]) 
    N, edges = numpy.histogram(k.flat, bins=edges)
    center= edges[1:] + edges[:-1]
    power = H *boxsize**3 / N
    power[power == 0] = np.NaN
    return 0.5 * center,  power


#################################################################################


def get_ps(iterand, truth, bs):

    ic1, fin1 = iterand
    ic2, fin2 = truth

    pks = []
    #if abs(ic1.mean()) < 1e-3: ic1 += 1
    #if abs(ic.mean()) < 1e-3: ic += 1                                                                                                                  
    k, p1 = power(ic1+1, boxsize=bs)
    k, p2 = power(ic2+1, boxsize=bs)
    k, p12 = power(ic1+1, f2=ic2+1, boxsize=bs)
    pks.append([p1, p2, p12])
    if fin1.mean() < 1e-3: fin1 += 1
    if fin2.mean() < 1e-3: fin2 += 1
    k, p1 = power(fin1, boxsize=bs)
    k, p2 = power(fin2, boxsize=bs)
    k, p12 = power(fin1, f2=fin2, boxsize=bs)
    pks.append([p1, p2, p12])

    return k, pks
