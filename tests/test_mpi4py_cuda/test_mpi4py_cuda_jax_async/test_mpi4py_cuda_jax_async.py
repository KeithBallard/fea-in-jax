from mpi4py import MPI
import jax
import jax.numpy as jnp
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Ensure GPU is available in JAX
if not jax.devices('gpu'):
    raise RuntimeError('No GPU available for JAX!')

# 2D halo exchange between two adjacent cells
#
# z--y
# |
# x
#
# data:
# [ 0, 1, 2, 3]
# [ 4, 5, 6, 7]
# [ 8, 9,10,11]
# [12,13,14,15]
#
# ranks:
# 0 2
# 1 3
#
# decomp:
# [ 0, 1] [ 2, 3]
# [ 4, 5] [ 6, 7]
#
# [ 8, 9] [10,11]
# [12,13] [14,15]
#
# with halo:
# [ _, _, _, _] [ _, _, _, _]
# [ _, 0, 1, _] [ _, 2, 3, _]
# [ _, 4, 5, _] [ _, 6, 7, _]
# [ _, _, _, _] [ _, _, _, _]
#               
# [ _, _, _, _] [ _, _, _, _]
# [ _, 8, 9, _] [ _,10,11, _]
# [ _,12,13, _] [ _,14,15, _]
# [ _, _, _, _] [ _, _, _, _]
#
# axes:
# +x,-x,+y,-y,+z,-z
#
# NOTE: assuming orthorhombic decomposition
# NOTE: halo size for each axis must be less than the length of that axis

# the tilesize returned here is fractional
def get_tiling(n_ranks, shape, mask=(1,1,1)):
    mask = tuple(int(bool(x)) for x in mask)
    assert len(shape) == len(mask)
    n_dim = sum(mask)
    if n_ranks > n_dim and n_ranks >= shape[0]:
        q,r = divmod(n_ranks, n_dim)
        factors = [q]*n_dim
        for i in range(r):
            factors[i] += 1
        c = 0
        tiling = []
        tilesize = []
        for m,s in zip(mask, shape):
            if m:
                tiling.append(factors[c])
                tilesize.append(s/factors[c])
                c+=1
            else:
                tiling.append(1)
                tilesize.append(s)
    else:
        tiling = (n_ranks, 1, 1)
        tilesize = (shape[0]/n_ranks, shape[1], shape[2]) # 1D
    #tilesize = [int(np.floor(x)) for x in tilesize]
    return tuple(tiling), tuple(tilesize)

def get_topology(n_ranks, tiling):
    assert np.prod(tiling) == n_ranks
    return np.arange(n_ranks).reshape(tiling, order='F')

def get_rank_index(rank, topology):
    return tuple(int(x[0]) for x in np.where(topology==rank))

def get_shardsize(tiling, tilesize, ijk):
    shardsize=[]
    for axis,(idx,v) in enumerate(zip(ijk, tilesize)):
        f = np.ceil 
        if idx+1 == tiling[axis]:
            shardsize.append(int(np.ceil(v)))
        else:
            shardsize.append(int(np.floor(v)))
    return tuple(shardsize)

def domain_decomposition(domain):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # broadcast shape
    shape = None
    if rank == 0:
        shape = domain.shape
    shape = comm.bcast(shape, root=0)
    
    # all ranks need to know the topology
    tiling, tilesize = get_tiling(size, shape, mask=(1,1,0))
    topology  = get_topology(size, tiling)
    ijk       = get_rank_index(rank, topology)
    shardsize = get_shardsize(tiling, tilesize, ijk)

    # shard the data from rank 0
    if rank == 0:
        tilesize = tuple([int(np.floor(x)) for x in tilesize])
        requests = []
        for dst in range(size):
            idx = []
            for axis,i in enumerate(get_rank_index(dst, topology)):
              x0 = i*tilesize[axis]
              x1 = x0 + tilesize[axis]
              if ijk[axis]+1 == tiling[axis]:
                  x1 = None
              idx.append(slice(x0,x1))
            idx = tuple(idx)
            if dst == 0:
                shard = domain[idx]
            else:
                req = comm.isend(domain[idx], dest=dst, tag=0)
                requests.append(req)
        MPI.Request.Waitall(requests)
    else:
        shard = comm.recv(source=0, tag=0)
    return shard, topology

def init_halo(data, halosize=(0,0,0)):
    hx,hy,hz = halosize
    return jnp.pad(data, ((hx,hx),(hy,hy),(hz,hz)))

def get_data_without_halo(data, halosize=(0,0,0)):
    hx,hy,hz = halosize
    idx = (
        slice(hx,-hx) if hx>0 else slice(None),
        slice(hy,-hy) if hy>0 else slice(None),
        slice(hz,-hz) if hz>0 else slice(None),
    )
    return data[idx]

def get_shape_without_halo(data, halosize=(0,0,0)):
    sx,sy,sz = data.shape
    hx,hy,hz = halosize
    return (sx-(2*hx),sy-(2*hy),sz-(2*hz))

# +x, -x, +y, -y, +z, -z,
# NOTE: this is an incomplete example since it is missing diagonals
# that's okay for now since this is just a test of whether 
# async point to point on the GPU works properly
# +x+y +x-y -x+y -x-y
# +y+z +y-z -y+z -y-z
# +z+x +z-x -z+x -z-x
def init_neighbor_list(topology):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ijk  = get_rank_index(rank, topology)
    neigh = []
    for axis,(n,i) in enumerate(zip(topology.shape, ijk)):
        for sign in [+1, -1]:
            if i+sign >= 0 and i+sign < n:
                idx = [x for x in ijk]
                idx[axis] = idx[axis]+sign
                neigh.append(int(topology[tuple(idx)]))
            else:
                neigh.append(None)
    return tuple(neigh)
    



# init data on rank 0
domain = None
if rank == 0:
    domain = jnp.arange(16).reshape((4,4,1))
    print(f'\nRank 0 original data:\n{domain[:,:,0]}')

# shard data among ranks
data, topology = domain_decomposition(domain)
if rank == 0:
    print(f'\nRank topology:\n{topology[:,:,0]}')

# create halos to store neighbor boundaries
halosize=(1,1,0)
shard = init_halo(data, halosize)

# create the neighbor list
neigh = init_neighbor_list(topology)

# halo exchange
requests = []
for axis_id,dst in enumerate(neigh):
    if dst is None:
        continue
    axis = (axis_id//2)
    sign = (axis_id%2)
    halo = halosize[axis]
    data = get_data_without_halo(shard, halosize)
    
    # select the positive or negative perpendicular plane
    i = slice(
        -halo if sign==0 else None,
         halo if sign==1 else None
    )

    # select the axis
    idx = (
        i if axis==0 else slice(None),
        i if axis==1 else slice(None),
        i if axis==2 else slice(None)
    )
    
    # set the tag to the axis_id of the reciever (same axis, but flipped sign)
    # invert sign on same axis
    tag = (axis*2)+(sign^1)

    # send 2D slice
    buf = data[idx]
    req = comm.isend(buf, dest=dst, tag=tag)
    requests.append(req)

# recv
n_recv = 0
n_neigh = sum([1 for _ in neigh if _ is not None])
while n_recv < n_neigh:
    status = MPI.Status()
    comm.probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
    src = status.Get_source()
    axis_id = status.Get_tag()
    axis = (axis_id//2)
    sign = (axis_id%2)
    shape = get_shape_without_halo(shard, halosize)
    
    buf = comm.recv(source=src, tag=axis_id)
    n_recv += 1

    # get the slice indices of where the data should go
    idx = []
    for _axis,(h,l) in enumerate(zip(halosize,shape)):
        if _axis == axis:
            if sign == 0:
                idx.append(slice(-h,None))
            elif sign == 1:
                idx.append(slice(0,h))
        else:
            idx.append(slice(h,l+h))
    idx = tuple(idx)

    # update the local data
    shard = shard.at[idx].set(buf)

MPI.Request.Waitall(requests)

# print output
for r in range(size):
    if rank == r:
        print(f'\nRank {rank} now has:\n{shard[:,:,0]}')
    comm.barrier()

