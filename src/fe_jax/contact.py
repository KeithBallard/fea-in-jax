import jax
from jax import numpy as jnp

def contact(x_nd,y_nd,float: radius):
    Nx = len(x_nd)
    Ny = len(y_nd) 
    contacts = []
    for i in range(Nx):
        for j in range(Ny):
            if jnp.linalg.norm(X_nd[i,:]-Y_nd[j,:])<=radius:
                contacts.append(X_nd[i,:],Y_nd[j,:])

def contact_batch(list: X_fnd,float: radius):
    # I am using m for number of fibers
    f = len(X_fnd)
    contacts = []
    for i in range(f):
        for j in range(i+1,f):
            contacts.append(contact(X_fnd[i,:],Y_fnd[j,:]))
