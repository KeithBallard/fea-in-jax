from jax import numpy as jnp
from dataclasses import dataclass
from typing import Callable

@dataclass
class ContactPointPair:
    """
    A node-node contact candidate between two fibers.

    This records a pair of nodes whose Euclidean distance is within the
    contact radius. Fibers may have different numbers of nodes.
    """
    # Which two fibers are considered in this contact.
    fiber_i: int
    fiber_j: int

    node_i: int # nodal value in fiber i of the contact
    node_j: int # nodal value in fiber j on the contact

    x_i: jnp.ndarray # coordinate position of the contact point in fiber i
    x_j: jnp.ndarray # coordinate position of the contact point in fiber j

    distance: float # Distance between x_i and x_j

def duplicate_filtering(
        contacts: list[ContactPointPair]
) -> list[ContactPointPair]:
    """
    Remove duplicate contact candidates from a contact list.

    This function depends on the contact pairs already being canonicalized.
    In particular, it assumes each ContactPointPair satisfies the invariant

        (fiber_i, node_i) <= (fiber_j, node_j)

    lexicographically. Under that assumption, two contact records that refer
    to the same pair of endpoints will have the same key and the later one will
    be discarded.

    Parameters
    ----------
    contacts : list[ContactPointPair]
        Contact candidates, preferably already canonicalized.

    Returns
    -------
    list[ContactPointPair]
        Contacts with repeated endpoint pairs removed, preserving the first
        occurrence of each unique pair.
    """
    contacts_filtered = []
    keys = set()
    for contact in contacts:
        assert (contact.fiber_i, contact.node_i) <= (contact.fiber_j, contact.node_j), f"duplicate_filtering expects canonicalized contacts, but got ({contact.fiber_i},{contact.node_i}) > ({contact.fiber_j,contact.node_j}). \n After detection, run values through canonicalize_contact_point_pair."
        key = (
            contact.fiber_i,
            contact.node_i,
            contact.fiber_j,
            contact.node_j
        )
        if key not in keys:
            keys.add(key)
            contacts_filtered.append(contact)
    return contacts_filtered

def distinct_fiber_node2node(
    fiber_x: int,
    x_nd: jnp.ndarray,
    fiber_y: int,
    y_nd: jnp.ndarray,
    radius: float
) -> list[ContactPointPair]:
    """
    Find node-node contact candidates between two distinct fibers.

    Parameters
    ----------
    fiber_x: int
        index for the first fiber
    x_nd : array-like, shape (n_x, 3)
        Node coordinates for the first fiber.
    fiber_y: int
        index for the second fiber
    y_nd : array-like, shape (n_y, 3)
        Node coordinates for the second fiber.
    radius : float
        Contact threshold. A node pair is considered in contact if the
        distance between them is <= radius.

    Returns
    -------
    list[ContactPointPair]
        All node pairs within the threshold.
    """
    x_nd = jnp.asarray(x_nd)
    y_nd = jnp.asarray(y_nd)
    if radius <= 0:
        raise ValueError("radius must be positive")
    if x_nd.ndim != 2 or x_nd.shape[1] != 3:
        raise ValueError(f"Fiber x_nd must have shape (n_nodes, 3)")
    if y_nd.ndim != 2 or y_nd.shape[1] != 3:
        raise ValueError(f"Fiber y_nd must have shape (n_nodes, 3)")


    Nx = len(x_nd)
    Ny = len(y_nd)
    contacts = []
    for i in range(Nx):
        for j in range(Ny):
            dist = jnp.linalg.norm(x_nd[i,:]-y_nd[j,:])
            if dist<=radius:
                contacts.append(
                    canonicalize_contact_point_pair(
                        fiber_i  = fiber_x,
                        fiber_j  = fiber_y,
                        node_i   = i,
                        node_j   = j,
                        x_i      = x_nd[i,:],
                        x_j      = y_nd[j,:],
                        distance = dist,
                    )
                )
    return contacts

def self_fiber_node2node(
    fiber_x: int,
    x_nd: jnp.ndarray,
    radius: float
) -> list[ContactPointPair]:
    """
    Find node-node self-contact candidates within a single fiber.
    This is a first-pass self-contact detector for a fiber represented as an
    ordered sequence of nodes. It checks all node pairs (i, j) with j > i and
    reports a pair only if:

    - the Euclidean distance between the two nodes is <= radius, and
    - the arc length along the fiber between the two nodes is sufficiently
      large to avoid detecting neighboring or near-neighboring nodes as contact

    The arc length is computed from the ordered node sequence by summing the
    lengths of the segments between node indices i and j.

    Parameters
    ----------
    fiber_x : int
        Index of the fiber.
    x_nd : array-like, shape (n_nodes, 3)
        Node coordinates for the fiber. Nodes must be ordered along the fiber.
    radius : float
        Contact threshold. A node pair is considered in contact if its Euclidean
        distance is <= radius and its arc length separation is sufficiently large.

    Returns
    -------
    list[ContactPointPair]
        All non-neighboring self-contact candidates on the fiber.
    """
    x_nd = jnp.asarray(x_nd)
    if radius <= 0:
        raise ValueError("radius must be positive")
    if x_nd.ndim != 2 or x_nd.shape[1] != 3:
        raise ValueError(f"Fiber x_nd must have shape (n_nodes, 3)")

    Nx = len(x_nd)
    contacts = []
    for i in range(Nx):
        for j in range(i+1,Nx):
            dist = jnp.linalg.norm(x_nd[i,:]-x_nd[j,:])
            diff = x_nd[i+1:j+1] - x_nd[i:j]
            arclength = jnp.linalg.norm(diff,axis=1).sum()
            if dist<=radius and arclength>2*radius:
                contacts.append(
                    canonicalize_contact_point_pair(
                        fiber_i  = fiber_x,
                        fiber_j  = fiber_x,
                        node_i   = i,
                        node_j   = j,
                        x_i      = x_nd[i,:],
                        x_j      = x_nd[j,:],
                        distance = dist,
                    )
                   )
    return contacts

def canonicalize_contact_point_pair(
    fiber_i: int,
    fiber_j: int,
    node_i: int,
    node_j: int,
    x_i: jnp.ndarray,
    x_j: jnp.ndarray,
    distance
) -> ContactPointPair:
    """
    Canonicalize a ContactPointPair to ensure (fiber_i,node_i)<=(fiber_j,node_j) lexicographically.
        - if fiber_i != fiber_j, the smaller fiber index comes first.
        - if fiber_i == fiber_j, the smaller node sets the order.

    Parameters
    ----------
    fiber_i: int
        The first fiber listed in the contact
    fiber_j: int
        The second fiber listed in the contact
    node_i: int
        nodal value in fiber i of the contact
    node_j: int
        nodal value in fiber j of the contact
    x_i: jnp.ndarray
        coordinate position of the contact point in fiber i
    x_j: jnp.ndarray
        coordinate position of the contact point in fiber j
    distance: float
        distance between x_i and x_j

    Returns
    -------
    ContactPointPair
        canonicalized ((fiber_i,node_i)<=(fiber_j,node_j)) contact
    """
    A = (fiber_i, node_i)
    B = (fiber_j, node_j)

    if A<B:
        # Keep the order the same
        return ContactPointPair(
            fiber_i  = fiber_i,
            fiber_j  = fiber_j,
            node_i   = node_i,
            node_j   = node_j,
            x_i      = x_i,
            x_j      = x_j,
            distance = distance,
        )
    else:
        # Swap the order
        return ContactPointPair(
            fiber_i  = fiber_j,
            fiber_j  = fiber_i,
            node_i   = node_j,
            node_j   = node_i,
            x_i      = x_j,
            x_j      = x_i,
            distance = distance,
        )

def contact_batch(
    fibers: list[jnp.ndarray],
    radius: float,
    distinct_fiber_fn: Callable = distinct_fiber_node2node,
    self_fiber_fn: Callable = self_fiber_node2node,
) -> list[ContactPointPair]:
    """
    Find node-node contact candidates across a collection of fibers.

    This function orchestrates contact detection by calling one detector for
    distinct-fiber pairs and one detector for self-contact on each fiber.
    The detector functions are injected so alternative contact algorithms can
    be tested without modifying this routine. The returned contact pairs are
    expected to be canonicalized ContactPointPair objects, because duplicate
    filtering depends on the canonical endpoint ordering invariant.

    Parameters
    ----------
    fibers : sequence of arrays, each shape (n_i, 3)
        Collection of fibers. Each fiber may have a different number of nodes.
    radius : float
        Contact threshold. A node pair is considered in contact if the
        distance between them is <= radius.
    distinct_fiber_fn : Callable
        Function used to detect contact between two different fibers.
        It must accept two fiber indices, two node-coordinate arrays, and
        radius, and return a list[ContactPointPair].
    self_fiber_fn : Callable
        Function used to detect self-contact within a single fiber.
        It must accept one fiber index, one node-coordinate array, and radius,
        and return a list[ContactPointPair].

    Returns
    -------
    list[ContactPointPair]
        Flat list of all contact candidates across all fibers, with duplicate
        endpoint pairs removed.
    """
    normalized_fibers = [jnp.asarray(fiber) for fiber in fibers]
    if radius <= 0:
        raise ValueError("radius must be positive")
    for idx, fiber in enumerate(normalized_fibers):
        if fiber.ndim != 2 or fiber.shape[1] != 3:
            raise ValueError(f"Fiber {idx} must have shape (n_nodes, 3)")

    # I am using F for number of fibers
    F = len(normalized_fibers)
    contacts = []
    # Find and add contact points between distinct fibers
    for i in range(F):
        for j in range(i+1,F):
            contacts += distinct_fiber_fn(
                fiber_x = i,
                x_nd    = normalized_fibers[i],
                fiber_y = j,
                y_nd    = normalized_fibers[j],
                radius  = radius
            )
    # Find and add self contact points for on a single fiber
    for i in range(F):
        contacts += self_fiber_fn(
            fiber_x = i,
            x_nd    = normalized_fibers[i],
            radius  = radius
        )
    return duplicate_filtering(contacts)
