import numpy as np
from trimesh import Trimesh
from trimesh import grouping
from trimesh import transformations as tf
from trimesh import util
from trimesh.constants import tol

def sweep_circle(
    path: np.ndarray,
    radii: np.ndarray,
    *,
    segments: int = 64,
    angles: np.ndarray | None = None,
    cap: bool = True,
    connect: bool = True,
    kwargs: dict | None = None,
) -> Trimesh:
    """
    Sweep a circular cross section along a 3D path, allowing the radius to vary
    at each path vertex.

    Parameters
    ----------
    path : (n, 3) float
        3D path vertices (world space).
    radii : (n,) float
        Circle radius at each path vertex. The radius between vertices is
        effectively *linearly interpolated* by the side faces.
    segments : int
        Number of vertices used to discretize the circle boundary.
    angles : (n,) float or None
        Optional roll angle (radians) applied at each vertex about the local
        sweep axis (useful for twisting the section).
    cap : bool
        If True and the path is open, cap both ends.
    connect : bool
        If True and the path is closed, connect the end to the start to make
        a single watertight body.
    kwargs : dict
        Passed to Trimesh constructor (e.g., process=False).

    Returns
    -------
    mesh : trimesh.Trimesh
        The swept, watertight mesh (when capped or closed+connected).
    """
    # --- Validate inputs ------------------------------------------------------
    path = np.asanyarray(path, dtype=np.float64)
    if not util.is_shape(path, (-1, 3)):
        raise ValueError("path must be (n, 3) float array")

    n = len(path)
    radii = np.asanyarray(radii, dtype=np.float64)
    if radii.shape != (n,):
        raise ValueError(f"radii must have shape ({n},), got {radii.shape}")

    if angles is not None:
        angles = np.asanyarray(angles, dtype=np.float64)
        if angles.shape != (n,):
            raise ValueError(f"angles must have shape ({n},), got {angles.shape}")
    else:
        angles = np.zeros(n, dtype=np.float64)

    if segments < 3:
        raise ValueError("segments must be >= 3 for a meaningful circle")

    closed = np.linalg.norm(path[0] - path[-1]) < tol.merge

    # --- Build a unit circle boundary in 2D (XY plane) ------------------------
    # We'll scale this boundary per-slice by the local radius.
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    unit_boundary_2D = np.column_stack([np.cos(theta), np.sin(theta)])  # (m, 2)
    m = segments  # number of boundary vertices per slice

    # For capping, prepare cap triangulation for a 2D disk (fan from center).
    # We'll build a fan [i, i+1, center], with 'center' index == m when used
    # within a single slice. Note: these indices are per-slice (we'll offset).
    cap_faces_2D = np.column_stack([
        np.arange(m, dtype=np.int64),
        (np.arange(m, dtype=np.int64) + 1) % m,
        np.full(m, m, dtype=np.int64)  # center index
    ])

    # For side faces we only need ordered boundary edges per slice: [i, i+1]
    boundary_edges = np.column_stack([
        np.arange(m, dtype=np.int64),
        (np.arange(m, dtype=np.int64) + 1) % m
    ])
    unique = np.arange(m, dtype=np.int64)  # boundary vertex indices
    boundary = boundary_edges  # shape (m, 2)

    # --- Construct the local frame transforms exactly like sweep_polygon -------
    vector = util.unitize(path[1:] - path[:-1])              # (n-1, 3)
    vector_mean = util.unitize(vector[1:] + vector[:-1])     # (n-2, 3) averaged
    normal = np.concatenate([[vector[0]], vector_mean, [vector[-1]]], axis=0)
    if closed and connect:
        normal[0] = util.unitize(normal[[0, -1]].mean(axis=0))
    assert normal.shape == path.shape

    # Convert normals to spherical and produce the per-vertex transform matrices.
    theta_n, phi_n = util.vector_to_spherical(normal).T
    cos_t, sin_t = np.cos(theta_n), np.sin(theta_n)
    cos_p, sin_p = np.cos(phi_n), np.sin(phi_n)
    cos_r, sin_r = np.cos(angles), np.sin(angles)

    zeros = np.zeros(n)
    ones = np.ones(n)

    transforms = np.column_stack(
        [
            -sin_r * cos_p * cos_t + sin_t * cos_r,
            sin_r * sin_t + cos_p * cos_r * cos_t,
            sin_p * cos_t,
            path[:, 0],
            -sin_r * sin_t * cos_p - cos_r * cos_t,
            -sin_r * cos_t + sin_t * cos_p * cos_r,
            sin_p * sin_t,
            path[:, 1],
            sin_p * sin_r,
            -sin_p * cos_r,
            cos_p,
            path[:, 2],
            zeros,
            zeros,
            zeros,
            ones,
        ]
    ).reshape((-1, 4, 4))

    if tol.strict:
        # Sanity: each transform should map +Z to `normal[i]`.
        for nrm, mat in zip(normal, transforms):
            check = tf.transform_points([[0.0, 0.0, 1.0]], mat, translate=False)[0]
            assert np.allclose(check, nrm)

    # --- Build all slice vertices (scale unit circle by local radius) ----------
    # vertices per slice: boundary only; we don't pre-append cap centers here.
    all_vertices = []
    for i, mat in enumerate(transforms):
        # scale boundary by local radius
        b2d = unit_boundary_2D * radii[i]                           # (m, 2)
        verts_h = np.column_stack([b2d, np.zeros(m), np.ones(m)])    # (m, 4)
        v3d = (verts_h @ mat.T)[:, :3]                               # (m, 3)
        all_vertices.append(v3d)
    vertices_3D = np.concatenate(all_vertices, axis=0)               # (n*m, 3)

    # --- Side faces between consecutive slices --------------------------------
    # For each consecutive pair of slices (i, i+1), connect quads -> 2 tris:
    # Using boundary pairs [a,b] in slice i and corresponding [a', b'] in slice i+1
    stride = m
    faces = []
    boundary_next = boundary + stride  # (m, 2), template for "next slice"
    faces_slice = np.column_stack(
        [boundary, boundary_next[:, :1], boundary_next[:, ::-1], boundary[:, 1:]]
    ).reshape((-1, 3))  # triangles for one pair of slices

    # add faces for each segment between path vertices
    for offset in np.arange(n - 1) * stride:
        faces.append(faces_slice + offset)

    # --- Closed loop handling --------------------------------------------------
    if closed and connect:
        # remove duplicated last slice of vertices and wrap faces
        max_vertex = (n - 1) * stride
        vertices_3D = vertices_3D[:max_vertex]
        faces[-1] %= max_vertex

    # --- Caps for open paths ---------------------------------------------------
    if not closed and cap:
        # We'll add a center vertex for each end-cap and fan to boundary
        # First slice (index 0)
        first_center_idx = vertices_3D.shape[0]
        first_center = tf.transform_points([[0.0, 0.0, 0.0]], transforms[0])[0]
        # Append first cap center
        vertices_3D = np.vstack([vertices_3D, first_center])

        # Last slice (index n-1)
        last_center_idx = vertices_3D.shape[0]
        last_center = tf.transform_points([[0.0, 0.0, 0.0]], transforms[-1])[0]
        vertices_3D = np.vstack([vertices_3D, last_center])

        # Build cap faces by mapping the 2D fan to the first/last slice offsets
        # First cap uses slice 0 boundary [0..m-1] plus its center at first_center_idx
        cap0 = cap_faces_2D.copy()
        cap0[:, 2] = first_center_idx  # center index
        # Winding: original boundary lies in +Z in local coordinates; after
        # transform, we want outward normals. Keeping consistency with
        # sweep_polygon: bottom cap is flipped.
        faces.append(np.fliplr(cap0))  # flip for the "bottom" cap

        # Last cap uses slice (n-1) boundary, whose offset is (n-1-1)*stride for open path
        last_offset = (n - 1) * stride  # start index of last slice's boundary in open path
        capN = cap_faces_2D.copy() + last_offset
        capN[:, 2] = last_center_idx  # center index for top cap
        faces.append(capN)

    # --- Finalize mesh ---------------------------------------------------------
    if kwargs is None:
        kwargs = {}
    if "process" not in kwargs:
        # geometry is constructed cleanly; skip expensive post-processing by default
        kwargs["process"] = False

    faces = np.concatenate(faces, axis=0)
    mesh = Trimesh(vertices=vertices_3D, faces=faces, **kwargs)

    if tol.strict:
        assert len(np.unique(faces)) == len(vertices_3D)
        if (not closed) and cap:
            assert mesh.is_volume
        if closed and connect:
            assert mesh.is_volume
            assert mesh.body_count == 1

    return mesh
