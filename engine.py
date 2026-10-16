#!/usr/bin/env python3
"""Ortotrop qatlamli plastinka: CLPT (Ritz) + FSDT (MITC4 FEM) + Rjanitsin-Koltunov.

Bitta faylda ikkita fizik model saqlanadi (bitta model sifatida aralashtirilmaydi):
  CLPT: klassik yupqa plastinka, Ritz bazislari, aniq RK yadro hujayra integrallari.
  FSDT: 5-DOF MITC4 FEM, siljish deformatsiyasi, RK uchun SOE/Maxwell aproksimatsiyasi,
        zarb/garmonik/statik yuklar, qatlam bo'yicha kuchlanish va mustahkamlik.
Har ikki model: ortotrop qatlamlar, S/C/F tayanchlar, Winkler/Pasternak elastik asos.

Ishlatish (numpy scipy matplotlib):
  python birlashtirilgan_clpt_fsdt_rk.py --demo --model fsdt
  python birlashtirilgan_clpt_fsdt_rk.py --demo --model clpt
  python birlashtirilgan_clpt_fsdt_rk.py --demo --model both
  python birlashtirilgan_clpt_fsdt_rk.py --config sozlamalar.json --model both
  python birlashtirilgan_clpt_fsdt_rk.py --self-test

Eslatma: CLPT va FSDT chegaraviy kinematik farazlari, fazoviy diskretizatsiyasi
va viskoelastik vaqt algoritmi turlicha. Natijalar qat'iy teng bo'lishi shart emas.
Birliklar: m, s, kg, Pa, N; burchaklar gradusda.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import linalg, optimize, sparse, special
from scipy.sparse import linalg as spla

DOFS = 5
GAUSS = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))
EDGE_NAMES = ("left", "right", "bottom", "top")
EDGE_LABELS = {"left": "Chap (x=0)", "right": "O'ng (x=a)",
               "bottom": "Pastki (y=0)", "top": "Yuqori (y=b)"}
BC_LABELS = {"clamped": "Qistirib mahkamlangan", "simply_supported": "Sharnirli",
             "free": "Erkin"}
BC_ALIASES = {"1": "clamped", "c": "clamped", "clamped": "clamped",
              "2": "simply_supported", "s": "simply_supported",
              "simply_supported": "simply_supported", "sharnirli": "simply_supported",
              "3": "free", "f": "free", "free": "free", "erkin": "free"}


def normalize_edges(edges=None):
    if edges is None:
        edges = {name: "simply_supported" for name in EDGE_NAMES}
    elif isinstance(edges, (str, int)):
        edges = {name: str(edges) for name in EDGE_NAMES}
    if not isinstance(edges, dict) or set(edges) != set(EDGE_NAMES):
        raise ValueError("Chetlar left, right, bottom, top bo'yicha to'liq berilsin.")
    normalized = {}
    for edge in EDGE_NAMES:
        value = str(edges[edge]).strip().lower()
        if value not in BC_ALIASES:
            raise ValueError(f"{EDGE_LABELS[edge]}: 1/C, 2/S yoki 3/F tanlang.")
        normalized[edge] = BC_ALIASES[value]
    return normalized


def ask_boundaries():
    print("\n--- Chegaraviy shartlar ---")
    print("1 — Qistirib mahkamlangan; 2 — Sharnirli; 3 — Erkin.")
    method = ask_choice("Tanlash: 1=barcha chetga bir xil, 2=har bir chetga alohida",
                        ("1", "2"), "2")
    if method == "1":
        selected = ask_choice("Barcha chetlar uchun shart", ("1", "2", "3"), "2")
        return normalize_edges(selected)
    return normalize_edges({edge: ask_choice(EDGE_LABELS[edge], ("1", "2", "3"), "2")
                            for edge in EDGE_NAMES})


def finite(value: float, name: str, positive: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name}: chekli{' va musbat' if positive else ''} son kerak.")
    return value


@dataclass
class OrthotropicLayer:
    h: float
    theta: float  # degrees, NOT radians
    E1: float
    E2: float
    G12: float
    G13: float
    G23: float
    nu12: float
    rho: float
    is_visco: bool = False
    A: float = 0.0
    alpha: float = 0.5
    beta: float = 1.0
    name: str = ""
    criterion: str = "none"  # none, von_mises, max_stress
    strengths: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.strengths, dict):
            raise ValueError("strengths: parametrlar lug'ati kerak.")
        self.strengths = self.strengths.copy()
        for key in ("h", "E1", "E2", "G12", "G13", "G23", "rho"):
            setattr(self, key, finite(getattr(self, key), key, True))
        for key in ("theta", "nu12", "A", "alpha", "beta"):
            setattr(self, key, finite(getattr(self, key), key))
        if not isinstance(self.is_visco, bool):
            raise ValueError("is_visco: true/false kerak.")
        if self.A < 0 or 1 - self.nu12 * self.nu21 <= 0:
            raise ValueError("A >= 0 va 1-nu12*nu21 > 0 bo'lishi kerak.")
        if self.is_visco:
            if not 0 < self.alpha < 1 or self.beta <= 0:
                raise ValueError("KR yadro: 0 < alpha < 1 va beta > 0 kerak.")
            if self.loss_fraction >= 1:
                raise ValueError("A*Gamma(alpha)/beta**alpha < 1 kerak: K_inf musbat bo'lsin.")
        elif self.A != 0:
            raise ValueError("Elastik qatlamda A=0 bo'lishi kerak.")
        if self.is_isotropic and not -1 < self.nu12 < 0.5:
            raise ValueError("Izotrop material uchun -1 < nu < 0.5 kerak.")
        if self.criterion not in ("none", "von_mises", "max_stress", "max_stress_plane"):
            raise ValueError("criterion: none, von_mises, max_stress yoki max_stress_plane.")
        if self.criterion == "von_mises":
            if not self.is_isotropic:
                raise ValueError("von_mises mezoni bu dasturda faqat izotrop material uchun.")
            required = ("yield",)
        elif self.criterion == "max_stress":
            required = ("Xt", "Xc", "Yt", "Yc", "S12", "S13", "S23")
        elif self.criterion == "max_stress_plane":
            required = ("Xt", "Xc", "Yt", "Yc", "S12")
        else:
            required = ()
        for key in required:
            if key not in self.strengths:
                raise ValueError(f"Mustahkamlik parametri yetishmayapti: {key}")
            self.strengths[key] = finite(self.strengths[key], key, True)

    @property
    def nu21(self):
        return self.nu12 * self.E2 / self.E1

    @property
    def is_isotropic(self):
        shear = self.E1 / (2 * (1 + self.nu12)) if self.nu12 != -1 else -1
        return bool(np.isclose(self.E1, self.E2, rtol=1e-7, atol=0)
                    and all(np.isclose(g, shear, rtol=1e-7, atol=0)
                            for g in (self.G12, self.G13, self.G23)))

    @property
    def loss_fraction(self):
        if not self.is_visco or self.A == 0:
            return 0.0
        log_chi = math.log(self.A) + special.gammaln(self.alpha) - self.alpha * math.log(self.beta)
        return math.exp(log_chi) if log_chi < 700 else math.inf


def transformed_stiffness(layer: OrthotropicLayer):
    """Engineering strain order: [eps_x, eps_y, gamma_xy], [gamma_xz,gamma_yz]."""
    m, n = math.cos(math.radians(layer.theta)), math.sin(math.radians(layer.theta))
    den = 1 - layer.nu12 * layer.nu21
    q11, q22, q12, q66 = layer.E1 / den, layer.E2 / den, layer.nu12 * layer.E2 / den, layer.G12
    q16 = (q11-q12-2*q66)*m**3*n - (q22-q12-2*q66)*m*n**3
    q26 = (q11-q12-2*q66)*m*n**3 - (q22-q12-2*q66)*m**3*n
    qbar = np.array([
        [q11*m**4+2*(q12+2*q66)*m*m*n*n+q22*n**4,
         (q11+q22-4*q66)*m*m*n*n+q12*(m**4+n**4), q16],
        [(q11+q22-4*q66)*m*m*n*n+q12*(m**4+n**4),
         q11*n**4+2*(q12+2*q66)*m*m*n*n+q22*m**4, q26],
        [q16, q26, (q11+q22-2*q12-2*q66)*m*m*n*n+q66*(m**4+n**4)]
    ])
    rotation = np.array([[m, -n], [n, m]])
    qs = rotation @ np.diag([layer.G13, layer.G23]) @ rotation.T
    ts = np.array([[m*m, n*n, 2*m*n],
                   [n*n, m*m, -2*m*n], [-m*n, m*n, m*m-n*n]])
    return qbar, qs, ts, rotation


@dataclass
class Section:
    A: np.ndarray
    B: np.ndarray
    D: np.ndarray
    S: np.ndarray


def laminate_sections(layers, shear_correction=5.0/6.0):
    if not layers:
        raise ValueError("Kamida bitta qatlam kerak.")
    shear_correction = finite(shear_correction, "shear_correction", True)
    H = sum(layer.h for layer in layers)
    z = -H / 2
    parts, bounds = [], []
    inertia = np.zeros(3)
    for layer in layers:
        top = z + layer.h
        q, qs, _, _ = transformed_stiffness(layer)
        dz, dz2, dz3 = top-z, (top**2-z**2)/2, (top**3-z**3)/3
        parts.append(Section(q*dz, q*dz2, q*dz3, shear_correction*qs*dz))
        bounds.append((z, top))
        inertia += layer.rho * np.array([dz, dz2, dz3])
        z = top
    total = Section(*(sum((getattr(s, key) for s in parts),
                          np.zeros((2, 2) if key == "S" else (3, 3)))
                      for key in ("A", "B", "D", "S")))
    return total, parts, bounds, inertia


def calculate_fsdt_matrices(layers, H_total=None, k_shear=5.0/6.0):
    """Compatibility wrapper; thickness is checked, never trusted independently."""
    if H_total is not None and not math.isclose(H_total, sum(x.h for x in layers), rel_tol=1e-10):
        raise ValueError("H_total qatlam qalinliklari yig'indisiga teng emas.")
    s, _, _, _ = laminate_sections(layers, k_shear)
    return s.A, s.B, s.D, s.S


def shape(xi, eta, dx, dy):
    N = 0.25 * np.array([(1-xi)*(1-eta), (1+xi)*(1-eta),
                         (1+xi)*(1+eta), (1-xi)*(1+eta)])
    Nx = 0.5/dx * np.array([-(1-eta), 1-eta, 1+eta, -(1+eta)])
    Ny = 0.5/dy * np.array([-(1-xi), -(1+xi), 1+xi, 1-xi])
    return N, Nx, Ny


def strain_matrices(xi, eta, dx, dy):
    N, Nx, Ny = shape(xi, eta, dx, dy)
    bm, bb = np.zeros((3, 20)), np.zeros((3, 20))
    bm[0, 0::5], bm[1, 1::5] = Nx, Ny
    bm[2, 0::5], bm[2, 1::5] = Ny, Nx
    bb[0, 3::5], bb[1, 4::5] = Nx, Ny
    bb[2, 3::5], bb[2, 4::5] = Ny, Nx

    def raw_shear(x, y):
        ns, nsx, nsy = shape(x, y, dx, dy)
        b = np.zeros((2, 20))
        b[0, 2::5], b[0, 3::5] = nsx, ns
        b[1, 2::5], b[1, 4::5] = nsy, ns
        return b

    # Rectangular MITC4: interpolate the edge-midpoint tying strains.
    bs = np.zeros((2, 20))
    bs[0] = (1-eta)/2*raw_shear(0, -1)[0] + (1+eta)/2*raw_shear(0, 1)[0]
    bs[1] = (1-xi)/2*raw_shear(-1, 0)[1] + (1+xi)/2*raw_shear(1, 0)[1]
    return N, bm, bb, bs


def element_matrices(parts, inertia, dx, dy):
    mass = np.zeros((20, 20))
    stiff = [np.zeros((20, 20)) for _ in parts]
    i0, i1, i2 = inertia
    jmass = np.array([[i0, 0, 0, i1, 0], [0, i0, 0, 0, i1],
                     [0, 0, i0, 0, 0], [i1, 0, 0, i2, 0], [0, i1, 0, 0, i2]])
    for eta in GAUSS:
        for xi in GAUSS:
            N, bm, bb, bs = strain_matrices(xi, eta, dx, dy)
            weight = dx * dy / 4
            interp = np.kron(N.reshape(1, 4), np.eye(5))
            mass += interp.T @ jmass @ interp * weight
            for k, sec in enumerate(parts):
                stiff[k] += weight * (bm.T@sec.A@bm + bm.T@sec.B@bb
                                     + bb.T@sec.B.T@bm + bb.T@sec.D@bb + bs.T@sec.S@bs)
    return mass, [(k+k.T)/2 for k in stiff]


@dataclass
class Mesh:
    a: float
    b: float
    nx: int
    ny: int
    nodes: np.ndarray = field(init=False)
    elements: np.ndarray = field(init=False)
    edofs: np.ndarray = field(init=False)

    def __post_init__(self):
        self.a, self.b = finite(self.a, "a", True), finite(self.b, "b", True)
        for key in ("nx", "ny"):
            val = getattr(self, key)
            if isinstance(val, bool) or int(val) != val or val < 1:
                raise ValueError(f"{key}: musbat butun son kerak.")
            setattr(self, key, int(val))
        self.nodes = np.array([(x, y) for y in np.linspace(0, self.b, self.ny+1)
                               for x in np.linspace(0, self.a, self.nx+1)])
        self.elements = np.array([(j*(self.nx+1)+i, j*(self.nx+1)+i+1,
                                   (j+1)*(self.nx+1)+i+1, (j+1)*(self.nx+1)+i)
                                  for j in range(self.ny) for i in range(self.nx)], dtype=int)
        self.edofs = (self.elements[:, :, None]*5 + np.arange(5)).reshape(-1, 20)

    @property
    def dx(self):
        return self.a / self.nx

    @property
    def dy(self):
        return self.b / self.ny

    @property
    def ndof(self):
        return len(self.nodes) * 5

    def point_vector(self, x, y):
        x, y = finite(x, "x"), finite(y, "y")
        if not 0 <= x <= self.a or not 0 <= y <= self.b:
            raise ValueError("Yuk/kuzatuv nuqtasi plastinka ichida bo'lishi kerak.")
        ix, iy = min(int(x/self.dx), self.nx-1), min(int(y/self.dy), self.ny-1)
        N, _, _ = shape(2*(x-ix*self.dx)/self.dx-1, 2*(y-iy*self.dy)/self.dy-1,
                        self.dx, self.dy)
        out = np.zeros(self.ndof)
        out[self.elements[iy*self.nx+ix]*5+2] = N
        return out

    def pressure_vector(self):
        out = np.zeros(self.ndof)
        np.add.at(out, (self.elements*5+2).ravel(),
                  np.full(self.elements.size, self.dx*self.dy/4))
        return out

    def patch_vector(self, x, y, width, height):
        """Uniform rectangular contact pressure, normalized to unit TOTAL force."""
        self.point_vector(x, y)
        width, height = finite(width, "patch_width", True), finite(height, "patch_height", True)
        x0, x1, y0, y1 = x-width/2, x+width/2, y-height/2, y+height/2
        if x0 < 0 or x1 > self.a or y0 < 0 or y1 > self.b:
            raise ValueError("Kontakt to'rtburchagi plastinka ichida bo'lishi kerak.")
        out = np.zeros(self.ndof)
        for e, conn in enumerate(self.elements):
            xe, ye = self.nodes[conn[0]]
            lo_x, hi_x = max(x0, xe), min(x1, xe+self.dx)
            lo_y, hi_y = max(y0, ye), min(y1, ye+self.dy)
            if hi_x <= lo_x or hi_y <= lo_y:
                continue
            for gy in GAUSS:
                yy = (lo_y+hi_y)/2 + gy*(hi_y-lo_y)/2
                for gx in GAUSS:
                    xx = (lo_x+hi_x)/2 + gx*(hi_x-lo_x)/2
                    N, _, _ = shape(2*(xx-xe)/self.dx-1, 2*(yy-ye)/self.dy-1,
                                    self.dx, self.dy)
                    out[conn*5+2] += N*(hi_x-lo_x)*(hi_y-lo_y)/(4*width*height)
        return out


@dataclass
class FEMModel:
    mesh: Mesh
    layers: list
    section: Section
    bounds: list
    inertia: np.ndarray
    M: sparse.csr_matrix
    layer_K: list
    free: np.ndarray
    shear_correction: float
    edges: dict = field(default_factory=dict)
    in_plane: str = "fixed"
    foundation_K: sparse.csr_matrix | None = None
    kw: float = 0.0  # Winkler [N/m^3]
    kp: float = 0.0  # Pasternak [N/m]

    @property
    def K(self):
        base = self.foundation_K if self.foundation_K is not None else sparse.csr_matrix(self.M.shape)
        return sum(self.layer_K, base)

    @property
    def K_relaxed(self):
        base = self.foundation_K if self.foundation_K is not None else sparse.csr_matrix(self.M.shape)
        return sum(((1-layer.loss_fraction)*k for layer, k in zip(self.layers, self.layer_K)), base)

    def full(self, reduced):
        out = np.zeros(self.mesh.ndof)
        out[self.free] = reduced
        return out


def assemble_model(layers, a, b, nx=8, ny=8, edges=None, in_plane="fixed",
                   shear_correction=5.0/6.0, kw=0.0, kp=0.0):
    kw, kp = finite(kw, "kw"), finite(kp, "kp")
    if kw < 0 or kp < 0:
        raise ValueError("kw va kp manfiy bo'lmasligi kerak.")
    mesh = Mesh(a, b, nx, ny)
    sec, parts, bounds, inertia = laminate_sections(layers, shear_correction)
    me, ke = element_matrices(parts, inertia, mesh.dx, mesh.dy)
    rows = np.repeat(mesh.edofs, 20, axis=1).ravel()
    cols = np.tile(mesh.edofs, (1, 20)).ravel()

    def assemble(local):
        return sparse.coo_matrix((np.tile(local.ravel(), len(mesh.elements)), (rows, cols)),
                                 shape=(mesh.ndof, mesh.ndof)).tocsr()

    edges = normalize_edges(edges)
    if in_plane not in ("fixed", "normal", "free"):
        raise ValueError("in_plane: fixed, normal yoki free.")
    fixed = set()
    for edge, kind in edges.items():
        if kind not in ("simply_supported", "clamped", "free"):
            raise ValueError(f"{edge}: noma'lum tayanch turi {kind}")
        axis = 0 if edge in ("left", "right") else 1
        coordinate = 0 if edge in ("left", "bottom") else (mesh.a if axis == 0 else mesh.b)
        nodes = np.flatnonzero(np.isclose(mesh.nodes[:, axis], coordinate,
                                         rtol=0, atol=max(mesh.a, mesh.b)*1e-12))
        if kind == "clamped":
            dofs = (0, 1, 2, 3, 4)
        elif kind == "simply_supported":
            # Hard line hinge: w=0, tangential section slope=0; normal slope free.
            dofs = [2, 4 if axis == 0 else 3]
            if in_plane == "fixed":
                dofs += [0, 1]
            elif in_plane == "normal":
                dofs += [axis]
        else:
            dofs = ()
        fixed.update(int(node*5+dof) for node in nodes for dof in dofs)
    free = np.setdiff1d(np.arange(mesh.ndof), sorted(fixed))
    if not len(free):
        raise ValueError("Erkin noma'lum qolmadi: to'rni zichlashtiring yoki tayanchni tekshiring.")
    mass = assemble(me)[free][:, free].tocsr()
    stiffnesses = [assemble(k)[free][:, free].tocsr() for k in ke]
    # Winkler: (kw/2)*int(w^2)dA; Pasternak: (kp/2)*int(|grad(w)|^2)dA.
    foundation_local = np.zeros((20, 20))
    for eta in GAUSS:
        for xi in GAUSS:
            N, Nx, Ny = shape(xi, eta, mesh.dx, mesh.dy)
            nw, nxw, nyw = np.zeros(20), np.zeros(20), np.zeros(20)
            nw[2::5], nxw[2::5], nyw[2::5] = N, Nx, Ny
            foundation_local += mesh.dx*mesh.dy/4 * (
                kw*np.outer(nw, nw)
                + kp*(np.outer(nxw, nxw) + np.outer(nyw, nyw)))
    foundation = assemble(foundation_local)[free][:, free].tocsr()
    return FEMModel(mesh, layers, sec, bounds, inertia, mass, stiffnesses, free,
                    shear_correction, edges.copy(), in_plane, foundation, kw, kp)


def modes(model, count=30, relaxed=False):
    K = model.K_relaxed if relaxed else model.K
    n = len(model.free)
    if n <= 1200:
        vals, vecs = linalg.eigh(K.toarray(), model.M.toarray(),
                                subset_by_index=(0, min(n-1, count+7)), driver="gvx")
    else:
        k = min(n-2, count+8)
        scale = float(np.median(np.abs(K.diagonal()) / model.M.diagonal()))
        vals, vecs = spla.eigsh(K, M=model.M, k=k, sigma=-max(scale*1e-8, 1e-8),
                               which="LM", maxiter=10000)
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
    tol = max(1.0, float(np.max(np.abs(vals)))) * 1e-8
    if float(np.min(vals)) < -tol:
        raise RuntimeError("Manfiy fizik eigenvalue: model/bikrlikni tekshiring.")
    rigid = int(np.sum(np.abs(vals) <= tol))
    positive = vals > tol
    return np.sqrt(vals[positive])/(2*np.pi), vecs[:, positive], rigid


@dataclass
class KernelApproximation:
    rates: np.ndarray
    fractions: np.ndarray
    error: float
    chi: float


def fit_kernel(layer, dt, duration, tolerance=2e-5):
    """Positive Maxwell spectrum fitted to exact normalized relaxation, not R(0).

    Both endpoints g(0)=1 and g(infinity)=1-chi are enforced by normalization.
    Independent log-grid validation covers [dt/128, max(T,30/beta)].
    The reported error is absolute, relative to the instantaneous layer modulus.
    """
    dt, duration = finite(dt, "dt", True), finite(duration, "duration", True)
    tolerance = finite(tolerance, "tolerance", True)
    chi = layer.loss_fraction
    if chi == 0:
        return KernelApproximation(np.empty(0), np.empty(0), 0.0, 0.0)
    low_t = min(dt/128, 1/(128*layer.beta))
    high_t = max(duration, 30/layer.beta)
    train_t = np.geomspace(low_t, high_t, 300)
    check_t = np.geomspace(low_t, high_t, 1600)
    target = special.gammaincc(layer.alpha, layer.beta*train_t)
    upper_rate = max(layer.beta*100, 40/low_t)
    for number in (48, 80, 128, 192):
        rates = np.geomspace(layer.beta, upper_rate, number)
        exponential = np.exp(-np.outer(train_t, rates))
        matrix = np.vstack([exponential, np.full((1, number), 20.0)])
        rhs = np.r_[target, 20.0]
        try:
            fractions, _ = optimize.nnls(matrix, rhs, maxiter=number*20)
        except RuntimeError:
            continue
        if fractions.sum() <= 0:
            continue
        fractions /= fractions.sum()
        keep = fractions > 1e-12
        rates, fractions = rates[keep], fractions[keep]
        fractions *= chi/fractions.sum()
        exact = 1-chi*special.gammainc(layer.alpha, layer.beta*check_t)
        approximate = 1-chi + np.exp(-np.outer(check_t, rates)) @ fractions
        error = float(np.max(np.abs(exact-approximate)))
        if error <= tolerance:
            return KernelApproximation(rates, fractions, error, chi)
    raise RuntimeError(f"{layer.name or 'Qatlam'} yadrosi {tolerance:g} aniqlikka kelmadi. "
                       "Yadro parametrlarini yoki vaqt qadamini tekshiring.")


@dataclass
class Load:
    kind: str
    amplitude: float
    spatial: str = "point"
    x: float | None = None
    y: float | None = None
    patch_width: float | None = None
    patch_height: float | None = None
    frequency_hz: float = 0.0
    phase_degrees: float = 0.0
    duration: float = 0.0
    profile: str = "half_sine"
    rise_fraction: float = 0.1
    decay: float = 2.0
    start_time: float = 0.0
    ramp_time: float = 0.0

    def __post_init__(self):
        if self.kind not in ("point", "uniform", "harmonic", "pulse"):
            raise ValueError("load.kind: point, uniform, harmonic yoki pulse.")
        if self.kind == "point":
            self.spatial = "patch" if self.patch_width is not None else "point"
        if self.kind == "uniform":
            self.spatial = "uniform"
        if self.spatial not in ("point", "uniform", "patch"):
            raise ValueError("load.spatial: point, uniform yoki patch.")
        for name in ("amplitude", "frequency_hz", "phase_degrees", "duration", "decay",
                     "rise_fraction", "start_time", "ramp_time"):
            setattr(self, name, finite(getattr(self, name), name))
        if self.start_time < 0 or self.ramp_time < 0:
            raise ValueError("start_time va ramp_time >= 0 kerak.")
        if self.kind == "harmonic" and self.frequency_hz <= 0:
            raise ValueError("frequency_hz > 0 kerak; birlik Hz.")
        if self.kind == "pulse":
            if self.duration <= 0 or self.profile not in ("half_sine", "triangle", "friedlander"):
                raise ValueError("Zarb: duration > 0 va tanilgan profile kerak.")
            if not 0 < self.rise_fraction < 1 or self.decay < 0:
                raise ValueError("0 < rise_fraction < 1, decay >= 0 kerak.")

    def value(self, time):
        t = time-self.start_time
        if t < 0:
            return 0.0
        ramp = (0.5*(1-math.cos(math.pi*t/self.ramp_time))
                if self.ramp_time > 0 and t < self.ramp_time else 1.0)
        if self.kind in ("point", "uniform"):
            factor = 1.0
        elif self.kind == "harmonic":
            factor = math.sin(2*math.pi*self.frequency_hz*t+math.radians(self.phase_degrees))
        elif t > self.duration:
            factor = 0.0
        elif self.profile == "half_sine":
            factor = math.sin(math.pi*t/self.duration)
        elif self.profile == "triangle":
            s = t/self.duration
            factor = s/self.rise_fraction if s <= self.rise_fraction else (1-s)/(1-self.rise_fraction)
        else:
            s = t/self.duration
            factor = (1-s)*math.exp(-self.decay*s)
        return self.amplitude*factor*ramp

    def vector(self, mesh):
        if self.spatial == "uniform":
            return mesh.pressure_vector()
        x, y = mesh.a/2 if self.x is None else self.x, mesh.b/2 if self.y is None else self.y
        if self.spatial == "patch":
            return mesh.patch_vector(x, y, self.patch_width, self.patch_height)
        return mesh.point_vector(x, y)


def harmonic_response(model, load):
    """Exact KR steady-state frequency response; independent of the SOE fit."""
    if load.kind != "harmonic":
        return None
    omega = 2*np.pi*load.frequency_hz
    force = load.vector(model.mesh)[model.free]*load.amplitude
    if not any(layer.loss_fraction for layer in model.layers) and np.any(force):
        # A forced, undamped eigenmode has no finite steady-state amplitude.
        if len(model.free) <= 1200:
            vals, vecs = linalg.eigh(model.K.toarray(), model.M.toarray(),
                                    subset_by_value=((omega*(1-1e-8))**2,
                                                     (omega*(1+1e-8))**2), driver="gvx")
        else:
            vals, vecs = spla.eigsh(model.K, M=model.M, k=3,
                                   sigma=omega**2*(1+1e-6), which="LM")
            keep = np.abs(np.sqrt(np.maximum(vals, 0))/omega-1) < 1e-8
            vals, vecs = vals[keep], vecs[:, keep]
        for vector in vecs.T:
            if abs(vector@force) > 1e-10*np.linalg.norm(vector)*np.linalg.norm(force):
                return {"status": "unbounded_undamped_resonance", "center_amplitude_mm": None}
    dyn = -omega**2*model.M.astype(complex) + model.foundation_K
    for layer, k in zip(model.layers, model.layer_K):
        factor = 1.0
        if layer.loss_fraction:
            factor -= layer.loss_fraction*(layer.beta/(layer.beta+1j*omega))**layer.alpha
        dyn = dyn + factor*k
    with warnings.catch_warnings():
        warnings.simplefilter("error", spla.MatrixRankWarning)
        try:
            displacement = spla.spsolve(dyn.tocsc(), force)
        except (spla.MatrixRankWarning, RuntimeError):
            return {"status": "singular_resonance", "center_amplitude_mm": None}
    if not np.all(np.isfinite(displacement)):
        return {"status": "singular_resonance", "center_amplitude_mm": None}
    center = model.mesh.point_vector(model.mesh.a/2, model.mesh.b/2)[model.free]
    value = center @ displacement
    return {"status": "steady_state", "center_amplitude_mm": float(abs(value)*1000),
            "phase_relative_to_force_degrees": float(np.degrees(np.angle(value)))}


class StressTracker:
    """Gauss-point in-plane stresses at every layer's bottom/top.

    Transverse stresses are energy-corrected FSDT averages, not a 3-D
    through-thickness reconstruction. No point-contact/delamination claim.
    """
    def __init__(self, model):
        self.model = model
        matrices, coords = [], []
        for eta in GAUSS:
            for xi in GAUSS:
                _, bm, bb, bs = strain_matrices(xi, eta, model.mesh.dx, model.mesh.dy)
                matrices.append((bm, bb, bs))
                xy = model.mesh.nodes[model.mesh.elements[:, 0]] + np.array([
                    (xi+1)*model.mesh.dx/2, (eta+1)*model.mesh.dy/2])
                coords.append(xy)
        self.bm = np.concatenate([x[0] for x in matrices])
        self.bb = np.concatenate([x[1] for x in matrices])
        self.bs = np.concatenate([x[2] for x in matrices])
        self.coords = np.stack(coords, axis=1)
        self.material = [transformed_stiffness(layer) for layer in model.layers]
        self.reports = [{"layer": i+1, "name": layer.name, "criterion": layer.criterion,
                         "max_von_mises_pa": 0.0, "von_mises_location": None,
                         "max_failure_index": None if layer.criterion == "none" else 0.0,
                         "failure_location": None} for i, layer in enumerate(model.layers)]

    def update(self, effective_displacements, time):
        mesh = self.model.mesh
        for k, (layer, effective) in enumerate(zip(self.model.layers, effective_displacements)):
            elem = self.model.full(effective)[mesh.edofs]
            membrane = (elem @ self.bm.T).reshape(-1, 4, 3)
            curvature = (elem @ self.bb.T).reshape(-1, 4, 3)
            shear = (elem @ self.bs.T).reshape(-1, 4, 2)
            qb, qs, ts, rotation = self.material[k]
            z = np.array(self.model.bounds[k])
            eps = membrane[None, ...] + z[:, None, None, None]*curvature[None, ...]
            stress = eps @ qb.T
            transverse = self.model.shear_correction*(shear @ qs.T)
            sx, sy, txy = stress[..., 0], stress[..., 1], stress[..., 2]
            vm = np.sqrt(np.maximum(0, sx*sx-sx*sy+sy*sy
                                    +3*(txy*txy+np.sum(transverse**2, axis=-1)[None, ...])))

            def location(index):
                side, elem_id, gp = np.unravel_index(index, vm.shape)
                x, y = self.coords[elem_id, gp]
                return {"x_m": float(x), "y_m": float(y), "z_m": float(z[side]),
                        "time_s": float(time), "element": int(elem_id)}

            report = self.reports[k]
            idx = int(np.argmax(vm))
            maximum = float(vm.flat[idx])
            if maximum >= report["max_von_mises_pa"]:
                report["max_von_mises_pa"], report["von_mises_location"] = maximum, location(idx)
            if layer.criterion == "none":
                continue
            if layer.criterion == "von_mises":
                failure = vm/layer.strengths["yield"]
            else:
                local = stress @ ts.T
                transverse_local = transverse @ rotation
                p = layer.strengths
                r1 = np.where(local[..., 0] >= 0, local[..., 0]/p["Xt"], -local[..., 0]/p["Xc"])
                r2 = np.where(local[..., 1] >= 0, local[..., 1]/p["Yt"], -local[..., 1]/p["Yc"])
                r12 = np.abs(local[..., 2])/p["S12"]
                if layer.criterion == "max_stress_plane":
                    failure = np.maximum.reduce([r1, r2, r12])
                else:
                    r13 = np.broadcast_to(np.abs(transverse_local[..., 0])[None, ...]/p["S13"], r1.shape)
                    r23 = np.broadcast_to(np.abs(transverse_local[..., 1])[None, ...]/p["S23"], r1.shape)
                    failure = np.maximum.reduce([r1, r2, r12, r13, r23])
            idx = int(np.argmax(failure))
            maximum = float(failure.flat[idx])
            if maximum >= report["max_failure_index"]:
                report["max_failure_index"], report["failure_location"] = maximum, location(idx)


@dataclass
class SimulationResult:
    t: np.ndarray
    center_w: np.ndarray
    center_v: np.ndarray
    center_a: np.ndarray
    frequencies: np.ndarray
    relaxed_frequencies: np.ndarray
    rigid_modes: int
    peak: dict
    peak_displacement: np.ndarray
    final_displacement: np.ndarray
    final_velocity: np.ndarray
    stress_reports: list
    kernel_reports: list
    harmonic: dict | None
    dt: float
    energy: np.ndarray | None
    memory_work: np.ndarray | None = None
    external_work: np.ndarray | None = None


def simulate(model, load, total_time, dt=None, w0_mm=0.0, v0=0.0, kernel_tolerance=2e-5,
             memory_initial="instantaneous", max_steps=300000, record_energy=False,
             calculate_stress=True, progress: Callable | None = None,
             initial_displacement=None, initial_velocity=None):
    total_time = finite(total_time, "total_time", True)
    w0_mm, v0 = finite(w0_mm, "w0_mm"), finite(v0, "v0")
    kernel_tolerance = finite(kernel_tolerance, "kernel_tolerance", True)
    if memory_initial not in ("instantaneous", "relaxed"):
        raise ValueError("memory_initial: instantaneous yoki relaxed.")
    freq, vectors, rigid = modes(model)
    relaxed_freq = modes(model, count=4, relaxed=True)[0] if any(
        x.loss_fraction for x in model.layers) else freq.copy()
    mesh = model.mesh
    spatial = load.vector(mesh)[model.free]
    center = mesh.point_vector(mesh.a/2, mesh.b/2)[model.free]
    if np.linalg.norm(spatial) == 0:
        raise ValueError("Yuk faqat mahkamlangan DOFlarga tushmoqda; ta'sir joyini o'zgartiring.")
    w_positions = np.flatnonzero(model.free % 5 == 2)
    bending = []
    for j in range(len(freq)):
        vector = vectors[:, j]
        full = model.full(vector).reshape(-1, 5)
        physical_scale = max(np.max(np.abs(full[:, :3])),
                             sum(x.h for x in model.layers)*np.max(np.abs(full[:, 3:])))
        if np.max(np.abs(full[:, 2])) > physical_scale*1e-5:
            bending.append(j)
    if dt is None:
        excited = [j for j in bending if abs(vectors[:, j] @ spatial)
                   > 1e-4*max(abs(vectors[:, i] @ spatial) for i in bending)] if bending else []
        resolved = excited[:8] or bending[:3]
        dt = min(total_time/200, 1/(40*freq[max(resolved)])) if resolved else total_time/200
        if load.kind == "harmonic":
            dt = min(dt, 1/(60*load.frequency_hz))
        if load.kind == "pulse":
            rise = (load.duration*load.rise_fraction if load.profile == "triangle" else load.duration)
            dt = min(dt, rise/60)
        if load.ramp_time:
            dt = min(dt, load.ramp_time/60)
    dt = finite(dt, "dt", True)
    ratio = total_time/dt
    if not math.isfinite(ratio):
        raise ValueError("total_time/dt juda katta.")
    nsteps = max(1, int(math.ceil(ratio)))
    if nsteps > max_steps:
        raise ValueError(f"{nsteps} vaqt qadami talab qilindi (max_steps={max_steps}). "
                         "Hisob resursi, dt va konvergentsiyani tekshiring.")
    dt = total_time/nsteps
    times = np.arange(nsteps+1)*dt
    n = len(model.free)
    displacement = np.zeros(n)
    velocity = np.zeros(n)
    if w0_mm or v0:
        candidates = [j for j in bending if abs(center@vectors[:, j]) > 1e-8*np.max(
            np.abs(vectors[w_positions, j]))]
        if not candidates:
            raise ValueError("Markazda noldan farqli boshlang'ich flexural moda topilmadi.")
        mode = vectors[:, candidates[0]]/(center@vectors[:, candidates[0]])
        displacement, velocity = w0_mm/1000*mode, v0*mode
    for data, name in ((initial_displacement, "initial_displacement"),
                       (initial_velocity, "initial_velocity")):
        if data is not None and (np.shape(data) != (n,) or not np.all(np.isfinite(data))):
            raise ValueError(f"{name}: erkin DOFlar soniga mos chekli vektor kerak.")
    if initial_displacement is not None:
        displacement = np.array(initial_displacement, dtype=float, copy=True)
    if initial_velocity is not None:
        velocity = np.array(initial_velocity, dtype=float, copy=True)

    fits, states, exponentials, phis, coefficients = [], [], [], [], []
    kernel_reports = []
    keff = model.foundation_K.copy()
    for layer, stiffness in zip(model.layers, model.layer_K):
        fit = fit_kernel(layer, dt, total_time, kernel_tolerance)
        x = fit.rates*dt
        exp = np.exp(-x)
        phi = -np.expm1(-x)/x  # analytic integral; fit.rates are strictly positive
        state = np.tile(displacement, (len(x), 1))
        if memory_initial == "relaxed":
            state.fill(0)
        coefficient = 1-fit.chi + float(fit.fractions@phi)
        keff += coefficient*stiffness
        fits.append(fit); states.append(state); exponentials.append(exp); phis.append(phi)
        coefficients.append(coefficient)
        kernel_reports.append({"name": layer.name, "branches": len(x), "chi": fit.chi,
                               "max_relaxation_error": fit.error})

    def effective_fields():
        return [(1-f.chi)*displacement + np.einsum("i,ij->j", f.fractions, state)
                for f, state in zip(fits, states)]

    def energy_value():
        val = (0.5*float(velocity@(model.M@velocity))
               + 0.5*float(displacement@(model.foundation_K@displacement)))
        for stiffness, f, state in zip(model.layer_K, fits, states):
            val += 0.5*(1-f.chi)*float(displacement@(stiffness@displacement))
            if len(f.rates):
                products = stiffness@state.T
                val += 0.5*float(np.sum(f.fractions*np.einsum("ij,ji->i", state, products)))
        return val

    fields = effective_fields()
    K_instant = model.K
    internal = sum((k@field for k, field in zip(model.layer_K, fields)),
                   model.foundation_K@displacement)
    mass_solver = spla.splu(model.M.tocsc())
    acceleration = mass_solver.solve(spatial*load.value(0)-internal)
    # Newmark average acceleration, beta_N=1/4, gamma_N=1/2.
    acoef = 4/dt**2
    solver = spla.splu((acoef*model.M+keff).tocsc())
    history_w, history_v, history_a = (np.empty(nsteps+1) for _ in range(3))
    history_w[0], history_v[0], history_a[0] = center@displacement, center@velocity, center@acceleration
    memory_work, external_work = np.zeros(nsteps+1), np.zeros(nsteps+1)
    previous_memory_power = float((K_instant@displacement-internal)@velocity)
    previous_external_power = float(load.value(0)*(spatial@velocity))
    energy = np.empty(nsteps+1) if record_energy else None
    if energy is not None:
        energy[0] = energy_value()
    tracker = StressTracker(model) if calculate_stress else None
    if tracker:
        tracker.update(fields, 0)
    peak = {"abs_w_m": -1.0, "signed_w_m": 0.0, "x_m": 0.0, "y_m": 0.0, "time_s": 0.0}
    peak_displacement = model.full(displacement)

    def update_peak(time):
        nonlocal peak_displacement
        full = model.full(displacement)
        field_w = full[2::5]
        idx = int(np.argmax(np.abs(field_w)))
        if abs(field_w[idx]) > peak["abs_w_m"]:
            x, y = mesh.nodes[idx]
            peak.update(abs_w_m=float(abs(field_w[idx])), signed_w_m=float(field_w[idx]),
                        x_m=float(x), y_m=float(y), time_s=float(time))
            peak_displacement = full

    update_peak(0)
    report_every = max(1, nsteps//10)
    for step in range(1, nsteps+1):
        predicted = displacement+dt*velocity+0.25*dt**2*acceleration
        predicted_velocity = velocity+0.5*dt*acceleration
        history_force = np.zeros(n)
        for stiffness, f, state, exp, phi in zip(model.layer_K, fits, states, exponentials, phis):
            if len(f.rates):
                history_vector = (np.einsum("i,ij->j", f.fractions*exp, state)
                                  - float(f.fractions@phi)*displacement)
                history_force += stiffness@history_vector
        rhs = spatial*load.value(times[step])-history_force + acoef*(model.M@predicted)
        new_displacement = solver.solve(rhs)
        new_acceleration = acoef*(new_displacement-predicted)
        new_velocity = predicted_velocity+0.5*dt*new_acceleration
        increment = new_displacement-displacement
        for state, exp, phi in zip(states, exponentials, phis):
            if len(exp):
                state *= exp[:, None]
                state += phi[:, None]*increment
        displacement, velocity, acceleration = new_displacement, new_velocity, new_acceleration
        if not np.all(np.isfinite(displacement)):
            raise RuntimeError("Cheksiz/NaN javob: parametrlar va modelni tekshiring.")
        history_w[step], history_v[step], history_a[step] = center@displacement, center@velocity, center@acceleration
        update_peak(times[step])
        fields_now = effective_fields()
        internal_now = sum((k@field for k, field in zip(model.layer_K, fields_now)),
                           model.foundation_K@displacement)
        memory_power = float((K_instant@displacement-internal_now)@velocity)
        external_power = float(load.value(times[step])*(spatial@velocity))
        memory_work[step] = memory_work[step-1]+0.5*dt*(previous_memory_power+memory_power)
        external_work[step] = external_work[step-1]+0.5*dt*(previous_external_power+external_power)
        previous_memory_power, previous_external_power = memory_power, external_power
        if tracker:
            tracker.update(fields_now, times[step])
        if energy is not None:
            energy[step] = energy_value()
        if progress and (step % report_every == 0 or step == nsteps):
            progress(step/nsteps)
    if tracker:
        for report in tracker.reports:
            fi = report["max_failure_index"]
            report["assessment"] = ("not_evaluated" if fi is None else
                                    "limit_exceeded" if fi > 1 else "within_selected_criterion")
            if load.spatial == "point" and fi is not None:
                report["assessment"] = "point_load_local_stress_inconclusive"
    return SimulationResult(times, history_w, history_v, history_a, freq, relaxed_freq, rigid,
                            peak, peak_displacement, model.full(displacement), model.full(velocity),
                            tracker.reports if tracker else [], kernel_reports,
                            harmonic_response(model, load), dt, energy, memory_work, external_work)


def save_results(model, result, directory, show=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/"history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", "center_w_m", "center_w_mm", "center_v_m_s", "center_a_m_s2",
                         "memory_work_J", "external_work_J"])
        writer.writerows(zip(result.t, result.center_w, result.center_w*1000,
                             result.center_v, result.center_a, result.memory_work, result.external_work))
    summary = {
        "model": "linear MITC4 FSDT FEM; layerwise scalar KR kernels",
        "boundary_conditions": model.edges,
        "hinged_in_plane": model.in_plane,
        "foundation": {"kw_N_per_m3": model.kw, "kp_N_per_m": model.kp},
        "mesh": {"nx": model.mesh.nx, "ny": model.mesh.ny},
        "frequency_definition": "instantaneous and fully relaxed ELASTIC eigenfrequencies; not damped poles",
        "instantaneous_frequencies_hz": result.frequencies[:3].tolist(),
        "relaxed_frequencies_hz": result.relaxed_frequencies[:3].tolist(),
        "rigid_modes_in_search": result.rigid_modes, "dt_s": result.dt,
        "steps": len(result.t)-1, "peak": result.peak, "stress": result.stress_reports,
        "kernels": result.kernel_reports, "harmonic_steady_state": result.harmonic,
        "section": {key: getattr(model.section, key).tolist() for key in ("A", "B", "D", "S")},
        "inertia_I0_I1_I2": model.inertia.tolist(),
        "final_memory_work_J": float(result.memory_work[-1]),
        "final_external_work_J": float(result.external_work[-1]),
        "memory_work_definition": "Integral of memory force dot velocity; NOT dissipated energy.",
        "limits": ["Perfect bonding; small strain/rotation; no delamination/plasticity.",
                   "Transverse stresses are FSDT averages.",
                   "Mesh, dt, and kernel tolerance convergence must be checked.",
                   "Point-load local maximum stress is not a strength verdict."]
    }
    (directory/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    np.savez_compressed(directory/"fields.npz", nodes=model.mesh.nodes, elements=model.mesh.elements,
                        peak_displacement=result.peak_displacement,
                        final_displacement=result.final_displacement)
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(17, 5), constrained_layout=True)
    ax = fig.add_subplot(131)
    ax.plot(result.t, result.center_w*1000, lw=1.3, color="#00695c")
    ax.set(xlabel="Vaqt [s]", ylabel="Markaziy ko'chish [mm]", title="Dinamik javob")
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(132, projection="3d")
    nodes = model.mesh.nodes.reshape(model.mesh.ny+1, model.mesh.nx+1, 2)
    w = result.peak_displacement[2::5].reshape(model.mesh.ny+1, model.mesh.nx+1)*1000
    ax.plot_surface(nodes[..., 0], nodes[..., 1], w, cmap="viridis")
    ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="w [mm]",
           title=f"Maksimal salqilik payti: {result.peak['time_s']:.5g} s")
    ax = fig.add_subplot(133)
    ax.plot(result.t, result.memory_work, color="#d32f2f")
    ax.set(xlabel="Vaqt [s]", ylabel="Ish [J]", title="Xotira kuchining ishi")
    ax.grid(alpha=.3)
    fig.savefig(directory/"response.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    if result.stress_reports:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        labels, values, colors = [], [], []
        for report in result.stress_reports:
            labels.append(report["name"] or f"Qatlam {report['layer']}")
            value = report["max_failure_index"]
            values.append(0 if value is None else value)
            valid = report["assessment"] not in ("not_evaluated", "point_load_local_stress_inconclusive")
            colors.append("#c62828" if valid and value > 1 else "#2e7d32" if valid else "#9e9e9e")
        bars = ax.bar(labels, values, color=colors)
        ax.axhline(1, color="#c62828", linestyle="--", label="Mezon chegarasi")
        for bar, report in zip(bars, result.stress_reports):
            fi = report["max_failure_index"]
            label = "Baholanmagan" if fi is None else f"{fi:.4g}"
            if report["assessment"] == "point_load_local_stress_inconclusive":
                label += " (nuqtaviy yuk)"
            ax.annotate(label, (bar.get_x()+bar.get_width()/2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
        ax.set(ylabel="Tanlangan mustahkamlik mezoni indeksi",
               title="Qatlamlar bo'yicha mustahkamlik tahlili")
        ax.set_ylim(0, max(1.25, max(values)*1.2))
        ax.legend()
        fig.savefig(directory/"strength.png", dpi=160)
        if show:
            plt.show()
        plt.close(fig)


def demo_config():
    carbon = dict(h=0.0015, E1=140e9, E2=10e9, G12=5e9, G13=5e9, G23=3e9,
                  nu12=0.3, rho=1600, criterion="max_stress",
                  strengths=dict(Xt=1.2e9, Xc=800e6, Yt=50e6, Yc=150e6,
                                 S12=60e6, S13=50e6, S23=40e6))
    E, nu = 2e9, 0.35
    return {
        "a": 0.3, "b": 0.2, "nx": 8, "ny": 6, "in_plane": "fixed",
        "kw": 0.0, "kp": 0.0,
        "edges": {key: "simply_supported" for key in ("left", "right", "bottom", "top")},
        "layers": [
            dict(carbon, name="Pastki +30", theta=30, is_visco=True, A=0.08, alpha=0.35, beta=3),
            dict(name="Polimer", h=0.001, theta=0, E1=E, E2=E, G12=E/(2*(1+nu)),
                 G13=E/(2*(1+nu)), G23=E/(2*(1+nu)), nu12=nu, rho=1180,
                 is_visco=True, A=0.18, alpha=0.35, beta=3,
                 criterion="von_mises", strengths={"yield": 50e6}),
            dict(carbon, name="Yuqori -30", theta=-30, is_visco=True, A=0.04, alpha=0.45, beta=4)
        ],
        "load": {"kind": "pulse", "spatial": "patch", "amplitude": 60,
                 "x": 0.15, "y": 0.1, "patch_width": 0.03, "patch_height": 0.03,
                 "duration": 0.003, "profile": "half_sine"},
        "simulation": {"total_time": 0.04, "dt": 2e-5, "w0_mm": 0, "v0": 0,
                       "memory_initial": "instantaneous", "kernel_tolerance": 2e-5}
    }


def ask_float(prompt, default=None, positive=False):
    while True:
        text = input(prompt+(f" [{default}]" if default is not None else "")+": ").strip()
        try:
            value = default if not text and default is not None else float(text)
            return finite(value, prompt, positive)
        except (TypeError, ValueError) as exc:
            print(f"Qayta kiriting: {exc}")


def ask_choice(prompt, choices, default):
    while True:
        val = input(f"{prompt} ({'/'.join(choices)}) [{default}]: ").strip().lower() or default
        if val in choices:
            return val
        print("Ro'yxatdagi variantni tanlang.")


def ask_int(prompt, default, minimum=1):
    while True:
        value = ask_float(prompt, default)
        if value == int(value) and value >= minimum:
            return int(value)
        print(f"{minimum} yoki kattaroq butun son kerak.")


def interactive_config():
    print("=== Ortotrop plastinka: MITC4 FSDT FEM + qatlamlar bo'yicha KR yadrolari ===")
    print("Birliklar: m, Pa, kg/m^3, s. Qatlamlar pastdan yuqoriga kiritiladi.")
    a, b = ask_float("Uzunlik a [m]", 0.3, True), ask_float("En b [m]", 0.2, True)
    n = ask_int("Qatlamlar soni", 3)
    layers = []
    for i in range(n):
        while True:
            print(f"\n--- {i+1}-qatlam ---")
            data = dict(name=f"Qatlam {i+1}", h=ask_float("Qalinlik [m]", 0.001, True),
                        theta=ask_float("Tola burchagi [gradus]", 0))
            isotropic = ask_choice("Material turi", ("izotrop", "ortotrop"), "ortotrop") == "izotrop"
            if isotropic:
                E = ask_float("Oniy E [Pa]", 2e9, True)
                nu = ask_float("nu (-1,0.5)", 0.35)
                if not -1 < nu < 0.5:
                    print("-1 < nu < 0.5 kerak.")
                    continue
                data.update(E1=E, E2=E, nu12=nu, G12=E/(2*(1+nu)),
                            G13=E/(2*(1+nu)), G23=E/(2*(1+nu)))
            else:
                for key, default in (("E1", 140e9), ("E2", 10e9), ("G12", 5e9),
                                     ("G13", 5e9), ("G23", 3e9)):
                    data[key] = ask_float(f"Oniy {key} [Pa]", default, True)
                data["nu12"] = ask_float("nu12", 0.3)
            data["rho"] = ask_float("Zichlik [kg/m^3]", 1600, True)
            data["is_visco"] = ask_choice("Viskoelastik qatlammi", ("y", "n"), "n") == "y"
            if data["is_visco"]:
                data.update(A=ask_float("KR A [s^(-alpha)]", 0.08),
                            alpha=ask_float("KR alpha (0,1)", 0.35),
                            beta=ask_float("KR beta [1/s]", 3, True))
            assess = ask_choice("Mustahkamlikni baholash", ("y", "n"), "y") == "y"
            if assess and isotropic:
                data.update(criterion="von_mises", strengths={"yield": ask_float("sigma_Y [Pa]", 50e6, True)})
            elif assess:
                strengths = {key: ask_float(f"{key} [Pa]", default, True) for key, default in
                             (("Xt", 1.2e9), ("Xc", 800e6), ("Yt", 50e6), ("Yc", 150e6),
                              ("S12", 60e6))}
                data.update(criterion="max_stress_plane", strengths=strengths)
            try:
                OrthotropicLayer(**data)
                layers.append(data)
                break
            except ValueError as exc:
                print(f"Qatlamni qayta kiriting: {exc}")
    nx, ny = ask_int("x bo'yicha elementlar", 8, 2), ask_int("y bo'yicha elementlar", 8, 2)
    edges = ask_boundaries()
    kw = ask_float("Vinkler asos koeffitsiyenti kw [N/m^3]", 0)
    kp = ask_float("Pasternak koeffitsiyenti kp [N/m]", 0)
    if kw < 0 or kp < 0:
        raise ValueError("kw, kp >= 0 kerak.")
    # Oddiy sharnirli tayanchda u,v ham cheklangan. Ekspert JSON parametrida
    # in_plane='normal' yoki 'free' alohida tanlanishi mumkin.
    in_plane = "fixed"
    kind = ask_choice("Yuk turi", ("point", "uniform", "harmonic", "pulse"), "pulse")
    load = {"kind": kind}
    spatial = "uniform" if kind == "uniform" else ask_choice(
        "Yuk fazoviy taqsimoti", ("point", "patch", "uniform"), "patch")
    if kind == "point" and spatial == "uniform":
        kind, load["kind"] = "uniform", "uniform"
    load["spatial"] = spatial
    load["amplitude"] = ask_float("Amplituda [Pa]" if spatial == "uniform" else "Kuch amplitudasi [N]", 60)
    if spatial != "uniform":
        load.update(x=ask_float("Ta'sir x [m]", a/2), y=ask_float("Ta'sir y [m]", b/2))
    if spatial == "patch":
        load.update(patch_width=ask_float("Kontakt eni x [m]", a/10, True),
                    patch_height=ask_float("Kontakt eni y [m]", b/10, True))
    if kind == "harmonic":
        load.update(frequency_hz=ask_float("Chastota [Hz]", 100, True),
                    phase_degrees=ask_float("Faza [gradus]", 0))
    if kind == "pulse":
        load.update(duration=ask_float("Zarb davomiyligi [s]", 0.003, True),
                    profile=ask_choice("Zarb shakli", ("half_sine", "triangle", "friedlander"), "half_sine"))
        if load["profile"] == "triangle":
            load["rise_fraction"] = ask_float("Maksimumga yetish vaqti / davomiylik", 0.1)
    sim = dict(total_time=ask_float("Kuzatuv vaqti [s]", 0.04, True),
               w0_mm=ask_float("Boshlang'ich markaziy ko'chish [mm]", 0),
               v0=ask_float("Boshlang'ich markaziy tezlik [m/s]", 0))
    dt = ask_float("dt [s], avtomatik tanlash uchun 0", 0)
    if dt < 0:
        raise ValueError("dt >= 0 kerak.")
    if dt:
        sim["dt"] = dt
    if sim["w0_mm"]:
        sim["memory_initial"] = ask_choice("Boshlang'ich material holati",
                                          ("instantaneous", "relaxed"), "instantaneous")
    return dict(a=a, b=b, nx=nx, ny=ny, layers=layers, edges=edges,
                in_plane=in_plane, kw=kw, kp=kp, load=load, simulation=sim)


def normalize_config(config):
    """Oldingi Ritz-SDOF JSON parametrlarini ham qabul qiladi; birliklar SI."""
    config = copy.deepcopy(config)
    config["edges"] = normalize_edges(config.get("edges", config.get("boundary_conditions")))
    if "k_shear" in config and "shear_correction" not in config:
        config["shear_correction"] = config["k_shear"]
    for i, data in enumerate(config["layers"]):
        data.setdefault("name", f"Qatlam {i+1}")
        strengths = data.get("strengths")
        if strengths and "criterion" not in data:
            if set(strengths) == {"sigma_y"}:
                data["criterion"] = "von_mises"
                data["strengths"] = {"yield": strengths["sigma_y"]}
            elif set(strengths) == {"Xt", "Xc", "Yt", "Yc", "S"}:
                data["criterion"] = "max_stress_plane"
                data["strengths"] = {("S12" if k == "S" else k): v
                                     for k, v in strengths.items()}
        elif strengths is None:
            data.pop("strengths", None)
    if "force" in config and "load" not in config:
        old = config["force"]
        kind = {"constant": "point", "pressure": "uniform", "pulse": "pulse",
                "harmonic": "harmonic"}.get(old["kind"])
        if kind is None:
            raise ValueError("Noma'lum force.kind.")
        config["load"] = dict(kind=kind, amplitude=old["amplitude"])
        if kind == "harmonic":
            config["load"]["frequency_hz"] = finite(old["omega"], "omega", True)/(2*np.pi)
        if kind == "pulse":
            config["load"]["duration"] = old["duration"]
    if "simulation" not in config:
        config["simulation"] = dict(total_time=config["t_sim"],
                                     w0_mm=1000*config.get("w0", 0), v0=config.get("v0", 0))
        if config.get("dt") is not None:
            config["simulation"]["dt"] = config["dt"]
    return config


def run_config(config, output="results", show=False, quiet=False, check_dt=False):
    config = normalize_config(config)
    layers = [OrthotropicLayer(**data) for data in config["layers"]]
    model = assemble_model(layers, config["a"], config["b"], config.get("nx", 8), config.get("ny", 8),
                           config.get("edges"), config.get("in_plane", "fixed"),
                           config.get("shear_correction", 5/6),
                           config.get("kw", 0.0), config.get("kp", 0.0))
    load = Load(**config["load"])
    load.vector(model.mesh)  # validate contact geometry BEFORE expensive simulation
    if not quiet:
        for edge in EDGE_NAMES:
            print(f"{EDGE_LABELS[edge]}: {BC_LABELS[model.edges[edge]]}")
        if all(value == "free" for value in model.edges.values()):
            print("Barcha chetlar erkin: tayanchsiz qattiq jism harakatlari saqlanadi.")
        print(f"To'r: {model.mesh.nx} x {model.mesh.ny}; erkin DOFlar: {len(model.free)}.")
        print(f"Elastik asos: kw={model.kw:g} N/m^3, kp={model.kp:g} N/m")
        print("Hisoblash boshlandi...")
    result = simulate(model, load, **config["simulation"],
                      progress=None if quiet else lambda x: print(f"{100*x:.0f}% ", end="", flush=True))
    save_results(model, result, output, show)
    if check_dt:
        settings = copy.deepcopy(config["simulation"])
        settings.update(dt=result.dt/2, calculate_stress=False, record_energy=False)
        fine = simulate(model, load, **settings)
        reference = np.interp(result.t, fine.t, fine.center_w)
        relative_error = float(np.max(np.abs(result.center_w-reference)) /
                               max(np.max(np.abs(fine.center_w)), 1e-30))
        data = {"dt_s": result.dt, "half_dt_s": fine.dt,
                "center_displacement_relative_error": relative_error,
                "within_1_percent": bool(relative_error <= .01),
                "scope": "Time step only; mesh and stress convergence not assessed."}
        Path(output, "time_convergence.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8")
        print(f"dt/2 tekshiruvi: markaziy ko'chish farqi {relative_error:.3%}.")
    print("\nNatijalar:")
    print("Oniy elastik chastotalar [Hz]:", np.round(result.frequencies[:3], 5))
    print("Relaksatsiyalangan elastik chastotalar [Hz]:", np.round(result.relaxed_frequencies[:3], 5))
    print(f"Qattiq jism modalarining soni: {result.rigid_modes}")
    print(f"dt = {result.dt:.6g} s; qadamlar = {len(result.t)-1}")
    p = result.peak
    print(f"Maksimal |w| = {p['abs_w_m']*1000:.6g} mm; "
          f"(x,y)=({p['x_m']:.6g},{p['y_m']:.6g}) m; t={p['time_s']:.6g} s")
    for report in result.stress_reports:
        fi = report["max_failure_index"]
        print(f"Qatlam {report['layer']} ({report['name']}): "
              f"max Mises={report['max_von_mises_pa']/1e6:.6g} MPa; "
              f"FI={'berilmagan' if fi is None else f'{fi:.6g}'}; {report['assessment']}")
    for report in result.kernel_reports:
        if report["branches"]:
            print(f"Yadro {report['name']}: {report['branches']} eksponenta; "
                  f"relaksatsiya xatosi <= {report['max_relaxation_error']:.3g} (tekshiruv nuqtalarida).")
    if result.harmonic:
        print("Garmonik barqaror javob:", result.harmonic)
    if load.spatial == "point":
        print("ESLATMA: ideal nuqtaviy yuk yaqinidagi mahalliy mustahkamlik xulosasi aniqlanmagan.")
    if result.peak["abs_w_m"] > sum(layer.h for layer in layers)/2:
        print("ESLATMA: salqilik katta; geometrik chiziqli modelni tekshirish kerak.")
    print(f"Natijalar: {Path(output).resolve()}")
    return model, result


# =============================================================================
# CLPT: CLASSICAL LAMINATE PLATE THEORY / RITZ
# (Original CLPT solver is kept mathematically separate from FSDT.)
# =============================================================================
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh, cho_factor, cho_solve
from scipy.special import gamma, gammainc, gammaincc

@dataclass(frozen=True)
class Layer:
    h: float                # m
    theta: float            # degrees
    E1: float               # Pa
    E2: float               # Pa
    G12: float              # Pa
    nu12: float
    rho: float              # kg/m^3
    is_visco: bool = False
    A: float = 0.0          # s**(-alpha), if visco
    alpha: float = 0.5      # 0 < alpha < 1
    beta: float = 1.0       # 1/s

    def __post_init__(self):
        scalars = (self.h, self.theta, self.E1, self.E2, self.G12,
                   self.nu12, self.rho, self.A, self.alpha, self.beta)
        if not all(np.isfinite(x) for x in scalars):
            raise ValueError("Barcha material parametrlari chekli son bo'lishi kerak.")
        if any(x <= 0 for x in (self.h, self.E1, self.E2, self.G12, self.rho)):
            raise ValueError("h, E1, E2, G12 va rho musbat bo'lishi kerak.")
        if 1.0 - self.nu12**2 * self.E2 / self.E1 <= 0:
            raise ValueError("1 - nu12*nu21 > 0 bo'lishi shart.")
        if self.is_visco:
            if self.A < 0 or not (0 < self.alpha < 1) or self.beta <= 0:
                raise ValueError("RK: A>=0, 0<alpha<1, beta>0 bo'lishi kerak.")
            relaxation_strength = self.A * gamma(self.alpha) / self.beta**self.alpha
            if not np.isfinite(relaxation_strength) or relaxation_strength >= 1:
                raise ValueError("RK: A*Gamma(alpha)/beta**alpha < 1 bo'lishi shart.")

    def qbar(self):
        den = 1 - self.nu12**2 * self.E2/self.E1
        q11 = self.E1 / den
        q22 = self.E2 / den
        q12 = self.nu12 * self.E2 / den
        q66 = self.G12
        m = np.cos(np.deg2rad(self.theta))
        n = np.sin(np.deg2rad(self.theta))
        q11b = q11*m**4 + 2*(q12+2*q66)*m*m*n*n + q22*n**4
        q22b = q11*n**4 + 2*(q12+2*q66)*m*m*n*n + q22*m**4
        q12b = (q11+q22-4*q66)*m*m*n*n + q12*(m**4+n**4)
        q66b = (q11+q22-2*q12-2*q66)*m*m*n*n + q66*(m**4+n**4)
        q16b = (q11-q12-2*q66)*m**3*n - (q22-q12-2*q66)*m*n**3
        q26b = (q11-q12-2*q66)*m*n**3 - (q22-q12-2*q66)*m**3*n
        return np.array([[q11b, q12b, q16b],
                         [q12b, q22b, q26b],
                         [q16b, q26b, q66b]])


def rk_cell_weights(layer, dt, steps):
    """Exact weights for memory cells [k*dt,(k+1)*dt]."""
    if not layer.is_visco or layer.A == 0:
        return np.zeros(steps)
    x0 = layer.beta * dt * np.arange(steps, dtype=float)
    x1 = layer.beta * dt * (np.arange(steps, dtype=float)+1)
    # Upper incomplete gamma avoids cancellation when arguments are large.
    diff = np.where(x0 < 1.0,
                    gammainc(layer.alpha, x1)-gammainc(layer.alpha, x0),
                    gammaincc(layer.alpha, x0)-gammaincc(layer.alpha, x1))
    return (layer.A * gamma(layer.alpha)/layer.beta**layer.alpha) * np.maximum(diff, 0.0)


def _basis_1d(x, L, n_modes, bc_pair, field='w'):
    """1D Ritz basis functions and 1st/2nd derivatives on [0, L].

    - For 'SS' pair, uses exact Navier trigonometric basis (sin/cos) for
      machine-precision compatibility with classical Navier solutions.
    - For any pair containing 'C' (Clamped) or 'F' (Free), constructs
      Gram-Schmidt L2-orthonormalized boundary polynomials (Bhat method):
        phi_k(xi) = xi^p * (1 - xi)^q * P_k(xi),  xi = x / L in [0, 1]
      where (p, q) enforce essential geometric BCs at x=0 and x=L:
        field='w'      : F -> 0, S -> 1 (w=0), C -> 2 (w=0, w'=0)
        field='u_norm' : F/S -> 0 (movable normal), C -> 1 (fixed normal)
        field='u_tang' : F -> 0, S/C -> 1 (tangential displacement = 0)
    """
    bc_pair = bc_pair.upper()
    if bc_pair == 'SS':
        m_idx = np.arange(1, n_modes + 1, dtype=float)[None, :]
        k = np.pi * m_idx / L
        kx = x[:, None] * k
        if field in ('w', 'u_tang'):
            phi = np.sin(kx)
            dphi = k * np.cos(kx)
            d2phi = -(k ** 2) * np.sin(kx)
        else:  # 'u_norm'
            phi = np.cos(kx)
            dphi = -k * np.sin(kx)
            d2phi = -(k ** 2) * np.cos(kx)
        return phi, dphi, d2phi

    left, right = bc_pair[0], bc_pair[1]
    if field == 'w':
        p_map = {'F': 0, 'S': 1, 'C': 2}
        p, q = p_map[left], p_map[right]
    elif field == 'u_norm':
        p = 1 if left == 'C' else 0
        q = 1 if right == 'C' else 0
    else:  # 'u_tang'
        p = 0 if left == 'F' else 1
        q = 0 if right == 'F' else 1

    gq = max(48, 4 * (n_modes + p + q + 2))
    xg, wg = leggauss(gq)
    xig = 0.5 * (xg + 1.0)
    wg = 0.5 * wg

    base = (Polynomial([0.0, 1.0]) ** p) * (Polynomial([1.0, -1.0]) ** q)
    polys = []
    for k in range(n_modes):
        cand = base * (Polynomial([0.0, 1.0]) ** k)
        for _ in range(2):
            for prev in polys:
                ip = np.sum(wg * cand(xig) * prev(xig))
                cand = cand - ip * prev
        norm = np.sqrt(np.sum(wg * (cand(xig) ** 2)))
        cand = cand / norm
        polys.append(cand)

    xi = x / L
    phi = np.zeros((x.size, n_modes))
    dphi = np.zeros((x.size, n_modes))
    d2phi = np.zeros((x.size, n_modes))
    for k, P in enumerate(polys):
        dP = P.deriv(1)
        d2P = P.deriv(2)
        phi[:, k] = P(xi)
        dphi[:, k] = dP(xi) / L
        d2phi[:, k] = d2P(xi) / (L ** 2)
    return phi, dphi, d2phi


class CLPTPlate:
    """Rectangular laminate plate with general (S, C, F) BCs and Winkler/Pasternak foundation.

    Parameters
    ----------
    bc : str, default 'SSSS'
        4-character boundary condition string in standard cyclic order (x=0, y=0, x=a, y=b).
        Each character can be:
          'S' - Sharnirli (Simply supported)
          'C' - Qistirilgan (Clamped)
          'F' - Erkin (Free)
        Examples: 'SSSS', 'CCCC', 'SCSC', 'SFSF', 'CFFF' (cantilever).
    kw : float, default 0.0
        Winkler elastic foundation stiffness (N/m^3).
    kp : float, default 0.0
        Pasternak shear foundation parameter (N/m).
    """
    def __init__(self, a, b, layers, nx=2, ny=2, gauss_order=None,
                 rotary_inertia=False, bc='SSSS', kw=0.0, kp=0.0):
        if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
            raise ValueError("Plastinka a va b o'lchamlari musbat bo'lishi kerak.")
        if not layers or any(not isinstance(x, Layer) for x in layers):
            raise ValueError("layers bo'sh bo'lmasin va Layer elementlaridan iborat bo'lsin.")
        if not isinstance(nx, int) or not isinstance(ny, int) or nx < 1 or ny < 1:
            raise ValueError("nx va ny musbat butun son bo'lishi kerak.")
        if not isinstance(bc, str) or len(bc) != 4 or any(c.upper() not in 'SCF' for c in bc):
            raise ValueError("bc 4 ta belgidan ('S', 'C', 'F') iborat bo'lishi kerak (masalan, 'SSSS', 'CCCC', 'SCSC').")
        if not np.isfinite(kw) or kw < 0 or not np.isfinite(kp) or kp < 0:
            raise ValueError("Vinkler (kw) va Pasternak (kp) koeffitsiyentlari manfiy bo'lmasligi kerak.")

        self.a, self.b, self.layers = float(a), float(b), list(layers)
        self.nx, self.ny = nx, ny
        self.bc = bc.upper()
        self.kw, self.kp = float(kw), float(kp)
        self.modes = [(m,n) for m in range(1,nx+1) for n in range(1,ny+1)]
        self.ndof = 3*len(self.modes)
        self.h = sum(ly.h for ly in layers)
        self.rotary_inertia = bool(rotary_inertia)
        g = gauss_order if gauss_order is not None else max(20, 7*max(nx,ny))
        if not isinstance(g,int) or g < 2:
            raise ValueError("gauss_order >= 2 bo'lishi kerak.")

        xx, wx = leggauss(g)
        yy, wy = leggauss(g)
        X, Y = np.meshgrid((xx+1)*a/2, (yy+1)*b/2, indexing='ij')
        W = np.outer(wx*a/2, wy*b/2).ravel()
        self.xq, self.yq, self.wq = X.ravel(), Y.ravel(), W
        p = W.size
        d = self.ndof
        E0 = np.zeros((p,3,d))
        Kap = np.zeros((p,3,d))
        U = np.zeros((p,d)); V = np.zeros((p,d)); Z = np.zeros((p,d))
        Wx = np.zeros((p,d)); Wy = np.zeros((p,d))

        # --- YANGI QISM 1: Umumiy chegara shartlari (S, C, F) uchun 1D bazislar ---
        bc_x, bc_y = self.bc[0] + self.bc[2], self.bc[1] + self.bc[3]
        wx_phi, wx_dphi, wx_d2phi = _basis_1d(self.xq, a, nx, bc_x, field='w')
        wy_phi, wy_dphi, wy_d2phi = _basis_1d(self.yq, b, ny, bc_y, field='w')

        ux_phi, ux_dphi, _ = _basis_1d(self.xq, a, nx, bc_x, field='u_norm')
        uy_phi, uy_dphi, _ = _basis_1d(self.yq, b, ny, bc_y, field='u_tang')

        vx_phi, vx_dphi, _ = _basis_1d(self.xq, a, nx, bc_x, field='u_tang')
        vy_phi, vy_dphi, _ = _basis_1d(self.yq, b, ny, bc_y, field='u_norm')

        for j,(m,n) in enumerate(self.modes):
            im, in_ = m - 1, n - 1
            u, v, z = 3*j, 3*j+1, 3*j+2

            U[:,u] = ux_phi[:,im] * uy_phi[:,in_]
            V[:,v] = vx_phi[:,im] * vy_phi[:,in_]
            Z[:,z] = wx_phi[:,im] * wy_phi[:,in_]

            Wx[:,z] = wx_dphi[:,im] * wy_phi[:,in_]
            Wy[:,z] = wx_phi[:,im] * wy_dphi[:,in_]

            E0[:,0,u] = ux_dphi[:,im] * uy_phi[:,in_]
            E0[:,1,v] = vx_phi[:,im] * vy_dphi[:,in_]
            E0[:,2,u] = ux_phi[:,im] * uy_dphi[:,in_]
            E0[:,2,v] = vx_dphi[:,im] * vy_phi[:,in_]

            Kap[:,0,z] = -wx_d2phi[:,im] * wy_phi[:,in_]
            Kap[:,1,z] = -wx_phi[:,im] * wy_d2phi[:,in_]
            Kap[:,2,z] = -2.0 * wx_dphi[:,im] * wy_dphi[:,in_]

        self._E0, self._Kap = E0, Kap
        self._U, self._V, self._Z, self._Wx, self._Wy = U,V,Z,Wx,Wy

        def gram(X,Y):
            return X.T @ (W[:,None]*Y)

        M0 = gram(U,U)+gram(V,V)+gram(Z,Z)
        I0=I1=I2=0.0
        self.K_layers=[]
        self.ABD_layers=[]
        zb=-self.h/2
        for ly in self.layers:
            zt=zb+ly.h
            dz=zt-zb
            dz2=(zt**2-zb**2)/2
            dz3=(zt**3-zb**3)/3
            Q=ly.qbar()
            Am=Q*dz; Bm=Q*dz2; Dm=Q*dz3
            self.ABD_layers.append((Am,Bm,Dm))
            # Energy 1/2*integral [eps0;kappa]^T [A B;B D] [eps0;kappa] dArea
            k = np.einsum('pik,ij,pjl,p->kl',E0,Am,E0,W,optimize=True)
            k += np.einsum('pik,ij,pjl,p->kl',E0,Bm,Kap,W,optimize=True)
            k += np.einsum('pik,ij,pjl,p->kl',Kap,Bm,E0,W,optimize=True)
            k += np.einsum('pik,ij,pjl,p->kl',Kap,Dm,Kap,W,optimize=True)
            self.K_layers.append((k+k.T)*0.5)
            I0 += ly.rho*dz
            I1 += ly.rho*dz2
            I2 += ly.rho*dz3
            zb=zt

        # --- YANGI QISM 2: Vinkler (kw) va Pasternak (kp) elastik asosi ---
        self.K_winkler = self.kw * gram(Z, Z)
        self.K_pasternak = self.kp * (gram(Wx, Wx) + gram(Wy, Wy))
        self.K0 = sum(self.K_layers) + self.K_winkler + self.K_pasternak
        self.K0 = (self.K0 + self.K0.T) * 0.5

        self.Amat=sum(v[0] for v in self.ABD_layers)
        self.Bmat=sum(v[1] for v in self.ABD_layers)
        self.Dmat=sum(v[2] for v in self.ABD_layers)
        self.M=I0*M0
        if self.rotary_inertia:
            self.M-=I1*(gram(U,Wx)+gram(Wx,U)+gram(V,Wy)+gram(Wy,V))
            self.M+=I2*(gram(Wx,Wx)+gram(Wy,Wy))
        self.M=(self.M+self.M.T)*0.5
        if self.bc != 'FFFF' or self.kw > 0:
            if np.linalg.eigvalsh(self.K0)[0] <= 0 or np.linalg.eigvalsh(self.M)[0] <= 0:
                raise ArithmeticError("Bikrlik yoki massa matritsasi musbat aniqlangan emas.")

    def frequencies(self, count=6):
        eigvals,eigvecs=eigh(self.K0,self.M)
        eigvals = np.maximum(eigvals, 0.0)
        return np.sqrt(eigvals[:count])/(2*np.pi), eigvecs[:,:count]

    def eval_w(self, q, x_pt, y_pt):
        """Evaluate deflection w(x_pt, y_pt) for arbitrary BCs."""
        x_arr = np.atleast_1d(float(x_pt))
        y_arr = np.atleast_1d(float(y_pt))
        wx_p, _, _ = _basis_1d(x_arr, self.a, self.nx, self.bc[0] + self.bc[2], field='w')
        wy_p, _, _ = _basis_1d(y_arr, self.b, self.ny, self.bc[1] + self.bc[3], field='w')
        basis = np.zeros(self.ndof)
        for j,(m,n) in enumerate(self.modes):
            basis[3*j+2] = wx_p[0, m-1] * wy_p[0, n-1]
        return np.asarray(q) @ basis

    def center_w(self, q):
        return self.eval_w(q, self.a/2.0, self.b/2.0)

    def history_response(self, time_end=0.06, dt=2e-4, initial_amplitude=1e-4,
                         initial_q=None, initial_v=None, external_force=None):
        """Solve M*qddot + K0*q - sum_l K_l * (R_l*q) = f(t).

        Right-endpoint history rule is 1st order for q(t), kernel cell
        integrals are exact. Newmark gamma=1/2, beta=1/4.
        external_force(t) -> vector length ndof (optional).
        """
        if time_end <= 0 or dt <= 0:
            raise ValueError("time_end va dt musbat bo'lishi kerak.")
        steps=int(np.ceil(time_end/dt))
        if steps > 10000:
            raise ValueError("Oddiy O(N^2) xotira integrali uchun <=10000 qadam tavsiya etiladi.")
        dt=time_end/steps
        t=np.arange(steps+1)*dt
        d=self.ndof
        q=np.zeros((steps+1,d)); v=np.zeros_like(q); acc=np.zeros_like(q)
        if initial_q is not None:
            q[0]=np.asarray(initial_q,dtype=float)
        else:
            q[0,2]=initial_amplitude # (m,n)=(1,1) mode
        if initial_v is not None:
            v[0]=np.asarray(initial_v,dtype=float)
        if q[0].shape != (d,) or v[0].shape != (d,):
            raise ValueError("Boshlang'ich vektor o'lchami ndof bo'lishi kerak.")
        def force(tval):
            if external_force is None:
                return np.zeros(d)
            f=np.asarray(external_force(tval),dtype=float)
            if f.shape!=(d,):
                raise ValueError("external_force ndof uzunlikdagi vektor qaytarsin.")
            return f
        f0=force(0.0)
        acc[0]=np.linalg.solve(self.M, f0-self.K0@q[0])
        kernels=[(self.K_layers[k],rk_cell_weights(ly,dt,steps))
                 for k,ly in enumerate(self.layers) if ly.is_visco and ly.A>0]
        Kinstant=self.K0.copy()
        for Kl,weights in kernels:
            Kinstant-=weights[0]*Kl
        bm=0.25; gm=0.5
        factor=cho_factor(self.M+bm*dt**2*Kinstant,lower=True,check_finite=False)
        # Cache K_l @ q_i for the memory sum; only filled history used.
        project=[np.zeros_like(q) for _ in kernels]
        for k,(Kl,_) in enumerate(kernels):
            project[k][0]=Kl@q[0]
        for j in range(1,steps+1):
            qpred=q[j-1]+dt*v[j-1]+dt**2*(0.5-bm)*acc[j-1]
            vpred=v[j-1]+dt*(1-gm)*acc[j-1]
            rhs=force(t[j])
            # history at t_j: w_0*q_j + w_1*q_(j-1) + ... + w_(j-1)*q_1
            for k,(_,weights) in enumerate(kernels):
                if j>1:
                    rhs+=weights[1:j]@project[k][j-1:0:-1]
            acc[j]=cho_solve(factor,rhs-Kinstant@qpred,check_finite=False)
            q[j]=qpred+bm*dt**2*acc[j]
            v[j]=vpred+gm*dt*acc[j]
            for k,(Kl,_) in enumerate(kernels):
                project[k][j]=Kl@q[j]
        return t,q,v,acc


def validate_clpt():
    """Five automatic consistency checks: Navier SSSS, B-matrix, RK memory,
    Winkler/Pasternak analytical frequency shift, and Clamped/Free BCs (Leissa benchmarks)."""
    E1=130e9; E2=10e9; G12=5e9; nu=.25; rho=1600.; h=0.01; a=1.; b=0.8
    lay=Layer(h,0,E1,E2,G12,nu,rho)
    p=CLPTPlate(a,b,[lay],nx=1,ny=1,gauss_order=24,bc='SSSS')
    freq,_=p.frequencies(3)
    q=lay.qbar(); D=q*h**3/12
    kx=np.pi/a; ky=np.pi/b
    omega2=(D[0,0]*kx**4+2*(D[0,1]+2*D[2,2])*kx*kx*ky*ky+D[1,1]*ky**4)/(rho*h)
    predicted=np.sqrt(omega2)/(2*np.pi)
    err=abs(freq[0]-predicted)/predicted
    assert err<1e-10, f"Analitik Navier tekshiruvi: nisbiy xato {err}"

    sym=CLPTPlate(1,1,[Layer(h/4,0,E1,E2,G12,nu,rho),
                       Layer(h/2,90,E1,E2,G12,nu,rho),
                       Layer(h/4,0,E1,E2,G12,nu,rho)],nx=2,ny=2)
    assert np.linalg.norm(sym.Bmat) < 1e-10*np.linalg.norm(sym.Dmat)/h
    unsym=CLPTPlate(1,1,[Layer(h/2,0,E1,E2,G12,nu,rho),
                         Layer(h/2,90,E1,E2,G12,nu,rho)],nx=2,ny=2)
    assert np.linalg.norm(unsym.Bmat)>1.0

    vis=CLPTPlate(a,b,[Layer(h,0,E1,E2,G12,nu,rho,True,.10,.5,5.)],nx=1,ny=1)
    te,qe,_,_=p.history_response(time_end=0.02,dt=2e-4)
    tv,qv,_,_=vis.history_response(time_end=0.02,dt=2e-4)
    assert np.max(np.abs(qe-qv))>1e-10

    # TEST 4: Vinkler va Pasternak asosi analitik siljishini tekshirish
    kw_test = 2.5e7  # N/m^3
    kp_test = 1.2e5  # N/m
    p_wink = CLPTPlate(a, b, [lay], nx=1, ny=1, gauss_order=24, bc='SSSS', kw=kw_test, kp=kp_test)
    f_wink, _ = p_wink.frequencies(1)
    omega2_wink = omega2 + (kw_test + kp_test * (kx**2 + ky**2)) / (rho * h)
    pred_wink = np.sqrt(omega2_wink) / (2 * np.pi)
    err_wink = abs(f_wink[0] - pred_wink) / pred_wink
    assert err_wink < 1e-10, f"Vinkler/Pasternak analitik tekshiruvi: xato {err_wink}"

    # TEST 5: Qistirilgan (C) va Erkin (F) chetlar — Leissa (1969) izotrop kvadrat plastinka etalonlari
    E_iso = 200e9; nu_iso = 0.3; G_iso = E_iso / (2*(1+nu_iso))
    lay_iso = Layer(h, 0, E_iso, E_iso, G_iso, nu_iso, rho)
    D_iso = E_iso * h**3 / (12 * (1 - nu_iso**2))
    scale = (1.0**2) * np.sqrt(rho * h / D_iso)
    # Leissa (1969) 1-moda o'lchamsiz chastotalari: CCCC=35.985, SCSC=28.951, SFSF=9.631
    benchmarks = {'CCCC': 35.9852, 'SCSC': 28.9509, 'SFSF': 9.6314}
    bc_results = {}
    for bc_code, ref_val in benchmarks.items():
        pl_bc = CLPTPlate(1.0, 1.0, [lay_iso], nx=6, ny=6, bc=bc_code)
        f_bc, _ = pl_bc.frequencies(1)
        om_bar = 2 * np.pi * f_bc[0] * scale
        rel_err = abs(om_bar - ref_val) / ref_val
        assert rel_err < 2e-3, f"{bc_code} Leissa etalonidan farq: {rel_err}"
        bc_results[bc_code] = (om_bar, ref_val, rel_err)

    print(f"TEST 1 | Navier analitik / Ritz (SSSS): {predicted:.8f} / {freq[0]:.8f} Hz; xato={err:.2e}")
    print("TEST 2 | Simmetrik B≈0; nosimmetrik B≠0: OK")
    print("TEST 3 | RK xotira elastik javobni o'zgartiradi: OK")
    print(f"TEST 4 | Vinkler+Pasternak analitik / Ritz: {pred_wink:.8f} / {f_wink[0]:.8f} Hz; xato={err_wink:.2e}")
    for bc_code, (val, ref, rerr) in bc_results.items():
        print(f"TEST 5 | {bc_code} Leissa etalon / Ritz: {ref:.4f} / {val:.4f} (nisbiy farq={rerr:.2e})")


def example_clpt():
    # Change values here for your own laminate, boundary conditions, and Winkler foundation.
    E1,E2,G12,nu12,rho=130e9,10e9,5e9,0.25,1600.0
    layers=[
        Layer(h=0.0025,theta=0,E1=E1,E2=E2,G12=G12,nu12=nu12,rho=rho),
        Layer(h=0.0050,theta=90,E1=E1,E2=E2,G12=G12,nu12=nu12,rho=rho,
              is_visco=True,A=0.12,alpha=0.5,beta=5.0),
        Layer(h=0.0025,theta=0,E1=E1,E2=E2,G12=G12,nu12=nu12,rho=rho),
    ]
    print("=== CLPT + Rjanitsin–Koltunov + Vinkler asosi: [0/90/0] kompozit plastinka ===")
    for bc_name in ['SSSS', 'SCSC', 'CCCC', 'SFSF']:
        pl_0 = CLPTPlate(a=1.0, b=1.0, layers=layers, nx=5, ny=5, bc=bc_name, kw=0.0)
        pl_w = CLPTPlate(a=1.0, b=1.0, layers=layers, nx=5, ny=5, bc=bc_name, kw=1e7)
        f0, _ = pl_0.frequencies(count=1)
        fw, _ = pl_w.frequencies(count=1)
        print(f"  Chegara {bc_name} | Vinklersiz f1 = {f0[0]:7.3f} Hz | Vinkler (kw=1e7 N/m^3) f1 = {fw[0]:7.3f} Hz")

    # Vaqt bo'yicha javob: CCCC (qistirilgan) va Vinkler asosidagi plastinka
    plate=CLPTPlate(a=1.0,b=1.0,layers=layers,nx=4,ny=4,
                    rotary_inertia=False,bc='CCCC',kw=1e7,kp=0.0)
    t,q,_,_=plate.history_response(time_end=0.08,dt=0.0002,initial_amplitude=1e-4)

    elastic_layers=[Layer(h=ly.h,theta=ly.theta,E1=ly.E1,E2=ly.E2,
                          G12=ly.G12,nu12=ly.nu12,rho=ly.rho)
                    for ly in layers]
    el_plate=CLPTPlate(plate.a,plate.b,elastic_layers,nx=plate.nx,ny=plate.ny,
                       bc=plate.bc,kw=plate.kw,kp=plate.kp)
    _,qe,_,_=el_plate.history_response(time_end=0.08,dt=0.0002,
                                       initial_amplitude=1e-4)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9,4.5))
    plt.plot(t,1e3*plate.center_w(q),label=f'RK + Vinkler ({plate.bc}, kw={plate.kw:.0e})',lw=1.8)
    plt.plot(t,1e3*el_plate.center_w(qe),label=f'Sof elastik + Vinkler ({plate.bc})',ls='--',lw=1.3)
    plt.xlabel('Vaqt, s');plt.ylabel('Markaziy egilish w(a/2,b/2), mm')
    plt.title(f'CLPT ({plate.bc}) + Vinkler asosi: plastinka erkin tebranishi')
    plt.grid(alpha=0.3);plt.legend();plt.tight_layout()
    plt.savefig('clpt_rjanitsin_koltunov_response.png',dpi=170)
    plt.show()
    print("Grafik: clpt_rjanitsin_koltunov_response.png")
    return plate,t,q


# =============================================================================
# SHARED CONFIGURATION AND SOLVER DISPATCH
# =============================================================================


def _clpt_materials(config):
    """FSDT's richer layer description -> CLPT's in-plane layer description.

    Shear moduli G13/G23, strengths and layer names are FSDT-only properties.
    All parameters used by CLPT are taken directly from the same JSON layers.
    """
    fields = ("h", "theta", "E1", "E2", "G12", "nu12", "rho",
              "is_visco", "A", "alpha", "beta")
    return [Layer(**{key: value for key, value in data.items() if key in fields})
            for data in config["layers"]]


def _clpt_shape_vector(plate, x, y):
    """Column coupling q to a unit transverse point force at (x,y)."""
    if not (0 <= x <= plate.a and 0 <= y <= plate.b):
        raise ValueError("CLPT: yuk nuqtasi plastinka tashqarisida.")
    px = _basis_1d(np.array([x]), plate.a, plate.nx,
                   plate.bc[0]+plate.bc[2], field="w")[0][0]
    py = _basis_1d(np.array([y]), plate.b, plate.ny,
                   plate.bc[1]+plate.bc[3], field="w")[0][0]
    shape_vector = np.zeros(plate.ndof)
    for k, (m, n) in enumerate(plate.modes):
        shape_vector[3*k+2] = px[m-1]*py[n-1]
    return shape_vector


def _clpt_load_vector(plate, load):
    """Consistent Ritz generalized load for a force, uniform pressure or patch.

    Unit point/patch vectors correspond to 1 N; uniform corresponds to 1 Pa.
    """
    if load.spatial == "uniform":
        return plate._Z.T @ plate.wq
    x = plate.a/2 if load.x is None else load.x
    y = plate.b/2 if load.y is None else load.y
    if load.spatial == "point":
        return _clpt_shape_vector(plate, x, y)
    wx, wy = finite(load.patch_width, "patch_width", True), finite(load.patch_height, "patch_height", True)
    x0, x1, y0, y1 = x-wx/2, x+wx/2, y-wy/2, y+wy/2
    if min(x0, y0) < 0 or x1 > plate.a or y1 > plate.b:
        raise ValueError("CLPT: kontakt yuzasi plastinka ichida bo'lishi kerak.")
    ng = max(12, 3*max(plate.nx, plate.ny))
    gp, gw = leggauss(ng)
    xx = (x0+x1)/2 + (wx/2)*gp
    yy = (y0+y1)/2 + (wy/2)*gp
    px = _basis_1d(xx, plate.a, plate.nx,
                   plate.bc[0]+plate.bc[2], field="w")[0]
    py = _basis_1d(yy, plate.b, plate.ny,
                   plate.bc[1]+plate.bc[3], field="w")[0]
    out = np.zeros(plate.ndof)
    for k, (m, n) in enumerate(plate.modes):
        # Average of the test function over the contact surface.
        out[3*k+2] = ((gw @ px[:, m-1])/2)*((gw @ py[:, n-1])/2)
    return out


def _clpt_relaxed_frequencies(plate, count=3):
    k = plate.K_winkler + plate.K_pasternak
    for layer, stiffness in zip(plate.layers, plate.K_layers):
        chi = (layer.A * gamma(layer.alpha) / layer.beta**layer.alpha
               if layer.is_visco and layer.A else 0.0)
        k = k + (1-chi)*stiffness
    eig = eigh((k+k.T)/2, plate.M, eigvals_only=True)
    return np.sqrt(np.maximum(eig[:count], 0))/(2*np.pi)


def _clpt_harmonic(plate, load, spatial):
    if load.kind != "harmonic":
        return None
    w = 2*np.pi*load.frequency_hz
    dynamic = (plate.K_winkler + plate.K_pasternak).astype(complex) - w*w*plate.M
    for layer, stiffness in zip(plate.layers, plate.K_layers):
        chi = (layer.A * gamma(layer.alpha) / layer.beta**layer.alpha
               if layer.is_visco and layer.A else 0.0)
        factor = 1.0-chi*(layer.beta/(layer.beta+1j*w))**layer.alpha if chi else 1.0
        dynamic += factor*stiffness
    try:
        q = np.linalg.solve(dynamic, spatial*load.amplitude)
    except np.linalg.LinAlgError:
        return {"status": "singular_resonance", "center_amplitude_mm": None}
    value = _clpt_shape_vector(plate, plate.a/2, plate.b/2) @ q
    return {"status": "steady_state", "center_amplitude_mm": float(abs(value)*1000),
            "phase_relative_to_force_degrees": float(np.degrees(np.angle(value)))}


def run_clpt_config(config, output="results_clpt", show=False, quiet=False,
                    check_dt=False):
    """Run CLPT on the *same* material geometry, BC, elastic base and load as FSDT."""
    config = normalize_config(config)
    layers = _clpt_materials(config)
    edges = normalize_edges(config["edges"])
    bc = "".join("C" if edges[k] == "clamped" else
                 "S" if edges[k] == "simply_supported" else "F"
                 for k in ("left", "bottom", "right", "top"))
    kw, kp = config.get("kw", 0.0), config.get("kp", 0.0)
    # CLPT nx,ny are Ritz modes; FEM nx,ny are element counts. Choose separately
    # for convergence without silently interpreting 8x8 elements as 8x8 Ritz modes.
    opts = config.get("clpt", {})
    plate = CLPTPlate(config["a"], config["b"], layers,
                      nx=opts.get("nx", min(4, config.get("nx", 8))),
                      ny=opts.get("ny", min(4, config.get("ny", 8))),
                      gauss_order=opts.get("gauss_order"),
                      rotary_inertia=opts.get("rotary_inertia", False),
                      bc=bc, kw=kw, kp=kp)
    load = Load(**config.get("load", {"kind": "point", "amplitude": 0.0}))
    spatial = _clpt_load_vector(plate, load)
    if not np.any(spatial) and load.amplitude != 0:
        raise ValueError("CLPT: yuk tanlangan Ritz modalarni qo'zg'atmaydi.")
    sim = config.get("simulation", {})
    total_time = finite(sim.get("total_time", 0.04), "total_time", True)
    w0 = finite(sim.get("w0_mm", 0.0), "w0_mm")*0.001
    v0 = finite(sim.get("v0", 0.0), "v0")
    if sim.get("memory_initial", "instantaneous") != "instantaneous" and (w0 or v0):
        raise ValueError("CLPT usulida faqat memory_initial='instantaneous' qo'llanadi.")
    frequencies, eigenvectors = plate.frequencies(count=min(plate.ndof, 8))
    if len(frequencies) == 0:
        raise ValueError("CLPT: chastotalar hisoblanmadi.")
    dt = sim.get("dt")
    if dt is None:
        positive = frequencies[frequencies > 1e-10]
        dt = min(total_time/200, 1/(40*positive[0])) if len(positive) else total_time/200
        if load.kind == "harmonic":
            dt = min(dt, 1/(60*load.frequency_hz))
        if load.kind == "pulse":
            dt = min(dt, load.duration/60)
    dt = finite(dt, "dt", True)
    center_shape = _clpt_shape_vector(plate, plate.a/2, plate.b/2)
    q0 = np.zeros(plate.ndof)
    vel0 = np.zeros(plate.ndof)
    if w0 or v0:
        for eigvec in eigenvectors.T:
            center_amp = center_shape @ eigvec
            if abs(center_amp) > 1e-10*np.max(np.abs(eigvec)):
                normalized = eigvec / center_amp
                q0 = normalized*w0
                vel0 = normalized*v0
                break
        else:
            raise ValueError("CLPT: markazda flexural boshlang'ich moda topilmadi.")
    if not quiet:
        print(f"\n=== CLPT Ritz | BC={bc}, Ritz={plate.nx}x{plate.ny}, kw={kw:g}, kp={kp:g} ===")
    t, q, velocities, accelerations = plate.history_response(
        time_end=total_time, dt=dt, initial_q=q0, initial_v=vel0,
        external_force=lambda tval: spatial*load.value(tval))
    cw = q @ center_shape
    cv = velocities @ center_shape
    ca = accelerations @ center_shape
    idx = int(np.argmax(np.abs(cw)))
    result = {"t": t, "center_w": cw, "center_v": cv, "center_a": ca,
              "frequencies": frequencies[:3],
              "relaxed_frequencies": _clpt_relaxed_frequencies(plate),
              "peak_center_w_m": float(cw[idx]), "peak_center_time_s": float(t[idx]),
              "harmonic": _clpt_harmonic(plate, load, spatial), "dt": float(t[1]-t[0])}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/"history.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["time_s", "center_w_m", "center_w_mm", "center_v_m_s", "center_a_m_s2"])
        writer.writerows(zip(t, cw, 1000*cw, cv, ca))
    summary = {
        "model": "CLPT Rayleigh-Ritz; layerwise exact-cell Rjanitsin-Koltunov quadrature",
        "boundary_conditions": bc, "foundation": {"kw_N_per_m3": plate.kw, "kp_N_per_m": plate.kp},
        "ritz": {"nx": plate.nx, "ny": plate.ny}, "rotary_inertia": plate.rotary_inertia,
        "instantaneous_frequencies_hz": result["frequencies"].tolist(),
        "relaxed_frequencies_hz": result["relaxed_frequencies"].tolist(),
        "peak_center_deflection_m": result["peak_center_w_m"],
        "peak_center_time_s": result["peak_center_time_s"],
        "harmonic_steady_state": result["harmonic"],
        "dt_s": result["dt"], "steps": len(t)-1,
        "ABD": {key: getattr(plate, key+"mat").tolist() for key in ("A", "B", "D")},
        "limits": ["Kirchhoff-Love: transverse shear strain not resolved.",
                   "No layerwise transverse shear or strength results in CLPT.",
                   "Ritz functions and quadrature should undergo convergence tests.",
                   "FSDT simply-supported constraints may differ in-plane from CLPT."]}
    (output/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    ax.plot(t, 1000*cw, label="CLPT + RK", linewidth=1.5)
    ax.set(xlabel="Vaqt [s]", ylabel="Markaziy salqilik w [mm]",
           title=f"CLPT / Ritz, {bc}, kw={kw:g}, kp={kp:g}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output/"response.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    if check_dt:
        fine_conf = copy.deepcopy(config)
        fine_conf.setdefault("simulation", {})["dt"] = result["dt"]/2
        # A smaller auxiliary run may write transient files to a temporary folder.
        import tempfile
        with tempfile.TemporaryDirectory(prefix="clpt_half_dt_") as tmp:
            _, finer = run_clpt_config(fine_conf, tmp, show=False, quiet=True, check_dt=False)
        error = (np.max(np.abs(cw-np.interp(t, finer["t"], finer["center_w"])))
                 / max(np.max(np.abs(finer["center_w"])), 1e-30))
        (output/"time_convergence.json").write_text(json.dumps({
            "dt_s": result["dt"], "half_dt_s": finer["dt"],
            "center_displacement_relative_error": float(error),
            "scope": "Time step only; spatial Ritz convergence not tested."}, indent=2))
        if not quiet:
            print(f"CLPT dt/2 xato: {error:.3%}")
    if not quiet:
        print("CLPT oniy chastotalar [Hz]:", np.round(result["frequencies"], 5))
        print(f"CLPT markaziy maksimum |w| = {abs(result['peak_center_w_m'])*1000:.6g} mm")
        print("CLPT natijalar:", output.resolve())
    return plate, result


def compare_models(fsdt_result, clpt_result, output, show=False):
    """Export side-by-side center traces, without claiming same discretization."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    t = fsdt_result.t
    clpt_interp = np.interp(t, clpt_result["t"], clpt_result["center_w"])
    differences = fsdt_result.center_w-clpt_interp
    report = {
        "fsdt_frequencies_hz": fsdt_result.frequencies[:3].tolist(),
        "clpt_frequencies_hz": clpt_result["frequencies"].tolist(),
        "fsdt_peak_center_mm": float(1000*np.max(np.abs(fsdt_result.center_w))),
        "clpt_peak_center_mm": float(1000*np.max(np.abs(clpt_result["center_w"]))),
        "center_max_abs_difference_mm": float(1000*np.max(np.abs(differences))),
        "note": "Different physical kinematics (FSDT vs CLPT), support DOF constraints, "
                "Ritz vs FEM spatial approximations and SOE vs direct RK time scheme. "
                "These are method comparisons, not a strict identity/convergence test."}
    (out/"comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    with (out/"comparison.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["time_s", "fsdt_center_w_mm", "clpt_center_w_mm", "difference_mm"])
        writer.writerows(zip(t, fsdt_result.center_w*1000, clpt_interp*1000, differences*1000))
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(t, fsdt_result.center_w*1000, label="FSDT MITC4 FEM", lw=1.6)
    ax.plot(clpt_result["t"], clpt_result["center_w"]*1000,
            label="CLPT Ritz", linestyle="--", lw=1.3)
    ax.set(xlabel="Vaqt [s]", ylabel="Markaziy salqilik [mm]",
           title="FSDT va CLPT: bir xil material/geometriya/yuk parametrlari")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out/"comparison.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    print("\nFSDT f1:", f"{fsdt_result.frequencies[0]:.5g} Hz" if len(fsdt_result.frequencies) else "n/a")
    print("CLPT f1:", f"{clpt_result['frequencies'][0]:.5g} Hz")
    print(f"Maksimal markaziy farq = {report['center_max_abs_difference_mm']:.6g} mm")
    print("Taqqoslash fayllari:", out.resolve())
    return report


def self_test_unified():
    """Run included CLPT validation and short FSDT integration/foundation checks."""
    print("\n=== CLPT 5 ta analitik/etalon tekshiruv ===")
    validate_clpt()
    print("\n=== FSDT umumiy dastur tekshiruvlari ===")
    simple = OrthotropicLayer(h=0.008, theta=0, E1=70e9, E2=70e9,
                              G12=70e9/(2*1.3), G13=70e9/(2*1.3),
                              G23=70e9/(2*1.3), nu12=0.3, rho=2700)
    edges = normalize_edges("2")
    base = assemble_model([simple], 0.6, 0.5, 4, 4, edges, kw=0, kp=0)
    supported = assemble_model([simple], 0.6, 0.5, 4, 4, edges,
                               kw=1e7, kp=1e5)
    f1 = modes(base, 1)[0][0]
    f2 = modes(supported, 1)[0][0]
    assert f2 > f1, "FSDT: elastik asos chastotani oshirmadi."
    assert np.allclose(base.M.toarray(), supported.M.toarray())
    assert (supported.foundation_K-supported.foundation_K.T).nnz == 0 or np.allclose(
        (supported.foundation_K-supported.foundation_K.T).toarray(), 0, atol=1e-9)
    load = Load(kind="pulse", spatial="patch", amplitude=10, duration=0.001,
                x=0.3, y=0.25, patch_width=0.06, patch_height=0.05)
    res = simulate(supported, load, total_time=0.001, dt=1e-4,
                   calculate_stress=False)
    assert np.all(np.isfinite(res.center_w))
    assert max(abs(res.center_w)) > 0
    assert res.dt > 0
    print(f"FSDT asos: f1 = {f1:.6g} -> {f2:.6g} Hz (oshish tasdiqlandi).")
    print(f"FSDT 10 qadam: max|w_center|={max(abs(res.center_w))*1000:.4g} mm. OK")
    import tempfile
    with tempfile.TemporaryDirectory(prefix="unified_check_") as tmp:
        cfg = {
            "a": 0.6, "b": 0.5, "nx": 4, "ny": 4, "edges": edges,
            "kw": 1e7, "kp": 1e5,
            "layers": [vars(simple).copy()],
            "load": {"kind": "pulse", "spatial": "patch", "amplitude": 10,
                     "duration": 0.001, "x": 0.3, "y": 0.25,
                     "patch_width": 0.06, "patch_height": 0.05},
            "simulation": {"total_time": 0.001, "dt": 1e-4},
            "clpt": {"nx": 2, "ny": 2}
        }
        _, cresult = run_clpt_config(cfg, Path(tmp)/"clpt", quiet=True)
        assert len(cresult["t"]) == len(res.t)
        assert np.all(np.isfinite(cresult["center_w"]))
        compare_models(res, cresult, Path(tmp)/"compare")
        assert (Path(tmp)/"compare"/"comparison.json").exists()
    print("CLPT va FSDT umumiy konfiguratsiya orqali ishladi. TESTLAR MUVAFFAQIYATLI.")


def _inside_notebook_kernel():
    """True when code was executed as a cell, not as a normal CLI script.

    Colab/Jupyter put their private ``-f .../kernel.json`` option into
    sys.argv.  Never pass those kernel arguments to this application's CLI.
    ``%run script.py --demo ...`` is not affected: sys.argv[0] is then the
    actual script name, so the requested application options are preserved.
    """
    launcher = Path(sys.argv[0]).stem.lower()
    return launcher in {"colab_kernel_launcher", "ipykernel_launcher", "kernel_launcher"}


def unified_main(argv=None):
    # Preserve the usual argparse validation for command-line use.  When the
    # entire program is pasted into a Colab/Jupyter code cell, sys.argv is
    # owned by the notebook kernel, not by the user.  An explicit argv list
    # always has priority: unified_main(["--demo", "--model", "both"]).
    if argv is None and _inside_notebook_kernel():
        argv = []
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--demo", action="store_true", help="Tayyor namuna konfiguratsiya")
    source.add_argument("--config", type=Path, help="Umumiy JSON parametrlar fayli")
    source.add_argument("--self-test", action="store_true", help="CLPT/FSDT ichki testlari")
    parser.add_argument("--model", "--method", choices=("fsdt", "clpt", "both"),
                        default="fsdt", help="Hisoblash nazariyasi: FSDT, CLPT yoki ikkalasi")
    parser.add_argument("--output", default="results", help="Natija papkasi")
    parser.add_argument("--show", action="store_true", help="Grafiklarni ochish")
    parser.add_argument("--no-show", action="store_true", help="Grafiklar faqat faylga")
    parser.add_argument("--check-dt", action="store_true", help="Vaqt qadamini yarmiga kamaytirib tekshirish")
    parser.add_argument("--edges", nargs=4, metavar=("LEFT", "RIGHT", "BOTTOM", "TOP"),
                        help="C/qistirilgan, S/sharnirli, F/erkin chetlar")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        try:
            self_test_unified()
            return 0
        except (AssertionError, ValueError, RuntimeError, linalg.LinAlgError) as exc:
            print(f"TEKSHIRUV XATOSI: {exc}", file=sys.stderr)
            return 1
    try:
        config = (demo_config() if args.demo else
                  json.loads(args.config.read_text(encoding="utf-8")) if args.config else
                  interactive_config())
        if args.edges:
            config["edges"] = dict(zip(EDGE_NAMES, args.edges))
        output = Path(args.output)
        show = args.show and not args.no_show
        if args.model == "fsdt":
            run_config(config, output, show, args.quiet, args.check_dt)
        elif args.model == "clpt":
            run_clpt_config(config, output, show, args.quiet, args.check_dt)
        else:
            # Theories are computed separately on the same input; never
            # concatenate or add their system matrices/unknown vectors.
            _, fsdt_result = run_config(config, output/"fsdt", show, args.quiet, args.check_dt)
            _, clpt_result = run_clpt_config(config, output/"clpt", show, args.quiet, args.check_dt)
            compare_models(fsdt_result, clpt_result, output, show)
    except (ValueError, KeyError, TypeError, RuntimeError, OSError,
            linalg.LinAlgError, np.linalg.LinAlgError, ArithmeticError) as exc:
        print(f"XATO: {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nHisoblash to'xtatildi.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    _exit_status = unified_main()
    # sys.exit(), even sys.exit(0), raises SystemExit in Colab cells and can
    # show an unnecessary IPython traceback.  Preserve proper exit codes when
    # launched as a standalone script.
    if not _inside_notebook_kernel():
        raise SystemExit(_exit_status)
