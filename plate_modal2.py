#
# ..    # gedit: set fileencoding=utf8 :
#
# .. raw:: html
#
#  <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><p align="center"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png"/></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a></p>
#
# .. _ReissnerMindlinQuads:
#
# ==========================================
# Reissner-Mindlin plate with Quadrilaterals
# ==========================================
#
# -------------
# Introduction
# -------------
#
# This program solves the Reissner-Mindlin plate equations on the unit
# square with uniform transverse loading and fully clamped boundary conditions.
# The corresponding file can be obtained from :download:`reissner_mindlin_quads.py`.
#
# It uses quadrilateral cells and selective reduced integration (SRI) to
# remove shear-locking issues in the thin plate limit. Both linear and
# quadratic interpolation are considered for the transverse deflection
# :math:`w` and rotation :math:`\underline{\theta}`.
#
# .. note:: Note that for a structured square grid such as this example, quadratic
#  quadrangles will not exhibit shear locking because of the strong symmetry (similar
#  to the criss-crossed configuration which does not lock). However, perturbating
#  the mesh coordinates to generate skewed elements suffice to exhibit shear locking.
#
# The solution for :math:`w` in this demo will look as follows:
#
# .. image:: clamped_40x40.png
#    :scale: 40 %
#
#
#
# ---------------
# Implementation
# ---------------
#
#
# Material parameters for isotropic linear elastic behavior are first defined::

from __future__ import print_function
from dolfin import *
import numpy as np
from petsc4py import PETSc
import petsc4py

E = Constant(2.1e11)
nu = Constant(0.28)
rho = 8000

# Plate bending stiffness :math:`D=\dfrac{Eh^3}{12(1-\nu^2)}` and shear stiffness :math:`F = \kappa Gh`
# with a shear correction factor :math:`\kappa = 0.86` for clamped plates for a homogeneous plate
# of thickness :math:`h`::

thick = Constant(0.002)
D = (E*(thick**3))/(12.*(1-nu**2))
F = E/2/(1+nu)*thick*0.86


# The unit square mesh is divided in :math:`N\times N` quadrilaterals::

N = 100
mesh = UnitSquareMesh.create(N, N, CellType.Type.quadrilateral)

# The Rectangle Mesh is used with Quadrilaterals::

#Nx = 200
#Ny = 200
#mesh = RectangleMesh.create([Point(0.0,0.0), Point(1.0,1.0)], [100, 100], CellType.Type.quadrilateral)

# Continuous interpolation using of degree :math:`d=\texttt{deg}` is chosen for both deflection and rotation::

deg = 2
We = FiniteElement("Lagrange", mesh.ufl_cell(), deg)
Te = VectorElement("Lagrange", mesh.ufl_cell(), deg)
V = FunctionSpace(mesh, MixedElement([We, Te]))

# Clamped boundary conditions on the lateral boundary are defined as::

#def border(x, on_boundary):
 #   return on_boundary

#bc = [DirichletBC(V, Constant((0.0, 0.0, 0.0)), border)]


# Simply Supported Boundary Conditions on two opposite edges

def all_boundary(x, on_boundary):
    return all_boundary

def left(x, on_boundary):
    return on_boundary and near(x[0], 0.0)

def right(x, on_boundary):
    return on_boundary and near(x[0], 1.0)

#def bottom(x, on_boundary):
 #   return on_boundary and near(x[1], 0.0)

#def top(x, on_boundary):
 #   return on_boundary and near(x[1], 1.0)

#Simply supported boundary conditions.
bc = [DirichletBC(V.sub(1), Constant(0.0), all_boundary),
      DirichletBC(V.sub(0).sub(0), Constant(0.0), top),
      DirichletBC(V.sub(0).sub(0), Constant(0.0), bottom),
      DirichletBC(V.sub(0).sub(1), Constant(0.0), left),
      DirichletBC(V.sub(0).sub(1), Constant(0.0), right)]

# Some useful functions for implementing generalized constitutive relations are now
# defined::


def strain2voigt(eps):
    return as_vector([eps[0, 0], eps[1, 1], 2*eps[0, 1]])

def voigt2stress(S):
    return as_tensor([[S[0], S[2]], [S[2], S[1]]])

def curv(u):
    (w, theta) = split(u)
    return sym(grad(theta))

def shear_strain(u):
    (w, theta) = split(u)
    return theta-grad(w)

def bending_moment(u):
    DD = as_tensor([[D, nu*D, 0], [nu*D, D, 0],[0, 0, D*(1-nu)/2.]])
    return voigt2stress(dot(DD,strain2voigt(curv(u))))

def shear_force(u):
    return F*shear_strain(u)


# The contribution of shear forces to the total energy is under-integrated using
# a custom quadrature rule of degree :math:`2d-2` i.e. for linear (:math:`d=1`)
# quadrilaterals, the shear energy is integrated as if it were constant (1 Gauss point instead of 2x2)
# and for quadratic (:math:`d=2`) quadrilaterals, as if it were quadratic (2x2 Gauss points instead of 3x3)::

u = Function(V)
u_ = TestFunction(V)
du = TrialFunction(V)

(w_, theta_) = split(u_)
(dw, dtheta) = split(du)

dx_shear = dx(metadata={"quadrature_degree": 2*deg-2})

l_form = Constant(1.)*u_[0]*dx
k_form = inner(bending_moment(u_), curv(du))*dx + dot(shear_force(u_), shear_strain(du))*dx_shear

K = PETScMatrix()
b = PETScVector()

assemble_system(k_form, l_form, bc, A_tensor=K, b_tensor = b)


m_form = rho*thick*dot(dw, w_)*dx + (rho/12.)*(thick**3)*dot(dtheta, theta_)*dx + (rho/12.)*(thick**3)*dot(dtheta, theta_)*dx
M = PETScMatrix()
assemble(m_form, tensor=M)

for bci in bc:
    bci.zero(M)


eigensolver = SLEPcEigenSolver(K, M)
eigensolver.parameters['problem_type'] = 'gen_hermitian'
eigensolver.parameters['spectral_transform'] = 'shift-and-invert'
eigensolver.parameters['spectral_shift'] = 0.

N_eig = 6   # number of eigenvalues
print("Computing %i first eigenvalues..." % N_eig)
eigensolver.solve(N_eig)  

print(K)
# Exact solution computation
#from scipy.optimize import root
#from math import cos, cosh
#falpha = lambda x: cos(x)*cosh(x)+1
#alpha = lambda n: root(falpha, (2*n+1)*pi/2.)['x'][0]

# Set up file for exporting results
file_results = XDMFFile("modal_analysis.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True


# Extraction
for i in range(N_eig):
    # Extract eigenpair
    r, c, rx, cx = eigensolver.get_eigenpair(i)

    # 3D eigenfrequency
    freq_3D = sqrt(r)/2/pi


    print("Solid FE: {0:>10.5f}".format(freq_3D))

    # Initialize function and assign eigenvector (renormalize by stiffness matrix)
    eigenmode = Function(V,name="Eigenvector "+str(i))
    eigenmode.vector()[:] = rx



