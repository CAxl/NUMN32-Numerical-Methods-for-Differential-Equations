import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg as la


class IVP:
	# Base class for IVP y'(t) = f(t,y)
	def __init__(self, t0, T, y0):
		self.t0 = t0
		self.T = T
		self.y0 = y0
		

	def f(self, t, y):
		raise NotImplementedError("Subclasses must implement f(t, y).")
	

class LTE(IVP): # linear test problem \dot{y} = Ay, y(t_0) = y_0
	def __init__(self, t0, T, y0, A):
		super().__init__(t0, T, y0)
		self.A = A
		
	def f(self, t, y):
		return self.A @ y
		
	def exact(self, t):
		# exact solution y(t) = e^{A(t-t_0)}y0
		return la.expm(self.A * (t - self.t0)) @ self.y0
	


class Solver:
	def __init__(self, ivp, N):
		self.ivp = ivp
		self.N = N
		
		self.u = ivp.y0 # initial value
		self.grid = np.linspace(ivp.t0, ivp.T, N + 1) # set-up grid
		self.h = (ivp.T - ivp.t0) / N # grid spacing

	def step(self, tn, un):
		raise NotImplementedError("Subclass must implement step()")
	
	def integrate(self):
		# preallocate N+1 cols, and we want to fill then with ys -> y0.size()
		us = np.empty((self.ivp.y0.size, self.N+1))
		us[:, 0] = self.u # store y0 in first col
		
		for i in range(self.N):
			self.u = self.step(self.grid[i], self.u)
			us[:,i+1] = self.u
		
		self.us = us	

		return us
		

class ExplicitEuler(Solver):
	def __init__(self, ivp, N):
		super().__init__(ivp, N)

	def step(self, tn, un):
		return un + self.h * self.ivp.f(tn, un)
	

class ImplicitEuler(Solver):
	def __init__(self, ivp, N):
		super().__init__(ivp, N)
		self.I = np.eye(self.ivp.y0.size) # identity matrix to solve (I-hA)u_{n+1} = u_{n}
	
	def step(self, tn, un):
		A = self.ivp.A
		M = self.I - self.h*A

		unp1 = np.linalg.solve(M, un) # u_{n+1} = unp1
		
		return unp1


class evaluator:
	def __init__(self, ivp, solver_type):
		self.ivp = ivp
		self.solver_type = solver_type

	def errvstime(self, N, relative = False):
		solver = self.solver_type(self.ivp, N)
		solver.integrate() # solver now has us = [dim \times N+1] cols are u_k

		ts = solver.grid
		us = solver.us
		
		# compute exact
		y_exact = np.array([self.ivp.exact(t) for t in ts]).T # transpose to match solver indexing
		
		if relative:
			errs = np.linalg.norm(us - y_exact, axis = 0) / np.linalg.norm(y_exact, axis = 0)
		else:
			errs = np.linalg.norm(us - y_exact, axis=0)
			
		return ts, errs
	
	def errvsh(self, Ns, relative = False):
		errors = []
		hs = []
		
		for N in Ns:
			solver = self.solver_type(self.ivp, N)
			solver.integrate()

			h = solver.h
			us_final = solver.us[:, -1]      # u_N approx at t = T
			y_final = self.ivp.exact(self.ivp.T)

			if relative:
				err = np.linalg.norm(us_final - y_final) / np.linalg.norm(y_final)
			else:
				err = np.linalg.norm(us_final - y_final)

			hs.append(h)
			errors.append(err)

		return np.array(hs), np.array(errors)
		
	




		

		

		
	


