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
	

# Testing LTE

# # dfine A and initial condition
# A = np.array([[0, 1],
#               [-1, 0]])

# t0 = 0.0
# T  = 2 * np.pi
# y0 = np.array([1.0, 0.0])

# problem = LTE(t0, T, y0, A)

# # test exact solution at T
# yt_exact = problem.exact(T)

# print("y(T) =", yt_exact)

# ts = np.linspace(0, 10, 2000)
# ys = [problem.exact(t) for t in ts]
# plt.plot(ts, ys)


# plt.plot(ts, np.cos(ts), linestyle='--', color = 'black')
# plt.plot(ts, -np.sin(ts), linestyle='--', color = 'black')

# plt.show()




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
	


# test0: compare to LTE.exact

# dfine A and initial condition
A = np.array([[0, 1],
			  [-1, 0]])

t0 = 0.0
T  = 4 * np.pi
y0 = np.array([1.0, 0.0])
N = 1000

problem = LTE(t0, T, y0, A)

# test exact solution at T
yt_exact = problem.exact(T)

print("y(T) =", yt_exact)

ts = np.linspace(0, T, N)
ys = [problem.exact(t) for t in ts]
plt.plot(ts, ys, label = "exact solution")

# initialize solver with the LTE
solver = ExplicitEuler(problem, N)
solver.integrate()

plt.plot(solver.grid, solver.us[0,:], label = "Explicit Euler")


plt.plot(ts, np.cos(ts), linestyle='--', color = 'black', label = "cos(t)")
plt.plot(ts, -np.sin(ts), linestyle='-.', color = 'black', label = "-sin(t)")

plt.title(f"Explicit Euler for LTE with N = {N}")
plt.ylabel("y(t)")
plt.xlabel("t")
plt.grid()
plt.legend()
plt.show()


# test 2
# A = np.array([[-1, 1], [1, -3]])
# y0 = np.array ([1 , 2])
# ivp = LTE(t0 = 0, T = 1, y0 = y0, A = A)
# solver = ExplicitEuler(ivp, N = 100)
# solver.integrate ()
# # Plot the first component of the approximations { u_n }
# # vs. the temporal grid points { t_n }:
# plt.plot(solver.grid, solver.us[0,:])
# plt.show()


