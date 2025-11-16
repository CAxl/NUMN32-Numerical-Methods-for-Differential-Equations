import numpy as np
import matplotlib.pyplot as plt

from Project0 import IVP, LTE, Solver, ExplicitEuler, ImplicitEuler, evaluator




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




# test0: compare to LTE.exact

# # dfine A and initial condition
# A = np.array([[0, 1],
# 			  [-1, 0]])

# t0 = 0.0
# T  = 4 * np.pi
# y0 = np.array([1.0, 0.0])
# N = 1000

# problem = LTE(t0, T, y0, A)

# # test exact solution at T
# yt_exact = problem.exact(T)

# print("y(T) =", yt_exact)

# ts = np.linspace(0, T, N)
# y1 = [problem.exact(t)[0] for t in ts]
# plt.plot(ts, y1, label = "exact solution y1")
# y2 = [problem.exact(t)[1] for t in ts]
# plt.plot(ts, y2, label = "exact solution y2")

# # initialize solver with the LTE
# solver = ExplicitEuler(problem, N)
# solver.integrate()

# plt.plot(solver.grid, solver.us[0,:], label = "Explicit Euler y1")
# plt.plot(solver.grid, solver.us[1,:], label = "Explicit Euler y2")



# plt.plot(ts, np.cos(ts), linestyle='--', color = 'black', label = "cos(t)")
# plt.plot(ts, -np.sin(ts), linestyle='-.', color = 'black', label = "-sin(t)")

# plt.title(f"Explicit Euler for LTE with N = {N}")
# plt.ylabel("y(tn) / un")
# plt.xlabel("t")
# plt.grid()
# plt.legend()
# plt.show()


# ----------------------- Section 2 tests ---------------------
# # test 1
# ivp = LTE(t0 = 0, T = 1, y0 = np.array([2]), A = np.array([[-1]]))
# solver = ExplicitEuler(ivp, N = 100)
# solver.integrate()

# # exact solution
# y1 = np.array([ivp.exact(t)[0] for t in solver.grid])

# plt.plot(solver.grid, solver.us[0,:], label="Explicit Euler")
# plt.plot(solver.grid, y1, label="Exact")
# plt.title("Test 1")
# plt.xlabel("t")
# plt.ylabel("y(tn) / un")
# plt.legend()
# plt.show()

# # test 2
# A = np.array([[-1, 1], [1, -3]])
# y0 = np.array ([1 , 2])
# ivp = LTE(t0 = 0, T = 1, y0 = y0, A = A)
# solver = ExplicitEuler(ivp, N = 100)
# solver.integrate()

# # exact
# y2 = np.array([ivp.exact(t)[0] for t in solver.grid])
# # Plot the first component of the approximations { u_n }
# # vs. the temporal grid points { t_n }:
# plt.plot(solver.grid, solver.us[0,:], label = "Explicit Euler")
# plt.plot(solver.grid, y2, label = "Exact")
# plt.title("Test 2")
# plt.legend()
# plt.show()



# # errors test 0
# A = np.array([[0, 1],
# 			  [-1, 0]])

# t0 = 0.0
# T  = 4 * np.pi
# y0 = np.array([1.0, 0.0])

# ivp = LTE(t0, T, y0, A)
# ev = evaluator(ivp, ExplicitEuler)

# ts, errs = ev.errvstime(N=100)
# plt.plot(ts, errs)
# plt.grid()
# plt.title("Error test 0")
# plt.xlabel("t")
# plt.ylabel("err")
# plt.show()


# # errors test 1
# ivp = LTE(t0=0, T=1, y0=np.array([2]), A=np.array([[-1]]))
# ev = evaluator(ivp, ExplicitEuler)

# ts, errs = ev.errvstime(N=100)

# plt.plot(ts, errs)
# plt.grid()
# plt.title("Error test 1")
# plt.xlabel("t")
# plt.ylabel("err")
# plt.show()

# # errors test 2
# A = np.array([[-1, 1], [1, -3]])
# y0 = np.array ([1 , 2])
# ivp = LTE(0, 1, y0, A)
# ev = evaluator(ivp, ExplicitEuler)

# ts, errs = ev.errvstime(N=100)
# plt.plot(ts, errs)
# plt.grid()
# plt.title("Error test 2")
# plt.xlabel("t")
# plt.ylabel("err")
# plt.show()



# Section 7 tests

# # 7.1
# lambdas = [-3, -2, -1, 0, 1, 2, 3]
# Ns = [2**k for k in range(0,10)]
# print(Ns)


# for lam in lambdas:
#     ivp = LTE(t0=0, T=1, y0=np.array([2]), A=np.array([[lam]]))
#     ev = evaluator(ivp, ExplicitEuler)

#     hs, errs = ev.errvsh(Ns)

#     plt.loglog(hs, errs, 'o-', label=f"\lambda = {lam}")

# plt.xlabel("Step size h")
# plt.ylabel("Final error |u_N - y(T)|")
# plt.title("Convergence of Explicit Euler: error vs h")
# plt.grid(True, which="both")
# plt.legend()
# plt.show()


# # 7.2
# ivp1 = LTE(t0=0, T=1, y0=np.array([2]), A=np.array([[-1]]))
# ivp2 = LTE(t0=0, T=1, y0=np.array([2]), A=np.array([[1]]))
# eval1 = evaluator(ivp1, ExplicitEuler)
# eval2 = evaluator(ivp2, ExplicitEuler)

# ts1, errs1 = eval1.errvstime(N=100)
# ts2, errs2 = eval2.errvstime(N=100)

# plt.figure()
# plt.semilogy(ts1, errs1, label="A = -1")
# plt.semilogy(ts2, errs2, label="A = +1")
# plt.xlabel("t")
# plt.ylabel("Error")
# plt.title("Absolute error vs time")
# plt.grid()
# plt.legend()
# plt.show()


# # 7.3
# A = np.array([[-1, 10],
#               [0, -3]])
# y0 = np.array([1,1])
# ivp = LTE(0, 10, y0, A)

# ev = evaluator(ivp, ExplicitEuler)

# N = 500

# ts = np.linspace(0, ivp.T, N+1)
# ys = [ivp.exact(t)[0] for t in ts]

# solver = ExplicitEuler(ivp, N)
# solver.integrate()

# plt.plot(solver.grid, solver.us[0,:], label = "Explicit Euler")
# plt.plot(solver.grid, ys, label = "Exact")
# plt.title("Test 7.3 Exact vs Explicit Euler")
# plt.xlabel("t")
# plt.ylabel("y / u_k")
# plt.grid()
# plt.legend()
# plt.show()

# ts, errs = ev.errvstime(N)
# plt.figure()
# plt.semilogy(ts, errs)
# plt.title("Matrix A: error vs time")
# plt.xlabel("t")
# plt.ylabel("Error |u_k - y(t_k)|")
# plt.grid()
# plt.show()

# # Error vs step size
# Ns = [2**k for k in range(0, 10)]
# hs, errs_final = ev.errvsh(Ns)
# plt.figure()
# plt.loglog(hs, errs_final, 'o-')
# plt.title("Matrix A: final error vs h")
# plt.xlabel("Stepsize h")
# plt.ylabel("Final error e_N")
# plt.grid(True, which="both")
# plt.show()


# section 8 Implicit Euler tests

# 8.1 / 8.2
N = 100
ivp = LTE(0, 1, np.array([2]), np.array([[-10]]))
solver_e = ExplicitEuler(ivp, N)
solver_i = ImplicitEuler(ivp, N)

solver_e.integrate()
solver_i.integrate()

# plot exact solution
ts = np.linspace(0, ivp.T, N)
ys = [ivp.exact(t) for t in ts]

plt.plot(ts, ys, label = "exact")

# plot numerical solutions
plt.plot(solver_e.grid, solver_e.us[0,:], label="Explicit")
plt.plot(solver_i.grid, solver_i.us[0,:], label="Implicit")
plt.legend()
plt.grid()
plt.xlabel("t")
plt.ylabel("y")
plt.show()


# test 8.4

A = np.array([[-1, 100],
              [0, -30]])
y0 = np.array([1,1])

t0 = 0
T = 10
N = 100

ivp = LTE(t0, T, y0, A)

expl = ExplicitEuler(ivp, N)
expl.integrate()
impl = ImplicitEuler(ivp, N)
impl.integrate()

plt.plot(expl.grid, expl.us[0,:], label = "Explicit Euler")
plt.plot(impl.grid, impl.us[0,:], label = "Implicit Euler") # should be the same grid given the same initial conds

#exact
ts = np.linspace(0, T, N+1) # for some reason I need N+1 (annoying)
ys = [ivp.exact(t)[0] for t in ts]
#plt.plot(ts, ys, label = "exact")

plt.legend()
plt.loglog()
plt.grid()
plt.show()

# test errvstime (and errvsh) below! (@tomorrow self)




