import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg


class IVP:
	# Base class for IVP y'(t) = f(t,y)
	def __init__(self, t0, T, y0):
		self.t0 = t0
		self.T = T
		self.y0 = y0
		

	def f(self, t, y):
		raise NotImplementedError("Subclasses must implement f(t, y).")
	

class LTE(IVP):
	def __init__(self, t0, T, y0, A):
		super().__init__(t0, T, y0)
		self.A = A
		
	def f(self, t, y):
		return self.A @ y
		



