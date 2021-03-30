import numpy as np
import random
from math import floor
from help_function import bound
from help_function import flatten
from help_function import reshape
def sigma(x,c,delta_f):
  return 2*c*delta_f/x
def differential_privacy(Gradient,privacy_budget_per_para,gamma,theta,tua,parameter_shape):
  Gradient_flattened = flatten(Gradient)
  parameter_number = len(Gradient_flattened)

  c = floor(theta*parameter_number)
  epsilon = privacy_budget_per_para*c
  epsilon_1 = 8/9*epsilon
  epsilon_2 = 2/9*epsilon

  sigma_1 = sigma(epsilon_1,c,2*gamma)
  sigma_2 = sigma(epsilon_2,c,2*gamma)

  Tua_with_noise = [np.random.laplace(0,sigma_1)+tua for _ in range(parameter_number)]
  R_w = [np.random.laplace(0,2*sigma_1) for _ in range(parameter_number)]
  Gradient_with_noise_flat1 = [abs(bound(Gradient_flattened[i],gamma))+R_w[i] for i in range(parameter_number)]
  index = []
  for i in range(parameter_number):
    if(Gradient_with_noise_flat1[i]>=Tua_with_noise[i]):
      index.append(i)
  if(len(index)<c):
    print("No enough parameters suit the conditions")
  else:
    np.random.shuffle(index)
    index = index[0:c]
  Gradient_with_noise_flat2 = np.zeros(parameter_number)
  for i in index:
    Gradient_with_noise_flat2[i]=Gradient_flattened[i]+np.random.laplace(0,sigma_2)
  return reshape(Gradient_with_noise_flat2,parameter_shape)
