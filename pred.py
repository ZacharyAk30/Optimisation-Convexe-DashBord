import numpy as np

def forward_propagation_type_1(X, parametres):
  activations = {'A0': X}
  C = len(parametres) // 2
  for c in range(1, C + 1):
    Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
    activations['A' + str(c)] = 1 / (1 + np.exp(-Z))
  return activations
def forward_propagation_type_2(X, parametres):
  activations = {'A0': X}
  C = len(parametres) // 4 
  for c in range(1, C + 1):
    Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
    activations['A' + str(c)] = 1 / (1 + np.exp(-Z))
  return activations

def predict_RN(X, parametres):
  activations = forward_propagation_type_1(X, parametres)
  C = len(parametres) // 2
  Af = activations['A' + str(C)]
  return Af >= 0.5

def predict_Adam(X, parametres):
  activations = forward_propagation_type_1(X, parametres)
  C = len(parametres) // 2
  Af = activations['A' + str(C)]
  return Af >= 0.5
def predict_SGD_Moment(X, parametres):
  activations = forward_propagation_type_2(X, parametres)
  C = len(parametres) // 4 
  Af = activations['A' + str(C)]
  return Af >= 0.5

def predict_RMSProp(X, parametres):
  activations = forward_propagation_type_2(X, parametres)
  C = len(parametres) // 4 
  Af = activations['A' + str(C)]
  return Af >= 0.5
def predict_AdaGrad(X, parametres):
  activations = forward_propagation_type_2(X, parametres)
  C = len(parametres) // 4 
  Af = activations['A' + str(C)]
  return Af >= 0.5