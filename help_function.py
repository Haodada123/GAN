import numpy as np
# help functions
def flatten(W_or_G):
  flattened = np.array([])
  for i in range(len(W_or_G)):
    flattened = np.concatenate((flattened, W_or_G[i].flatten()))

  return flattened
def reshape(Flattened,para_shapes): 
#Flattened: a np.array of all parameters.
#para_shapes: a list of tulpes representing the number and shape of parameters of neural networks layers
  reshaped_Parameter = []
  for p_num,shape in para_shapes:
    reshaped_Parameter.append(np.reshape(Flattened[0:p_num],shape))
    Flattened = Flattened[p_num:]
  return reshaped_Parameter



def bound(w,gamma):
  if(w>gamma):
    return gamma
  if(w<-gamma):
    return -gamma 
  return w 

# def change_Weight(Weight1,Weight2,stat):#set index to 0 before using
#   global index
  
#   if(type(Weight1[0])==np.int64 or type(Weight1[0])==np.float32):
#     for i in range(len(Weight1)):
#       if index in stat:
#         Weight1[i] = Weight2[i]
#       index += 1
#   else:
#     for i in range(len(Weight1)):
#       change_Weight(Weight1[i],Weight2[i],stat)