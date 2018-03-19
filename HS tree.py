import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Node:
  def __init__(self, left=None, right=None, r=0, l=0, split_attrib=0, split_value=0.0, depth=0):
      self.left = left
      self.right = right
      self.r = r
      self.l = l
      self.split_attrib = split_attrib
      self.split_value = split_value
      self.k = depth


def generate_max_min(dimensions):
    max_arr = np.zeros((dimensions))
    min_arr = np.zeros((dimensions))
    for q in range(dimensions):
      s_q = np.random.random_sample()
      max_value = max(s_q, 1-s_q)
      max_arr[q] = s_q + 2*max_value
      min_arr[q] = s_q - 2*max_value
    return max_arr, min_arr

def BuildSingleHSTree(max_arr, min_arr, k, h, dimensions):
  if k == h:
    return Node(depth=k)
  node = Node()
  q = np.random.randint(dimensions)
  p = (max_arr[q] + min_arr[q])/2.0
  temp = max_arr[q]
  max_arr[q] = p
  node.left = BuildSingleHSTree(max_arr, min_arr, k+1, h, dimensions)
  max_arr[q] = temp
  min_arr[q] = p
  node.right = BuildSingleHSTree(max_arr, min_arr, k+1, h, dimensions)
  node.split_attrib = q
  node.split_value = p
  node.k = k
  return node

def UpdateMass(x, node, ref_window):
  if(node):
    if(node.k != 0):
      if ref_window:
        node.r += 1
      else:
        node.l += 1
    if(x[node.split_attrib] > node.split_value):
      node_new = node.right
    else:
      node_new = node.left
    UpdateMass(x, node_new, ref_window)

def ScoreTree(x,node, k):
  s = 0
  if(not node):
    return s
  s += node.r * (2**k)

  if(x[node.split_attrib] >node.split_value):
    node_new = node.right
  else:
    node_new = node.left
  s += ScoreTree(x, node_new, k+1)
  return s

def UpdateResetModel(node):
  if(node):
    node.r = node.l
    node.l = 0
    UpdateResetModel(node.left)
    UpdateResetModel(node.right)

def PrintTree(node):
  if(node):
    print(('Dimension of the node is:%d and split value is:%f, depth is:%d, reference_value:%d') %(node.split_attrib, node.split_value, node.k, node.r))
    PrintTree(node.left)
    PrintTree(node.right)




def StreamingHSTrees(X, psi, t, h):
  dimensions = X.shape[1]
  score_list = np.zeros((X.shape[0]))
  HSTree_list = []
  for i in range(t):
    max_arr, min_arr = generate_max_min(dimensions)
    tree = BuildSingleHSTree(max_arr, min_arr, 0, h, dimensions)
    HSTree_list.append(tree)
  for i in range(psi):
    for tree in HSTree_list:
      UpdateMass(X[i], tree, True)
  count = 0
  for i in range(X.shape[0]):
    x = X[i]
    s = 0
    for tree in HSTree_list:
      s = s + ScoreTree(x, tree, 0)
      UpdateMass(x, tree, False)
    print(('Score is %f for instance %d') %(s, i))
    score_list[i] = s
    count += 1

    if count == psi:
      print('Reset tree')
      for tree in HSTree_list:
        UpdateResetModel(tree)
      count = 0

  return score_list

''' Presently takes num number of lowest values and marks them as anomalies. num is obtained from total number of anomalies in the dataset.
 This is a bad way of doing it, but temporarily there is no defined way to compare the scores. This will be changed later ''' 
def accuracy_value(scores, y, num):
  tn = 0
  fp = 0
  tp = 0
  fn = 0
  ranks = np.argsort(scores)
  for rank in ranks[:num]:
    #print(rank)
    #print(y[rank])
    if y[rank] !=0:
      tp +=1
    else:
      fp += 1
  for rank in ranks[num:]:
    if y[rank] != 0:
      fn += 1
    else:
      tn += 1

  print(tp, fp, tn, fn)



def test_example1():
  x = [[0.5], [0.45], [0.66], [0.7], [0.43], [0.48], [0.61]]
  h = 3
  dimensions = 1
  max_arr, min_arr = generate_max_min(dimensions)
  tree = BuildSingleHSTree(max_arr, min_arr, 0, h, dimensions)
  PrintTree(tree)

  for i in range(len(x)):
    UpdateMass(x[i], tree, True)
  PrintTree(tree)
  print(ScoreTree([-0.1], tree, 0))


def test_example2():
  X = [[0.5], [0.45], [0.43], [0.44], [0.445], [0.45], [0.0]]
  X = np.array(X)
  y = [0,0,0,0,0,0,1]

  final_scores = StreamingHSTrees(X, 3, 5, 3)
  print(final_scores)
  accuracy_value(final_scores, y, 1)



def test_html():
  X = np.genfromtxt('http.csv',delimiter=',')
  y = np.genfromtxt('labels.csv',delimiter=',')
  X = X[311000:311500]
  y = y[311000:311500]

  print(X.shape)
  print(y.shape)
  anomalies = np.nonzero(y)[0]
  print(anomalies.shape)

  scaler = MinMaxScaler()
  X_new = scaler.fit_transform(X)

  final_scores = StreamingHSTrees(X_new, 250, 25, 15)
  accuracy_value(final_scores, y, 48)


def test_smtp():
  X = np.genfromtxt('smtp.csv',delimiter=',')
  y = np.genfromtxt('smtp_labels.csv',delimiter=',')
  X = X[11000:16000]
  y = y[11000:16000]

  print(X.shape)
  print(y.shape)
  anomalies = np.nonzero(y)[0]
  print(anomalies.shape)

  scaler = MinMaxScaler()
  X_new = scaler.fit_transform(X)

  final_scores = StreamingHSTrees(X_new, 250, 25, 15)
  accuracy_value(final_scores, y, 13)

test_smtp()