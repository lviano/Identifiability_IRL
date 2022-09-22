import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
import time
from itertools import product
import seaborn as sb

def tauchen_86(n,m,rho,sigma_eps):
    
    sigma_lz = sigma_eps/np.sqrt(1-rho**2) 
    Phi      = norm.cdf
    lz_n     = m*0.6 #sigma_lz
    h        = 2*lz_n/(n-1) ##
    lzs      = np.linspace(-lz_n,lz_n,n)
    Tz       = np.empty((n,n))
    const    = h/(2*sigma_eps)

    for i in range(n):
         for j in range(n):
            diff_ij = (lzs[j] - rho*lzs[i])/sigma_eps  
            first   = int(j == 0)
            last    = int(j == n-1) 

            Tz[i,j] = Phi(diff_ij+const)*(1-last) + last - \
                     Phi(diff_ij-const)*(1-first)  

    return {'Tz':Tz,'lzs':lzs,'sigma_lz':sigma_lz}

def states_f(ks,zs):
  return np.array([(k,z) for k in ks for z in zs]) ### returns an array where the first column is ks and the second is zs

def transition_matrix_f(Tz,nActions):
  nShocks = Tz.shape[0]
  nStates = nActions*nShocks

  T   = np.zeros((nActions,nStates,nStates)) # Transistion matrix
  
  Tz_ = np.tile(Tz,(nActions,1)) # stacked Tz arrays to build transistion matrix

  for i in range(nActions):
    T[i][:,i*nShocks:(i+1)*nShocks] = Tz_
  return T 
def soft_value_iteration(n_states, n_actions, transition_probabilities, reward, discount,threshold=1e-5, temp=1):
    v = np.zeros(n_states)
    q = np.zeros((n_states, n_actions))

    while True:
        v_old = np.copy(v)
        for a in range(n_actions):
            q[:, a] = reward[:, a] + discount * transition_probabilities[:,a,:].dot(v)
        v = 1/temp*softmax(temp*q).reshape(n_states)
        if np.linalg.norm(v - v_old) < threshold:
            break
    return softmax_probs(temp*q), v

def softmax(x):
    return x.max(axis=1).reshape(x.shape[0], 1) + np.log(np.exp(x - x.max(axis=1).reshape(x.shape[0], 1)).sum(axis=1)).reshape(x.shape[0], 1)


def softmax_probs(x):
    return np.exp(x-np.max(x, axis=1).reshape(x.shape[0], 1)) / np.exp(x-np.max(x, axis=1).reshape(x.shape[0], 1)).sum(axis=1).reshape(x.shape[0], 1)

def gen_opt(theta,delta,r,sigma_eps,rho):
  def opt_policy_f(z):
    return ((theta* z**rho* np.exp(0.5*sigma_eps**2))/(r+delta))**(1/(1-theta))

  def opt_reward_f(s): ### s is the current state k,z
    k,z = s
    return z*k**theta +(1-delta)*k - opt_policy_f(z)

  return opt_policy_f,opt_reward_f

def generate_random_transition(n_states, n_actions):
    
    matrix = np.random.randn(n_states*n_actions, n_states)
    matrix = np.abs(matrix).T/np.sum(np.abs(matrix), axis=1)
    
    return matrix.T

def generate_random_reward(n_states, n_actions):
    matrix = np.random.randn(n_states, n_actions)
    return matrix

def plot_reward(reward,title,max_c, min_c, color="hot"):
    plt.figure()
    img=plt.imshow(reward)#.reshape(n_s, n_a))
    img.set_cmap(color)
    img.set_clim(min_c,max_c)
    plt.axis('off')
    plt.ylabel('States')
    plt.xlabel('Actions')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(title+".pdf",bbox_inches='tight')

def plot_state_only_reward(reward,title,max_c, min_c):
    plt.figure()
    reward = reward[:,0]
    s = np.int(np.sqrt(reward.shape[0]))
    img=plt.imshow(reward.reshape(s, s))
    img.set_cmap('copper')
    img.set_clim(min_c,max_c)
    plt.axis('off')
    plt.ylabel('States')
    plt.xlabel('Actions')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(title+".pdf",bbox_inches='tight')