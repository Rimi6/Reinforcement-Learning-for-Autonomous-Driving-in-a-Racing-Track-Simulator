import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from models.cnn import BasicCNNForPPO


# =========================================================================================================================
class Box2DAgent():
    # --------------------------------------------------------------------------------------
    def __init__(self, device, p_oConfig):
        # ................................................................
        self.Config = p_oConfig
        
        nImageWidth, nImageHeight = self.Config["Data.EnvironmentResolution"]
        nImageStackCount          = self.Config["Data.ImageStackCount"]

        oTransitionDType = [('state'      , np.float64, (nImageStackCount, nImageWidth, nImageHeight)), 
                            ('action'     , np.float64, (3,)), 
                            ('a_logp'     , np.float64),
                            ('reward'     , np.float64), 
                            ('next_state' , np.float64, (nImageStackCount, nImageWidth, nImageHeight))]
                            
        self.max_size = self.Config["Training.RL.UpdateSteps"]
        self.training_step = 0
        self.net = BasicCNNForPPO(self.Config).double().to(device)
        self.buffer = np.empty(self.max_size, dtype=np.dtype(oTransitionDType))
        self.counter = 0
        self.device = device
        self.gamma = self.Config["Training.RL.Gamma"]  
        self.epochs = self.Config["Training.RL.Epochs"] 
        self.batch = self.Config["Training.RL.BatchSize"]
        self.eps = self.Config["Training.RL.EPS"]
        self.learning_rate = self.Config["Training.RL.LearningRate"]
                 
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)  ## lr=1e-3
        
        self.state  = None
        self.action = None
        self.a_logp = None
        # ................................................................
    # --------------------------------------------------------------------------------------
    def act(self, state):
        self.state = state
        
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    # --------------------------------------------------------------------------------------    
    def perceive(self, reward, next_state):
        transition_tuple = (self.state, self.action, self.a_logp, reward, next_state)
        self.buffer[self.counter] = transition_tuple
        self.counter += 1
        if self.counter == self.max_size:
            self.counter = 0
            self.update()
    # --------------------------------------------------------------------------------------
    def update(self):
        self.training_step += 1

        s       = torch.tensor(self.buffer['state'], dtype=torch.double).to(self.device)
        a       = torch.tensor(self.buffer['action'], dtype=torch.double).to(self.device)
        r       = torch.tensor(self.buffer['reward'], dtype=torch.double).to(self.device).view(-1, 1)
        next_s  = torch.tensor(self.buffer['next_state'], dtype=torch.double).to(self.device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(self.device).view(-1, 1)

        with torch.no_grad():
            target_v = r + self.gamma * self.net(next_s)[1]
            adv = target_v - self.net(s)[1]
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for nEpochNumber in range(self.epochs):
            print("Agent updating. Epoch %d" % (nEpochNumber + 1))
            for index in BatchSampler(SubsetRandomSampler(range(self.max_size)), self.batch, False):

                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                
                # clipped function
                surr2 = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    # --------------------------------------------------------------------------------------          
                
    '''
    # [PANTELIS] Sub-optimal source code refactored into perceive
    # action, a_logp = oAgent.act(state)
    # if oAgent.store((state, action, a_logp, reward, next_state)):
    #      print('updating')
    #      oAgent.update()
    def store(self, transition):

              
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.max_size:
            self.counter = 0
            return True
        else:
            return False
    '''              
# =========================================================================================================================