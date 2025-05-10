import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from game_env.gym_env import Denovo_DDPG, dt_state
from itertools import count
from torchvision import models

model = models.alexnet(pretrained=True)


# Bounding Box : Alexnet ( except softmax layer ) + fully connected layer
class BB_model(nn.Module):
    def __init__(self):
        super(BB_model, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        self.features1 = nn.Sequential(alexnet.features[:3])
        self.features2 = nn.Sequential(alexnet.features[3:6])
        self.features3 = nn.Sequential(alexnet.features[6:8])
        self.features4 = nn.Sequential(alexnet.features[8:10])
        self.features5 = nn.Sequential(alexnet.features[10:])
        self.avg = nn.Sequential(alexnet.avgpool)
        self.fc1 = nn.Sequential(alexnet.classifier[:3])
        self.fc2 = nn.Sequential(alexnet.classifier[3:6])

    def forward(self, x):
        x = x.reshape((-1, 3, 227, 227))
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        x = self.avg(x)
        x = nn.Flatten()(x)

        x = self.fc1(x)
        x = self.fc2(x)
        return x


# GPU device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load game env.
env = Denovo_DDPG()
bb_model = BB_model().to(device)


# Directory
def makedir(trial):
    if not os.path.exists('./result'):
        os.makedirs('./result')

    directory = './result/trial_{}'.format(trial)
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.makedirs(directory + '/actor')
        os.makedirs(directory + '/critic')

    print('=' * 50)
    print('Directory has been made.')
    print('=' * 50)

    return directory


# Replay Buffer
class Replay_buffer:
    def __init__(self, max_size=50000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        # reward prediction error. ( modify ).

        index = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in index:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


# Actor
class Actor(nn.Module):
    def __init__(self, s_dim, bb_model=bb_model):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.action_r = nn.Linear(256, 1)
        self.action_theta = nn.Linear(256, 1)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = x.squeeze()
        x = bb_model(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))

        action_r = torch.sigmoid(self.action_r(x))
        action_theta = torch.tanh(self.action_theta(x))

        action = torch.cat([action_r, action_theta], dim=1)

        return action


# Critic
class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(s_dim + a_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 1)

        self.bn0 = nn.BatchNorm1d(2048)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(256)

        self.f1 = nn.Linear(154587, 2048)
        self.f2 = nn.Linear(2048, 1024)

    def forward(self, x, u):
        x = x.squeeze()
        x = F.relu(self.bn0(self.f1(x)))
        x = F.relu(self.bn2(self.f2(x)))

        x = F.tanh(self.bn1(self.fc1(torch.cat([x, u], dim=1))))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn1(self.fc3(x)))
        x = F.relu(self.bn3(self.fc4(x)))
        x = self.fc5(x)

        return x


# DDPG
class DDPG(object):
    def __init__(self, s_dim, a_dim, a_lr, c_lr, directory):

        # Actor
        self.actor = Actor(s_dim).to(device)
        self.actor_target = Actor(s_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=a_lr)
        self.actor.eval()
        self.actor_target.eval()

        # Critic
        self.critic = Critic(s_dim, a_dim).to(device)
        self.critic_target = Critic(s_dim, a_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=c_lr)
        self.critic.eval()
        self.critic_target.eval()

        # Replay Buffer
        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)

        # Parameters
        self.critic_iter = 0
        self.actor_iter = 0
        self.train_iter = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, b_size, g_reward, epoch):
        for i in range(200):
            x, y, u, r, d = self.replay_buffer.sample(batch_size=b_size)
            state = torch.FloatTensor(x).to(device)
            next_state = torch.FloatTensor(y).to(device)
            action = torch.FloatTensor(u).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # next_state reward (predict) + current reward (actual)
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * 0.99 * target_Q).detach()

            # current_state reward (predict)
            current_Q = self.critic(state, action)

            # critic_loss
            # mse_loss between real_Q and pred_Q.
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('critic_loss', critic_loss, global_step=self.critic_iter)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # actor_loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('actor_loss', actor_loss, global_step=self.actor_iter)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.writer.add_scalar('rewards', g_reward, global_step=epoch)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(0.01 * param.data + 0.99 * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(0.01 * param.data + 0.99 * target_param.data)

            self.actor_iter += 1
            self.critic_iter += 1

    def save(self, directory, epoch=113):
        torch.save(self.actor, directory + '/actor' + '/actor_{}.pt'.format(epoch))
        torch.save(self.critic, directory + '/critic' + '/critic_{}.pt'.format(epoch))
        print('=' * 50)
        print('Epoch : {} // Model saved...'.format(epoch))
        print('=' * 50)

    def load(self, directory, epoch):
        self.actor = torch.load(directory + '/actor' + '/actor_{}.pt'.format(epoch), map_location=torch.device('cpu'))
        self.critic = torch.load(directory + '/critic' + '/critic_{}.pt'.format(epoch),
                                 map_location=torch.device('cpu'))
        print('=' * 50)
        print('Model has been loaded...')
        print('=' * 50)
        self.actor.eval()
        self.critic.eval()


def main():
    # Parameters
    state_dim = 1024
    action_dim = env.action_r.shape[0] + env.action_theta.shape[0]
    actor_lr = 0.00001
    critic_lr = 0.0001
    batch_size = 64
    flag = 'False'

    mode = input('Please enter the mode : ')
    max_episode = 1000
    trial = input('Please enter the trial number : ')
    directory = makedir(trial)

    agent = DDPG(state_dim, action_dim, actor_lr, critic_lr, directory)
    test_reward = 0
    reward_list = []

    epoch = 94
    if mode == 'test':
        agent.load(directory, epoch)

        # Check the parameters.
        for param_name, param_ in zip(agent.actor.parameters(), agent.actor.state_dict()):
            print(param_name, param_)

        for param_name, param_ in zip(agent.critic.parameters(), agent.critic.state_dict()):
            print(param_name, param_)

        for iter_ in range(10):
            state = dt_state(env.reset().copy())
            for step_ in count():
                action_r, action_theta = agent.select_action(state)
                action_r = np.array([action_r])
                action_theta = np.array([action_theta])

                env.render()
                next_state, reward, done, info = env.step(action_r * 3, action_theta * 180)
                next_state = dt_state(next_state)
                test_reward += reward
                env.render()

                if done or step_ > env.parameter.duration:
                    print('Episode : {}, Reward : {:.2f}, Step : {}'.format(int(iter_), test_reward, int(step_)))
                    test_reward = 0
                    break
                state = next_state.copy()

    elif mode == 'train':
        for episode in range(max_episode):
            total_reward = 0
            step = 0
            env.reset()
            state = env.to_frame().copy()
            total_step = 0

            for t in count():
                action_r, action_theta = agent.select_action(state)
                action_r = (action_r + np.random.normal(0, 0.2, size=1)).clip(0, 1)
                action_theta = (action_theta + np.random.normal(0, 0.5, size=1)).clip(-1, 1)

                env.render()

                next_state, reward, done, info = env.step(action_r * 4, action_theta * 180)
                next_state = env.to_frame()

                action = np.array([action_r.item(), action_theta.item()], dtype=float)
                agent.replay_buffer.push((state, next_state, action, reward, float(done)))

                state = next_state.copy()

                step += 1
                total_reward += reward

                if done:
                    break

            reward_list.append(total_reward)
            total_step += step + 1
            print('Episode : {}, Total Step : {}, Total Reward : {:.2f}'.format(episode, total_step, total_reward))
            agent.update(batch_size, total_reward, episode)

            if total_reward == max(reward_list):
                agent.save(directory=directory, epoch=episode)

    else:
        raise NameError('Please enter the right mode! [train/test]')


if __name__ == '__main__':
    main()