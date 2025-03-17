import gym
from gym import spaces
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import os
#from pyswmm import Simulation, Nodes
#from swmm.toolkit.shared_enum import LinkAttribute
#from pyswmm import Output
import yaml
#from string import Template
import pyswmm.toolkitapi as tkai
from pyswmm.simulation import Simulation
import random
from collections import deque
import matplotlib.pyplot as plt
import scipy.stats as ss
class Swmm_model:

    
    def __init__(self, config, action_params = None):
        self.config = yaml.load(open(config, "r"), yaml.FullLoader)
        self.sim = Simulation(self.config["model_folder"] +
                              self.config["model_name"] +".inp") # initialize simulation
        self.sim.start()

        # methods
        self.methods = {
            "depthN": self._getNodeDepth,
            "depthL": self._getLinkDepth,
            "volumeN": self._getNodeVolume,
            "volumeL": self._getLinkVolume,
            "flow": self._getLinkFlow,
            "flooding": self._getNodeFlooding,
            "inflow": self._getNodeInflow,
            "setting": self._getValvePosition,
            "total_precip": self._getRainfall
        }
        
        self.action_params = action_params
        
        # create datalog
        self.data_log = {"time":[],
                         "flow": {}, "inflow": {}, "flooding": {}, 'depthN':{}, 'setting':{}, 'total_precip': {}}
        
        if self.config["states_for_computing_objectives"] is not None:
            for entity, attribute in self.config["states_for_computing_objectives"]:
                self.data_log[attribute][entity] = []
    
        if self.config["states"] is not None:
            for entity, attribute in self.config["states"]:
                self.data_log[attribute][entity] = []
    
    def run_simulation(self):

        done = False
        while done == False:
            if self.action_params is not None:
                actions = self._compute_actions()
                self._take_action(actions)
            time = self.sim._model.swmm_step()

            # log information
            self._log_tstep()
            self.data_log['time'].append(self.sim._model.getCurrentSimulationTime())
            
            done = False if time > 0 else True # time increases till the end of the sim., then resets to 0
        self._end_and_close()
    
    def _log_tstep(self):
        for attribute in self.data_log.keys():
            if attribute != "time" and len(self.data_log[attribute]) > 0:
                for entity in self.data_log[attribute].keys():# ID
                    self.data_log[attribute][entity].append(
                        self.methods[attribute](entity))
 
        
    # ------ Valve modifications -------------------------------------------
    def _getValvePosition(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.setting.value)

    def _setValvePosition(self, ID, valve):
        return self.sim._model.setLinkSetting(ID, valve)
        
    # ------ Node Parameters  ----------------------------------------------
    def _getNodeDepth(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.newDepth.value)
    
    def _getRainfall(self, ID):
        return self.sim._model.getGagePrecip(ID, tkai.RainGageResults.total_precip.value)   

    def _getNodeFlooding(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.overflow.value)

    def _getNodeLosses(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.losses.value)
    
    def _getNodeVolume(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.newVolume.value)

    def _getNodeInflow(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.totalinflow.value)

    def _setInflow(self, ID, value):
        return self.sim._model.setNodeInflow(ID, value)

    # ------ Link modifications --------------------------------------------
    def _getLinkDepth(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.newDepth.value)

    def _getLinkVolume(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.newVolume.value)

    def _getLinkFlow(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.newFlow.value)
        
    def _get_state(self):
        # create list of tuples with all state variables
        states = self.config["states_for_computing_objectives"].copy()
        states.extend(self.config["states"])
        
        state = []
        for s in states:
            entity = s[0] # name of object in swmm
            attribute = s[1] # attribute of interest (e.g. flow)
            state.append(self.methods[attribute](entity))

        state = np.asarray(state)
        
        return state
    
    def _get_lagged_states(self):
        states = self.config["lagged_state_variables"]
        lag_states = []        
        for s in states:
            entity = s[0] # name of object in swmm
            attribute = s[1] # attribute of interest (e.g. flow)
            for lag_hrs in self.config['lags_hrs']:
                ct = self.data_log['time'][-1]
                lagged_idx = 0
                dif = 0
                while dif < lag_hrs:
                    lagged_idx -= 1
                    lt = self.data_log['time'][lagged_idx]
                    dif = (ct - lt).total_seconds() / 60 / 60 # hours
                    
                lag_states.append(self.data_log[attribute][entity][lagged_idx])
                
        return lag_states
    
    def _compute_actions(self):

        actions = []
        actions.append(self.action_params[0])
        actions.append(self.action_params[1])
 
        return actions
    
    
    def export_df(self):
        for key in self.data_log.keys():
            if key == 'time':
                df = pd.DataFrame({key : self.data_log[key]})
                continue
            tmp = pd.DataFrame.from_dict(self.data_log[key])
            if len(tmp) == 0:
                continue
            new_col_names = []
            for col_name in tmp.columns:
                new_col_names.append(str(col_name) + '_' + key)
            tmp.columns = new_col_names
            
            df = df.merge(tmp, left_index=True, right_index=True)
            
        return df
        
    
    def _take_action(self, actions=None):
        if actions is not None:
            for entity, valve_position in zip(self.config["action_space"], actions):
                self._setValvePosition(entity, valve_position)
                
    def _end_and_close(self):
        """
        Terminates the simulation
        """
        self.sim._model.swmm_end()
        self.sim._model.swmm_close()
class Swmm_model_rl:

    def __init__(self, config):#, action_params = None):
        self.config = yaml.load(open(config, "r"), yaml.FullLoader)
        self.sim = Simulation(self.config["model_folder"] +
                              self.config["model_name"] +".inp") # initialize simulation
        self.current_step = 0
        self.num_states = 3
        self.num_actions = 2
        
        self.sim.start()
        
        #self.num_states = 5
        #self.num_actions = 2
        self.state = np.array([0,1,1])
        
        # This could be changed
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_actions,), dtype=np.float32)
        
        # methods
        self.methods = {
            "depthN": self._getNodeDepth,
            "depthL": self._getLinkDepth,
            "volumeN": self._getNodeVolume,
            "volumeL": self._getLinkVolume,
            "flow": self._getLinkFlow,
            "flooding": self._getNodeFlooding,
            "inflow": self._getNodeInflow,
            "setting": self._getValvePosition,
            "total_precip": self._getRainfall
        }
        
        #self.action_params = action_params
        
        # create datalog
        self.data_log = {"time":[],
                         "flow": {}, "inflow": {}, "flooding": {}, 'depthN':{}, 'setting':{}, 'total_precip': {}}
        
        
        if self.config["states_for_computing_objectives"] is not None:
            for entity, attribute in self.config["states_for_computing_objectives"]:
                self.data_log[attribute][entity] = []
    
        if self.config["states"] is not None:
            for entity, attribute in self.config["states"]:
                self.data_log[attribute][entity] = []
    

    
    def _log_tstep(self):
        for attribute in self.data_log.keys():
            if attribute != "time" and len(self.data_log[attribute]) > 0:
                for entity in self.data_log[attribute].keys():# ID
                    self.data_log[attribute][entity].append(
                        self.methods[attribute](entity))
 
        
    # ------ Valve modifications -------------------------------------------
    def _getValvePosition(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.setting.value)

    def _setValvePosition(self, ID, valve):
        return self.sim._model.setLinkSetting(ID, valve)
        
    # ------ Node Parameters  ----------------------------------------------
    def _getNodeDepth(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.newDepth.value)
    
    def _getRainfall(self, ID):
        return self.sim._model.getGagePrecip(ID, tkai.RainGageResults.total_precip.value)   

    def _getNodeFlooding(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.overflow.value)

    def _getNodeLosses(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.losses.value)
    
    def _getNodeVolume(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.newVolume.value)

    def _getNodeInflow(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.totalinflow.value)

    def _setInflow(self, ID, value):
        return self.sim._model.setNodeInflow(ID, value)

    # ------ Link modifications --------------------------------------------
    def _getLinkDepth(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.newDepth.value)

    def _getLinkVolume(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.newVolume.value)

    def _getLinkFlow(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.newFlow.value)
        
       
    def _end_and_close(self):
        """
        Terminates the simulation
        """
        self.sim._model.swmm_end()
        self.sim._model.swmm_close()

    def reset(self):
        # Reset the environment to its initial state
        #self.current_step = 0
        # Reset to the current state
        #self.sim.start()
        
        self.state = np.array([0,1,1])
        
        self.time = self.sim._model.getCurrentSimulationTime()
        #self._end_and_close()
        return self.state,self.time

    def step(self, actions=None):
        #done = False
        #while done == False:
        
        if actions is not None:
            for asset, valve_position in zip(self.config["action_space"], actions):
                self._setValvePosition(asset, valve_position)
                    
        time = self.sim._model.swmm_step()
        print(self.current_step)
        self._log_tstep()
        self.data_log['time'].append(self.sim._model.getCurrentSimulationTime())

        states = self.config["states"].copy()

        
        state = []
        for s in states:
            entity = s[0] # name of object in swmm
            attribute = s[1] # attribute of interest (e.g. flow)
            state.append(self.methods[attribute](entity))


        
        next_state = np.array(state)
        
        # This will be removed for three objectives
        
        #sum_result = next_state
        #sum_result[3] = np.sum(next_state[3:5])

        # Remove the 5th element
        next_state = next_state[0:3]
        
        self.state = next_state
            


        self.current_step += 1
        done = False if time > 0 else True # time increases till the end of the sim., then resets to 0

    
        if done:
            #self._end_and_close()
            print('Episode completed')
                    
        return next_state, done
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.2, min_sigma=0.01, decay_period=8111):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0): 
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return random.sample(self.buffer, len(self.buffer))
        else:
            return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class Actor(tf.keras.Model):
    def __init__(self, num_actions):
        super(Actor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.10), kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.1))
        #self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.10), kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.1))
        #self.batch_norm2 = tf.keras.layers.BatchNormalization()
        #self.dense3 = tf.keras.layers.Dense(8, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.1))
        #self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='sigmoid', kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.1))

    def call(self, state):
        state_float32 = tf.cast(state, dtype=tf.float32)
        
        x = self.dense1(state_float32)
        #x = self.batch_norm1(x)
        x = self.dense2(x)
        #x = self.batch_norm2(x)
        #x = self.dense3(x)
        return self.output_layer(x)

class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.1))
        #self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.1))
        #self.batch_norm2 = tf.keras.layers.BatchNormalization()
        #self.dense3 = tf.keras.layers.Dense(8, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.1))
        #self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.output_layer = tf.keras.layers.Dense(1, activation='linear', kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.1))

    def call(self, state, action):
        state_float32 = tf.cast(state, dtype=tf.float32)
        x = tf.concat([state_float32, action], axis=-1)
        x = self.dense1(x)
        #x = self.batch_norm1(x)
        x = self.dense2(x)
        #x = self.batch_norm2(x)
        #x = self.dense3(x)
        #x = self.batch_norm3(x)
        return self.output_layer(x)

class DDPG:
    def __init__(self, num_states, num_actions, gamma=0.99, tau=0.001, actor_lr=0.00008, critic_lr=0.001, replay_buffer_capacity=100000):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau

        # Create actor and critic networks
        self.actor = Actor(num_actions)
        self.target_actor = Actor(num_actions)
        self.critic = Critic()
        self.target_critic = Critic()

        # Initialize target networks with the same weights as the online networks
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        # Define optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        #self.actor_optimizer = tf.keras.optimizers.RMSprop(actor_lr)
        #self.critic_optimizer = tf.keras.optimizers.RMSprop(critic_lr)
        
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        
    def get_action(self,state):
        
        action = self.actor.call(state)
        
        return action
        
    def update_target_networks(self):
        # Update target networks with a weighted average of the online and target network weights
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()

        for i in range(len(actor_weights)):
            target_actor_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * target_actor_weights[i]

        for i in range(len(critic_weights)):
            target_critic_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * target_critic_weights[i]

        self.target_actor.set_weights(target_actor_weights)
        self.target_critic.set_weights(target_critic_weights)

    def train(self, batch_size=8):
        
        batch = self.replay_buffer.sample(batch_size)
        
        state, action, reward, next_state, done = zip(*batch)
        
        state = np.array(state)
        next_state = np.array(next_state)
        action = np.array(action)
        reward = np.array(reward)
        
        done = tf.convert_to_tensor(done, dtype=tf.float32)

        # Update critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state)
            target_q_values = self.target_critic(next_state, target_actions)
            target_value = reward + self.gamma * target_q_values * (1 - done)
            predicted_value = self.critic(state, action)
            critic_loss = tf.keras.losses.MSE(target_value, predicted_value)

        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        # Update actor
        with tf.GradientTape() as tape:
            actor_actions = self.actor(state)
            actor_q_values = self.critic(state, actor_actions)
            actor_loss = -tf.reduce_mean(actor_q_values)

        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        # Update target networks
        self.update_target_networks()
    def get_actor_weights(self):
        return self.actor.get_weights()

    def get_critic_weights(self):
        return self.critic.get_weights()

    def get_target_actor_weights(self):
        return self.target_actor.get_weights()

    def get_target_critic_weights(self):
        return self.target_critic.get_weights()

tf.random.set_seed(23)
np.random.seed(23)
random.seed(23)


ddpg_agent = DDPG(3, 2)



num_episodes = 1

reward_list = []
Q_list = []

flooding_down = []

flooding_up = []


for episode in range(num_episodes):
    
    env = Swmm_model_rl(config=f"C:/Users/hk3sku/SWMM/swmm_models2/Config_Events3/Train_{episode+5} - Copyr.yaml")
    noise = OUNoise(env.action_space)
    state,time = env.reset()
    
    date_str = time.strftime("%Y-%m-%d")
    hour_str = time.strftime("%H:%M:%S")
    datetime_str = date_str + ' ' + hour_str
    
    filtered_forecast = df_dvlpd["norfolk_airport_total_precip"][df_dvlpd['time'].values == datetime_str]
    
    start_index = filtered_forecast.index[0]
    
    noise.reset()
    total_reward = 0
    done = False
    step = 0
    state[0] = df_dvlpd["norfolk_airport_total_precip"][start_index]
    
    while done == False:

        
        state = np.array([state[0]/rain_max,state[1]/6,state[2]/6])
        state = np.round(state, decimals=4)
        action = ddpg_agent.get_action(np.reshape(state, [1, env.num_states])).numpy()
        
        print("Current state is", state)
        
        action_list = action.tolist()[0]
        #action_list = noise.get_action(action_list, step)
        #action_list = action_list.tolist()
        
        #print("Current action is",action_list)
        
        
        print(f"Episode {episode + 1}, Step {step+1}: Chosen Action: {action_list}")
        
        next_state, done = env.step(actions=action_list)
        
        next_state[0] = df_dvlpd["norfolk_airport_total_precip"][start_index+step]
        
        down = (env.methods['flooding']('P1J')+ env.methods['flooding']('P2J'))/max_flooding_downstream
        depth = (env.methods['depthN']('P1')+ env.methods['depthN']('P2'))/2/6
        up = (env.methods['flow']('OvF1')+ env.methods['flow']('OvF2'))/max_flooding_upstream
        
        if down > 0 and up > 0:
            
            r1 = ss.norm.cdf(np.log(down), mu_down, sigma_down)
            r2 = ss.norm.cdf(np.log(up), mu_ovf, sigma_ovf)
            
            reward = -0.5 *r1 - 0.5 *r2
            
        elif down > 0 and up == 0:
            
            r1 = ss.norm.cdf(np.log(down), mu_down, sigma_down)

            reward = -0.5 * r1
            
        elif down == 0 and up > 0:
            
            r2 = ss.norm.cdf(np.log(up), mu_ovf, sigma_ovf)

            reward = -0.5 * r2
            
        else:
            
            reward = 0
            
        flooding_down.append(env.methods['flooding']('P1J')+ env.methods['flooding']('P2J'))
        flooding_up.append(env.methods['depthN']('P1')+ env.methods['depthN']('P2'))
    
        reward_list.append(reward)
        # Train the agent
        next_state = np.array([next_state[0]/rain_max,next_state[1]/6,next_state[2]/6])
        next_state = np.round(next_state, decimals=4)
        
        print("Next state is", next_state)
        
        ddpg_agent.replay_buffer.add((state, action_list, reward, next_state, done))
        
        ddpg_agent.train()
        
        # Get Q-values from the critic network
        q_values = ddpg_agent.critic(np.reshape(state, [1, env.num_states]), np.reshape(action_list, [1, env.num_actions])).numpy()
        print(f"Q-values: {q_values}")
        Q_list.append(q_values)
        
        #state_list.append(state)
        
        state = next_state
        total_reward += reward
        step += 1

        if done:
            print(done)
            env._end_and_close()
            break
            
    #Tot_reward.append(total_reward)
    
    print("Episode {}: Total Reward: {}".format(episode + 1, total_reward))
  
