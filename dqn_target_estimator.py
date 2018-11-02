'''
DQN using two seperate target and estimator networks
'''

# importing the dependencies
import numpy as np # linear algebra
import pandas as pd # dataframe
import tensorflow as tf # Machine learning
from glob import glob # file handling
from tqdm import tqdm # progress bar
from collections import deque # for simpler implementation of memory

# This is a DQN with estimator and target networks
class DQNetwork():
    # initialization function
    def __init__(self, input_dim, scope, is_eval = False):
        # input_dim: state size
        # is_eval: is being evaluated
        # scope: scope of the model
        self.state_size = input_dim
        self.action_space = 3 # sell, sit, buy
        self.scope = scope # name of scope

        # code to load previously trained model --> skip for now
        # self.model = load_model("models/" + model_name) if is_eval else self._model()
        self._build_model()
        self._build_loss()

    # inhouse functions
    def _build_model(self):
        self.input_placeholder = tf.placeholder(tf.float32, [None, self.state_size], name = 'inputs')
        self.q_placeholder = tf.placeholder(tf.float32, [None, self.action_space], name = 'q_value')

        # layers
        h1 = tf.contrib.layers.fully_connected(self.input_placeholder, 64)
        h2 = tf.contrib.layers.fully_connected(h1, 32)
        h3 = tf.contrib.layers.fully_connected(h2, 8)
        self.action_pred = tf.contrib.layers.fully_connected(h3, self.action_space, activation_fn = tf.nn.softmax)

    def _build_loss(self):
        self.loss = tf.reduce_mean(tf.square(self.q_placeholder - self.action_pred))
        self.update_step = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        
    # user callable functions
    def save_model(self, frozen = False):
        # code to save the model as frozen inference graph (optional) or 
        # checkpoint file (default)
        if frozen:
            # convert to frozen graph here
            model_path = 'some_path'
            print('Model saved at {0}'.format(some_path))
            return
        return
    
    # no need for operation functions since everything is called from outside

# function to properly return the string of price
def format_price(price):
    return ("-$" if price < 0 else "$") + "{0:.2f}".format(abs(price))

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# function to get the state
def get_state(data, t, n):
    d = t - n + 1
    if d >= 0:
        block = data[d:t+1]
    else:
        # pad with t0
        block = -d*[data[0]] + data[0:t+1].tolist()
        
    # block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
    
    # get results
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    
    # return numpy array
    return np.array([res])

# functions to copy parameters between two networks
def copy_parameters(q_network, target_network, sess):
    # q_network (source) to target_network (target)
    # sess: tensorflow session
    
    # source
    source_params = [t for t in tf.trainable_variables() if t.name.startswith(q_network.scope)]
    source_params = sorted(source_params, key = lambda v: v.name)
    
    # target
    target_params = [t for t in tf.trainable_variables() if t.name.startswith(target_network.scope)]
    target_params = sorted(target_params, key = lambda v: v.name)
    
    # do assign operations in loop
    for s_v, t_v in zip(source_params, target_params):
        op = t_v.assign(s_v)
        sess.run(op)

def train_dqn(q_network,
              target_network,
              sess,
              data,
              max_mem_size = 1000,
              num_episodes = 50,
              train_target_every = 5,
              gamma = 0.99,
              epsilon_start = 0.99,
              epsilon_end = 0.001,
              epsilon_decay = 0.995):
    # function variables
    train_global_step = 0 # global step needed in parameter update
    train_loss = [] # training loss in each episode
    train_profits = [] # for profits in each episode
    
    # memory_buffer
    memory_buffer = []
    
    # initialize variables
    sess.run(tf.global_variables_initializer())
    
    # init stuff
    epsilon = epsilon_start
    
    # iterate over each episode
    for ep in range(num_episodes):
        # for each training episode
        state = get_state(data, 0, window_size + 1)
        
        # init values for new episode
        total_profit = 0.0 # total profit in this episode
        # q_network.inventory = [] # holdings by q_network
        inventory = [] # inventory for this episode
        ep_loss = [] # total loss in this episode
        
        for t in tqdm(range(len_data)):
            # take action according to epsilon greedy policy
            if np.random.random() > epsilon:
                action = np.random.randint(q_network.action_space)
            else:
                feed_dict = {q_network.input_placeholder: state}
                action = sess.run(q_network.action_pred, feed_dict = feed_dict)
                action = np.argmax(action[0])
            
            # next state
            next_state = get_state(data, t + 1, window_size + 1)
            reward = 0
            
            # now go according to the actions
            if action == 2:
                # buy
                inventory.append(data[t])
                if LOG:
                    print('Buy:' + format_price(data[t]))
                    
            elif action == 0 and len(inventory) > 0:
                bought_price = inventory.pop(0) # remove the first element and return the value
                profit = data[t] - bought_price # profit this transaction
                reward = max(data[t] - bought_price, 0) # reward
                total_profit += profit # add to total profit
                if LOG:
                    print("Sell: " + format_price(data[t]) + " | Profit: " + format_price(profit))
                    
            # condition for done
            done = t == len_data - 1
            
            # add to memory and make sure it's of fixed size
            memory_buffer.append((state, action, reward, next_state, done))
            if len(memory_buffer) > max_mem_size:
                memory_buffer.pop(0)
            
            # update state
            state = next_state
            
            # train the model
            if len(memory_buffer) > batch_size:
                
                # sample minibatches here
                mini_batch = memory_buffer[-batch_size:]
                    
                # calculate q_value and target_values
                for state_t, action_t, reward_t, next_state_t, done_t in mini_batch:
                    # condition for calculating y_j
                    if done_t:
                        target_pred = reward
                    
                    else:
                        feed_dict = {target_network.input_placeholder: next_state_t}
                        target_pred = sess.run(target_network.action_pred, feed_dict = feed_dict)
                        target_value = reward_t + gamma*np.amax(target_pred[0])
                    
                    # q_value
                    feed_dict = {q_network.input_placeholder: state_t}
                    q_values = sess.run(q_network.action_pred, feed_dict = feed_dict)
                    q_values[0][action_t] = target_value
                    
                    # drop epsilon value after every action taken
                    if epsilon > epsilon_end:
                        epsilon *= epsilon_decay
                    
                    # update the q_network parameters
                    feed_dict = {q_network.input_placeholder: state_t,
                                 q_network.q_placeholder: q_values}
                    loss, _ = sess.run([q_network.loss, q_network.update_step], feed_dict = feed_dict)
                    
                    # update the lists
                    ep_loss.append(loss)
                    
                    # update target network
                    train_global_step += 1
                    if ep % train_target_every == 0:
                        copy_parameters(q_network, target_network, sess)
                        
        # update the outer values
        train_loss.append(ep_loss)
        train_profits.append(total_profit)
        
        # print val
        print('[*]Episode: {0}, loss: {1}, profits: {2}, epsilon: {3}'\
              .format(ep + 1, np.sum(train_loss[-1]), train_profits[-1], epsilon))
    
    # return the values
    return train_loss, train_profits

# SCRIPT #
def main():
    # load the csv file
    path_folder = './data_raw/'
    file_path = glob(path_folder + '*.csv')[0]
    stock_name = file_path.split('.')[-2].split('/')[-1]
    data = pd.read_csv(file_path)

    # constants
    LOG = False
    episode_count = 10
    window_size = 100
    data = data['Close'].values # what our data is
    len_data = len(data) - 1 # total information length
    batch_size = 32 # minibatch size

    # logs
    loss_global = []
    profits_global = []

    # run the model
    q_network = DQNetwork(window_size, 'q_network')
    target_network = DQNetwork( window_size, 'target_network')
    sess = tf.Session()
    loss, profits = train_dqn(q_network, target_network, sess, data)

if __name__ == '__main__':
    main()
