import os
import math
import random
import numpy as np

import tensorflow as tf

from gym_2048.envs.game2048_env import Game2048Env

class ReplayMemory:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.reset()

    def append(self, data):
        self.buf[self.index] = data
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen

    @property
    def isFull(self):
        return self.length == self.maxlen

    def sample(self, batch_size, with_replacement=True):
        if with_replacement:
            indices = np.random.randint(self.length, size=batch_size) # faster
        else:
            indices = np.random.permutation(self.length)[:batch_size]

        cols = [[], [], [], [], []]  # state, action, reward, next_state, continue
        for memory in self.buf[indices]:
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)

    def reset(self):
        self.buf = np.empty(shape=self.maxlen, dtype=np.object)
        self.index = 0
        self.length = 0


class Stats:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.reset()

    def append(self, score, highest):
        self.score[self.index] = score
        self.highest[self.index] = highest
        self.index = (self.index + 1) % self.maxlen
        self.total_game += 1
        self.reach2048 += 1 if highest == 2048 else 0
        self.highest_reached = max(self.highest_reached, highest)

    def getStat(self):
        return [self.score.mean(),
                self.highest.mean(),
                self.total_game,
                self.reach2048,
                self.highest_reached]

    def reset(self):
        self.score = np.zeros(shape=self.maxlen, dtype=np.int32)
        self.highest = np.zeros(shape=self.maxlen, dtype=np.int32)
        self.index = 0
        self.total_game = 0
        self.reach2048 = 0
        self.highest_reached = 0

class DQN:
    def __init__(self):
        self.stat = Stats(maxlen = 10)
        self.replay_memory_size = 50000
        self.replay_memory = ReplayMemory(maxlen = self.replay_memory_size)

        self.n_input = [None, 256]
        self.n_hidden = [100, 100, 100]
        self.names = ["H1", "H2", "H3"]
        self.n_outputs = 4

        self.learning_rate = 0.01
        self.momentum = 0.95
        self.discount_rate = 0.95
        self.batch_size = 50

        self.eps_min = 0.1
        self.eps_max = 1.0
        self.eps_decay_steps = 200000

        self.n_steps = self.eps_decay_steps * 2
        self.training_start = self.replay_memory_size  # start learning only when replay memory is full
        self.training_interval = 1

        self.save_steps = 1000
        self.copy_steps = 3000

        self.loss_val = np.infty
        self.game_length = 0
        self.total_max_q = 0
        self.mean_max_q = 0.0

        self.config = tf.ConfigProto(device_count = {'GPU': 0})

        self.checkpoint_path = "F:/training_data/DQN/Pacman/my_dqn_2048.ckpt"

        self.reset_graph()

    def reset_graph(self, seed=42):
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        np.random.seed(seed)

        self.replay_memory.reset()
        self.stat.reset()

        self.initializer = tf.variance_scaling_initializer()

        self.X_state = tf.placeholder(tf.float32, shape=self.n_input)
        self.actor_Q_values, self.weight_actor = self.createModel(name = "Actor", prev_layer = self.X_state)
        self.critic_Q_values, self.weight_critic = self.createModel(name="Critic", prev_layer = self.X_state)

        self.transfertLearning = self.CopyCriticToActor()

        # with tf.variable_scope("train"):
        self.X_action = tf.placeholder(tf.int32, shape=[None])  # action taken (shape batch x 1)
        self.y = tf.placeholder(tf.float32, shape=[None, 1])       # Q-value computed (shape batch x 1)
        self.pred_q_value = tf.reduce_sum(self.actor_Q_values * tf.one_hot(self.X_action, self.n_outputs), axis=1,
                                keepdims=True)         # element-wise product Q-value x Action_OHE(batch x 4)

        # clipped loss
        error = tf.abs(self.y - self.pred_q_value)
        clipped_error = tf.clip_by_value(error, -1.0, 1.0)
        linear_error = 2 * (error - clipped_error)
        self.loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum, use_nesterov=True)
        self.training_operation = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def CopyCriticToActor(self):
        copy_operations = [target_var.assign(self.weight_actor[var_name])
            for var_name, target_var in self.weight_critic.items()]
        return tf.group(*copy_operations)

    def createModel(self, name, prev_layer):
        with tf.variable_scope(name) as scope:
            for name_layer, n_unit in zip(self.names, self.n_hidden):
                prev_layer = tf.layers.dense(inputs=prev_layer,
                                             units=n_unit,
                                             activation=tf.nn.relu,
                                             # name=name+ "/" + name_layer,
                                             kernel_initializer=self.initializer)
        outputs = tf.layers.dense(inputs=prev_layer,
                                  units=self.n_outputs,
                                  kernel_initializer=self.initializer)
        tensors = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        dict_weight = {}
        for var in tensors:
            dict_weight[var.name[len(scope.name):]] = var  # keep only the end the name
        return outputs, dict_weight

    def epsilon_greedy(self, q_values, step):
        epsilon = max(self.eps_min, self.eps_max - (self.eps_max - self.eps_min) * step / self.eps_decay_steps)
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_outputs)  # random action
        else:
            return np.argmax(q_values)  # optimal action

    def preprocess_observation(self, obs):
        result = np.zeros((16, 4, 4), dtype=np.uint8)
        for i, value in enumerate(obs):
            if value != 0:
                depth = int(math.log(value, 2) - 1)
                result[depth, i % 4, i // 4] = 1
        return result.flatten()

    def train(self):
        env = Game2048Env()
        done = True  # to directly reset the game

        with tf.Session(config=self.config) as sess:
            if os.path.isfile(self.checkpoint_path + ".index"):
                self.saver.restore(sess, self.checkpoint_path)
            else:
                self.init.run()
                self.transfertLearning.run()

            step = self.global_step.eval()
            iter = 0
            while step < self.n_steps:
                step = self.global_step.eval()

                if done:  # game over, start again
                    self.stat.append(env.score, env.highest())
                    obs = env.reset()
                    state = self.preprocess_observation(obs)

                avg_score, avg_highest, count_game, count_success, highest_reached = self.stat.getStat()
                print("\rIter {}\tTraining step {}/{} ({:.1f})%\tAVG Score {}"
                      "\tAVG Highest {}\tGame {}\t Win {}\t Best {}\t""".format(
                        iter, step, self.n_steps, step * 100 / self.n_steps, avg_score, avg_highest,
                        count_game, count_success, highest_reached), end="")

                q_values = self.actor_Q_values.eval(feed_dict={self.X_state: [state]})
                action = self.epsilon_greedy(q_values, step)

                obs, reward, done, info = env.step(action)
                next_state = self.preprocess_observation(obs)

                if reward >= 0: # we don't store if there is no move
                    self.replay_memory.append((state, action, reward, next_state, 1.0 - done))

                state = next_state
                iter += 1
                if self.replay_memory.isFull:
                    X_state_val, X_action_val, Rewards, X_next_state_val, Continues = self.replay_memory.sample(self.batch_size)
                    next_q_values = self.critic_Q_values.eval(feed_dict={self.X_state: X_next_state_val})
                    y_val = Rewards + Continues * self.discount_rate * np.max(next_q_values, axis=1, keepdims=True)

                    # Train the online DQN
                    _, loss_val, b = sess.run([self.training_operation, self.loss, self.pred_q_value ], feed_dict={
                        self.X_state: X_state_val, self.X_action: X_action_val, self.y: y_val})

                    # Regularly copy the online DQN to the target DQN
                    if step % self.copy_steps == 0:
                        self.transfertLearning.run()

                    if step % self.save_steps == 0:
                        self.saver.save(sess, self.checkpoint_path)

    def play(self, n_iter = 1):
        env = Game2048Env()
        i = 0
        with tf.Session(config=self.config) as sess:
            self.saver.restore(sess, self.checkpoint_path)
            for i in range(n_iter):
                obs = env.reset()
                state = self.preprocess_observation(obs)
                while True:
                    q_values = self.actor_Q_values.eval(feed_dict={self.X_state: [state]})
                    action = np.argmax(q_values)
                    obs, reward, done, info = env.step(action)
                    if reward < 0:
                        i+=1
                    if i == 10:
                        break
                    state = self.preprocess_observation(obs)
                    print(action, reward)
                    # env.render()
                    if done:
                        print("Game {} - Max Reached = {} - Score {}".format(i, env.highest(), env.score))
                        break
            env.close()


if __name__ == "__main__":
    dqn = DQN()
    dqn.train()
    dqn.play(n_iter = 1)