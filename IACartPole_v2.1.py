#!/usr/bin/env python3
#############################################################################
# Filename    : IACartPole.py
# Description : maintenir un bâton en équilibre (cartPole) avec Intelligence Artificielle
# Author      : papsdroid.fr
# modification: 2019/11/14
#
# CARTPOLE V2.1: gestion des rewards en injectant des malus additionnels
#                - ecart par rapport an centre
#                - angle trop important
#                afin d'accélérer l'apprentissage en injectant de la connaissance à travers les rewards.
########################################################################

import tkinter as tk
import numpy as np
import gym
import matplotlib.pyplot as plt

import logging
logging.getLogger('tensorflow').disabled = True  # ceci élimine tous les warning inutiles de Tensorflow....

import tensorflow as tf

#Reinforcement learning: neural Q-network with policy gradient
#  IA 0 = basic, not learning (default value)
#  IA 1 = DQN with no knowledge (SCORE = REWARDS)
#  IA 2 = DQN with Knowledge injected in rewards
class DQN_learning():
    def __init__(self, IA=0):
        self.IA = IA                    # 0, 1 or 2: see above comment
        self.n_games_per_update = 10    # number of games done before calculating mean rewards
        self.episode_number = 0         # 1 episode = 1 game completed
        self.current_iteration = 0      # 1 iteration = n_games_per_update" games completed 
        self.reward=0                   # total rewards of the previous episode
        self.score=0                    # total score of the previous episode
        self.mean_reward=0              # mean rewards of previous iteration
        self.mean_score=0               # mean score of the previous iteration
        self.var_score=0                # variance score of the previous iteration
        self.current_rewards=[]         # all rewards from the current episode
        self.current_score=[]           # all records from the current episode
        self.all_rewards=[]             # sequences of rewards for each episode
        self.all_score=[]               # sequences of records for each episode
        self.scoreMAX = 200             # score max to reach that ends the learning
        self.mean_scores=[]             # scores from the current episode to calculate mean score
        self.data_score=[]              # historical of mean records for all episode
        self.learning = True            # False means learning is complete
        
        if self.IA>0:
            print('Inializing Neural Network, may take a while ....')
            tf.reset_default_graph()    # to compute gradients, graph needs to be reset to default
            self.graph = tf.Graph()
            self.n_inputs = 4           # inputs = observation(pos_X, velocity_X, angle, angle_volecoty) from IAgym env "CartPole-v0"
            self.n_hidden = 4           # hidden layer of the neural network
            self.n_outputs = 1          # output = probalility going left (action 0: accelerate left, acction 1: accelerate rigth)
            self.initializer = tf.variance_scaling_initializer()
            self.learning_rate = 0.01   # optimizer learning rate   
            self.discount_rate = 0.95   # aslo called "gamma" factor
            
            #neural network configuration
            self.X = tf.placeholder(tf.float32, shape=[None, self.n_inputs])
            self.hidden = tf.layers.dense(self.X, self.n_hidden, activation=tf.nn.elu, kernel_initializer=self.initializer)
            self.logits = tf.layers.dense(self.hidden, self.n_outputs)
            self.outputs = tf.nn.sigmoid(self.logits)  # probability of going left, 1-outputs = probability going right
            self.p_left_and_right = tf.concat(axis=1, values=[self.outputs, 1 - self.outputs])
            self.action = tf.multinomial(tf.log(self.p_left_and_right), num_samples=1) #random action 0 or 1, based on probalibility going left or rigth
        
            #gradient descent policy
            self.y = 1. - tf.to_float(self.action)
            self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.logits)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.grads_and_vars = self.optimizer.compute_gradients(self.cross_entropy)
            self.gradients = [grad for grad, variable in self.grads_and_vars]
            self.gradient_placeholders = []
            self.grads_and_vars_feed = []
            for grad, variable in self.grads_and_vars:
                gradient_placeholder = tf.placeholder(tf.float32)
                self.gradient_placeholders.append(gradient_placeholder)
                self.grads_and_vars_feed.append((gradient_placeholder, variable))
            self.training_op = self.optimizer.apply_gradients(self.grads_and_vars_feed)
            self.init = tf.global_variables_initializer()
            
            #TF session initialisation
            self.init = tf.initialize_all_variables()
            self.sess = tf.Session()
            self.sess.run(self.init)
            self.all_gradients=[]     # gradient saved at each step of each episode    
            self.current_gradients=[] # all grandients from the current episode

            print('Complete!')

        else:
            self.learning = False  # no learning at all

    #compute total rewards, given rewards and discount_rate 
    def discount_rewards(self, rewards):
        discounted_rewards = np.empty(len(rewards))
        cumulative_rewards=0
        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards * self.discount_rate
            discounted_rewards[step] = cumulative_rewards
        return discounted_rewards

    #nomalize rewards accros multiple episodes
    #return normalized score for each action in each episode
    #all_rewards [a,b,c], [d,e] ....
    #return: [a',b',c'], [d',e'] normalized scores
    def discount_and_normalize_rewards(self):
        all_discount_rewards = [self.discount_rewards(rewards) for rewards in self.all_rewards]
        flat_rewards = np.concatenate(all_discount_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discount_rewards]
    
    # Run policy either by learning (IA=True) or with basic policy (IA=False)
    def learn(self,env, obs):            
        #takes decision going left or rigth, based on policy (either IA or basic)
        if self.IA>0: #neural network decision
            action_val, gradients_val = self.sess.run([self.action, self.gradients], feed_dict={self.X: obs.reshape(1, self.n_inputs)})
            self.current_gradients.append(gradients_val)
            action=action_val[0][0]
        else: #basic policy: accelerate left when pole is leading to left, else accelerate rigth
            action = 0 if obs[2] < 0 else 1 #obs[2] = pole angle 
        
        #new observation, folowing action
        new_obs, reward, done, info = env.step(action)
        self.current_score.append(reward) #score = reward from the env
        #par defaut reward = 1, ajout de malus proportionnel à l'écart au centre et l'angle pour IA type 2
        if self.IA==2:
            reward -=  abs(new_obs[0])/3  + abs(new_obs[2])*4
            
        self.current_rewards.append(reward)
        self.score = np.sum(self.current_score)
        
        if done: #game completed
            self.all_rewards.append(self.current_rewards)
            self.all_score.append(self.current_score)
            self.reward = np.sum(self.current_rewards)
            self.mean_scores.append(self.score)
            self.current_rewards=[]   # all rewards from the current episode
            self.current_score=[]     # all score from the current episode

            #if self.learning and self.mean_score >= self.scoreMAX:
            #    print ("Learning Complete !")
            #    self.learning = False
                
            if self.learning:
                self.all_gradients.append(self.current_gradients)
                self.current_gradients=[] # all grandients from the current episode

            #reset the environnement
            self.obs = env.reset()
            
            #update params & learn after n_games_per_update games
            if (self.episode_number % self.n_games_per_update) == 0:
                #rewards mean
                self.mean_score = np.mean(self.mean_scores)
                self.var_score = np.std(self.mean_scores)
                self.data_score.append(self.mean_score)

                if self.IA>0:
                    print('IA{} - Itération {} - Mean score during {} games: {} - variance: {}'.format(
                        self.IA,self.current_iteration, self.n_games_per_update,
                        self.mean_score, self.var_score))
                self.mean_scores= []
                self.current_iteration += 1

                if self.learning :
                    self.all_rewards = self.discount_and_normalize_rewards()
                    feed_dict = {}
                    for var_index, grad_placeholder in enumerate(self.gradient_placeholders):
                        #multiply the gradient by the action score, and compute the mean
                        mean_gradients = np.mean(
                            [reward * self.all_gradients[game_index][step][var_index]
                            for game_index, rewards in enumerate(self.all_rewards)
                            for step, reward in enumerate(rewards)],
                            axis=0)
                        feed_dict[grad_placeholder] = mean_gradients
                    self.sess.run(self.training_op, feed_dict=feed_dict)
                    self.all_gradients=[]  # gradient saved at each step of each episode
                self.all_rewards=[]        # sequences of rewards for each episode
                self.all_score=[]          # sequences of score for each episode
            self.episode_number += 1  
        
        return action, new_obs, reward, done, info           
        
#terrain de jeu
class Terrain:
    def __init__(self, canv, height, width):
        self.canv=canv      # canvas sur lequel le terrain est déssiné
        self.h=height       # hauteur
        self.w=width        # largeur
        self.carts=[]       #liste des CartPoles
        self.carts.append(CartPole(self, cart_y=self.h * 90 // 100, nom='Simplet', IA=0)) #cartPole simple sans IA
        self.carts.append(CartPole(self, cart_y=self.h * 40 // 100, nom='IA', IA=2))      #cartPole avec IA
        self.dessine()

    def dessine(self):
        #supprime tous les objets du canvas
        self.canv.delete(tk.ALL)
        #dessine les CartPoles
        for c in self.carts:
            c.dessine()

#class décrivant un CartPole, hérite de DQN_learning
class CartPole(DQN_learning):
    #A pole is attached by an un-actuated joint to a cart, 
    #which moves along a frictionless track. 
    #The system is controlled by applying a force of +1 or -1 to the cart. 
    #The pendulum starts upright, and the goal is to prevent it from falling over. 
    #A reward of +1 is provided for every timestep that the pole remains upright. 
    #The episode ends when the pole is more than 15 degrees from vertical, 
    #or the cart moves more than 2.4 units from the center.
    def __init__(self, terrain, cart_y, nom, IA=0):
        DQN_learning.__init__(self,IA)
        self.terrain=terrain      
        self.nom = nom
        self.color = 'blue' if self.IA>0 else 'black'
        self.env = gym.make("CartPole-v0") #"CartPole" environnement de la bibliothèque gym
        self.cart_w = self.terrain.w // 12
        self.cart_h = self.terrain.h // 15
        self.cart_y = cart_y
        self.pole_len = self.terrain.h // 3.5
        self.pole_w = self.terrain.w // 80 + 1
        self.x_width = 2*2.4     #position maxi x du cartpole [-2.4 à 2.4]   
        self.angle_max = 0
        self.pos_max = -1
        self.pos_min = 1
        self.data_rewards=[0]    # historique des moyennes des rewards entre chaque batch
        self.etat_inital()
    
    def etat_inital(self):
        self.obs = self.env.reset()     # obervation de départ [Xposition, Xvelocity, Angle, Angle_velocity]
            
    def dessine(self):
        pos, vel, ang, ang_vel = self.obs
        if abs(ang) > self.angle_max:
            self.angle_max = abs(ang)
        cart_x = pos * self.terrain.w // self.x_width + self.terrain.w//2  #pos: [-x_width/2, +x_width/2]
        #limites max atteintes par le cart
        if pos > self.pos_max: 
            self.pos_max = pos 
        if pos < self.pos_min: 
            self.pos_min = pos
        cart_x_max = self.pos_max * self.terrain.w // self.x_width + self.terrain.w // 2   
        cart_x_min = self.pos_min * self.terrain.w // self.x_width + self.terrain.w // 2
        top_pole_x = cart_x + self.pole_len * np.sin(ang)
        top_pole_x_max = cart_x + self.pole_len * np.sin(self.angle_max)
        top_pole_x_min = cart_x + self.pole_len * np.sin(-self.angle_max)
        top_pole_y = self.cart_y - self.cart_h // 2 - self.pole_len * np.cos(ang)
        top_pole_y_max= self.cart_y - self.cart_h // 2 - self.pole_len * np.cos(self.angle_max)
        top_pole_y_min= self.cart_y - self.cart_h // 2 - self.pole_len * np.cos(-self.angle_max)
        #horizon principal
        self.terrain.canv.create_line(0, self.cart_y, self.terrain.w, self.cart_y, fill='black',width=4)
        #rectangle du cart avec son score
        self.terrain.canv.create_rectangle(cart_x - self.cart_w // 2, self.cart_y - self.cart_h // 2, 
                                   cart_x + self.cart_w // 2, self.cart_y + self.cart_h // 2, 
                                   fill = self.color)
        self.terrain.canv.create_text(cart_x, self.cart_y, text=self.mean_score, fill='white')
        self.terrain.canv.create_text(cart_x, self.cart_y + self.cart_h, 
                                      text="{}: iteration: {} episode {} score {}".format(self.nom, self.current_iteration, self.episode_number, self.score), fill='blue')
        #ligne du pole
        self.terrain.canv.create_line(cart_x, self.cart_y - self.cart_h // 2, top_pole_x, top_pole_y,
                              fill='orange', width=self.pole_w)
        #limites du pole
        self.terrain.canv.create_line(cart_x, self.cart_y - self.cart_h // 2, top_pole_x_min, top_pole_y_min,
                              fill='red', width=1, dash=(3,2))
        self.terrain.canv.create_line(cart_x, self.cart_y - self.cart_h // 2, top_pole_x_max, top_pole_y_max,
                              fill='red', width=1,dash=(3,2))
        #limites position du Cart
        self.terrain.canv.create_line(cart_x_min-self.cart_w // 2, self.cart_y,
                              cart_x_min-self.cart_w // 2, self.cart_y-self.cart_h,
                              fill='black', width=3)
        self.terrain.canv.create_line(cart_x_max+self.cart_w // 2, self.cart_y,
                              cart_x_max+self.cart_w // 2, self.cart_y-self.cart_h,
                              fill='black', width=3)
              
    
#interface graphique simple sous Tkinter
class Application:
    def __init__(self,height=400,width=600, delais=30, render=True): 
        self.height=height   #hauteur
        self.width=width     #largeur
        self.delais=delais   #délais de rafraichissement de l'animation
        self.flag_play = 0   #1=animation en cours
        self.nb_cycles = 0   #nb de cycles
        self.render = render #true = dessine chaque étape, false: dessine quand le socre de 200 est atteint

        #interface graphique    
        self.fen1 = tk.Tk()
        self.fen1.title('P@psDroid - IA CartPole')
        self.fen1.resizable(width='No', height='No')
        self.can1 = tk.Canvas(self.fen1, width=width, height=height, bg ='white')
        self.can1.pack(side=tk.TOP, padx=5, pady=5)
        #boutons GO et STOP, RENDER et QUIT
        self.b1 = tk.Button(self.fen1, text ='Go!', command =self.go)
        self.b2 = tk.Button(self.fen1, text ='Stop', command =self.stop)
        self.b3 = tk.Button(self.fen1, text ='Render', command =self.rendering)
        self.b4 = tk.Button(self.fen1, text ='Quit', command =self.quitter)
        self.b1.pack(side =tk.LEFT, padx =3, pady =3)
        self.b2.pack(side =tk.LEFT, padx =3, pady =3)
        self.b3.pack(side =tk.LEFT, padx =3, pady =3)
        self.b4.pack(side =tk.LEFT, padx =3, pady =3)
        #label nbcycles
        self.labcycles = tk.Label(self.fen1)
        self.labcycles.configure(text="Cycles:")
        self.labcycles.pack(side=tk.LEFT,padx =3, pady =3)
        self.labnbcycles = tk.Label(self.fen1)
        self.labnbcycles.configure(text="0")
        self.labnbcycles.pack(side=tk.LEFT,padx =3, pady =3)
        self.terrain = Terrain(self.can1, height, width)
        #boucle principale
        self.fen1.mainloop()
       
    def go(self):
        if self.flag_play==0:
            print("démarrage de l'animation")
            self.flag_play=1
            self.play()
        
    def stop(self):
        self.flag_play = 0
        print("arrêt de l'animation")
        self.plotGraph()
        
    def plotGraph(self):
        itMAX = len(self.terrain.carts[1].data_score)
        plt.clf()
        plt.title("Evolution des Rewards par itération")
        plt.xlabel('itérations')
        plt.ylabel('Rewards')
        for c in self.terrain.carts:
            plt.plot(c.data_score[:itMAX], label=c.nom)
        plt.legend()
        plt.show(block=False)

        
    def rendering(self):
        self.render = not(self.render) #inverse l'option de dessin
    
    def quitter(self):
        #closing sessions
        for c in self.terrain.carts:
            if c.IA>0:
                c.sess.close()
            c.env.close()
        self.fen1.destroy()
        self.fen1.quit()         

    def play(self):
        if (self.flag_play>0):          
            self.nb_cycles += 1 
            self.labnbcycles.configure(text=self.nb_cycles)
                
            #nouvelle action du CartePole
            for c in self.terrain.carts:
                action, c.obs, reward, done, info = c.learn(c.env, c.obs)

            if self.render:
                self.terrain.dessine()
                #plotting rewards graphics every 1000 cycles
                if (self.nb_cycles % 500 == 0) :
                    self.plotGraph()
                self.fen1.after(self.delais,self.play) 
            else:
                self.fen1.after(0,self.play)


#interface graphique    
appl = Application() #instancation d'un objet Application
