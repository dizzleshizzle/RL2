import numpy as np

class environment:
    def __init__(self):
        self.N_states = 100
        self.target_position = 8
        self.starting_position = 30
        
        self.obstacle_interval = np.arange(9,12)
        self.P_obstacle = 0.0
    
class agent:
    def __init__(self,env_):
        self.N_episodes = 10000       #10000
        self.tmax_MSD = 100
        
        self.x = 1
        self.Q = np.zeros((env_.N_states,3))
        self.alpha = 0.01
        self.gamma = 0.9
        self.epsilon = 1.0
        self.target_reward = 10.0
        self.zero_fraction = 0.9

        self.D = 0.125
        if self.D > 0.5:
            print(f"Diffusionskonstante D={self.D} überschreitet den Maximalwert 0.5 für die gewählte Schrittgröße. Wähle ein kleineres D.")
            exit()
        self.P_diffstep = 2 * self.D
        
        self.x_old = None
        
        if self.P_diffstep is not None and self.P_diffstep > 1.0:
            print(f"self.P_diffstep = {self.P_diffstep} > 1.0 in agent.__init__(...)")
            print("Probability self.P_diffstep cannot exceed 1.0. Choose a smaller value.")
            exit()
    
    
    def random_step(self):
        if np.random.rand()<self.P_diffstep:
            self.x+=2*np.random.randint(0,2) -1
        
    def adjust_epsilon(self,episode):
        if(episode<self.zero_fraction*self.N_episodes):
            self.epsilon = 1-(episode/(self.N_episodes*self.zero_fraction))
        else:
            self.epsilon = 0
            
    def choose_action(self):
        """
        wählt eine Zufallsaktion aus mit Wahrscheinlichkeit self.epsilon oder falls zwei Aktionen die höchsten Q-Werte haben.
        Andernfalls wird der höchste Wert in der jeweiligen Zeile ausgewählt.
        """

        if np.random.rand()<self.epsilon:
            self.chosen_action= np.random.randint(0,3)
        else:
            zeile = self.Q[self.x,:]
            maximum = np.max(zeile) 
            maximum_action = np.where(zeile == maximum)[0] 
            self.chosen_action = np.random.choice(maximum_action)

    def perform_action(self,env_):
        """
        Hier werden die Aktionen ausgeführt. Der Index der Aktion entspricht der Verschiebung auf der x-Achse + 1
        """

        self.x = (self.x + (self.chosen_action -1))%env_.N_states

    
    def update_Q(self,env_):
        """
        Hier werden die Werte der Q-Matrix nach jeder Aktion entsprechend aktualisiert
        """

        if (self.x == env_.target_position) and (self.chosen_action == 1):
            R = self.target_reward
        else:
            R = 0
        self.Q[self.x_old,self.chosen_action] = self.Q[self.x_old,self.chosen_action] +self.alpha*(R+self.gamma*np.max(self.Q[self.x,:])-self.Q[self.x_old,self.chosen_action])




    def stoch_obstacle(self,env_):
        pass

