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
		self.N_episodes = 20000
		self.tmax_MSD = 100
		
		self.x = 1
		self.Q = np.zeros((env_.N_states,3))
		self.alpha = None
		self.gamma = None
		self.epsilon = 1.0
		self.target_reward = 10.0
		self.zero_fraction = 0.9

		self.D = 0.45
		# Physikalische Einschränkung: Für ein Gitter mit Abstand a=1 und Zeitschritt τ=1
		# ist die maximal mögliche Diffusionskonstante für ein einfaches ±1‑Schritt‑Modell
		# D_max = a²/(2·τ) = 0.5. Ist ein größerer Wert von D gefordert, wird das Programm beendet.
		if self.D > 0.5:
			print(f"Diffusionskonstante D={self.D} überschreitet den Maximalwert 0.5 für das gewählte Schrittgröße. Wähle einen kleineren D.")
			exit()
		# Setze die Bewegungswahrscheinlichkeit gemäß D = P_move/2  →  P_move = 2·D.
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
		pass
		    
	def choose_action(self):
		"""
		wählt eine Zufallsaktion aus mit Wahrscheinlichkeit self.epsilon oder falls zwei Aktionen die höchsten Q-Werte haben.
		Andernfalls wird der höchste Wert in der jeweiligen Zeile ausgewählt.
		"""
		pass
	def perform_action(self,env_):
		"""
		Hier werden die Aktionen ausgeführt. Der Index der Aktion entspricht der Verschiebung auf der x-Achse + 1
		"""

		pass
	
	def update_Q(self,env_):
		"""
		Hier werden die Werte der Q-Matrix nach jeder Aktion entsprechend aktualisiert
		"""
		pass
	
	def stoch_obstacle(self,env_):
		pass

