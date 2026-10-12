import fp_classes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import time as tm

from celluloid import Camera
from datetime import date
import datetime

now = date.today().strftime("%y-%m-%dT") + datetime.datetime.now().strftime("%H_%M_%S")



#T_XPOW2_DATAPOINTS = dict()
#T0 = tm.time()
#for i in range(learner.N_episodes):
#   #print(i)
#   if ((i%(learner.N_episodes//100)) == 0) and i:
#       progress = 1.*i/learner.N_episodes
#       print("------------------------------------------")
#       print(f"Fortschritt: {progress}")
#       Trem = (tm.time() - T0)*(1./progress - 1.)
#       print(f"Verbleibende Zeit: {Trem} s")
#       #print(i)
#   learner.x = 0
#
#   for t in range(learner.tmax_MSD):
#       if not t in T_XPOW2_DATAPOINTS.keys():
#           T_XPOW2_DATAPOINTS[t] = []
#       T_XPOW2_DATAPOINTS[t].append(learner.x**2)
#       
#       #AUFGABE: random_step here  
#       learner.random_step()
#
#
##exit()
#
#for key in T_XPOW2_DATAPOINTS.keys():
#   T_XPOW2_DATAPOINTS[key] =  np.mean(T_XPOW2_DATAPOINTS[key])
#
#X = np.array(list(T_XPOW2_DATAPOINTS.keys()))
#Y = np.array(list(T_XPOW2_DATAPOINTS.values()))
#plt.rcParams.update({'font.size': 18})
#
#fig,ax = plt.subplots()
#
#ax.plot(X,Y,label="Simulationsdaten")
#p = np.polyfit(X,Y,1)
#print(p)
#ax.plot(X,p[0]*X+p[1],linestyle="dashed",label="fit")
#ax.legend()
#ax.set_xlabel(r"$t$")
#ax.set_ylabel(r"$\left <x^2(t) \right >$")
#plt.show()
#
#D0 = learner.D
#D1 = p[0]/2
#graf_filename = f"Aufgabe_1_MSD_{D0}_{D1}_Kunstmann_{now}.png"
#fig.savefig(graf_filename, bbox_inches='tight')
#
#exit()
LAST_TRAJECTORY = []
steps = []
min_actions = []
P_obstacle_values = np.linspace(0.7, 0.8, num=21)
kappas = []
for p_obs in P_obstacle_values:
    env = fp_classes.environment()
    env.P_obstacle = p_obs
    learner = fp_classes.agent(env)
    total_action_displacement = 0
    for episode in range(learner.N_episodes):
        if ((episode%(learner.N_episodes//100)) == 0) and episode:
            progress = 1.*episode/learner.N_episodes
            print("------------------------------------------")
            print(f"Fortschritt: {progress}")

        # zufaellige startposition für jede episode
        #learner.x = np.random.randint(0, env.N_states)
        learner.x = env.starting_position
        learner.chosen_action = None

        # minimal moegliche schritte für diese startposition
        # zyklischer abstand zum ziel
        dist = abs(learner.x - env.target_position)
        cyclic_dist = min(dist, env.N_states - dist)
        # +1 für die abschließende "bleibe‑auf‑ziel"‑aktion
        min_actions.append(cyclic_dist + 1)

        step = 0
        while (learner.x != env.target_position) or learner.chosen_action != 1:
            if episode == learner.N_episodes-1: LAST_TRAJECTORY.append(learner.x)
            learner.x_old = learner.x
            learner.adjust_epsilon(episode)
            learner.choose_action()
            learner.random_step()
            learner.stoch_obstacle(env)
            learner.perform_action(env)
            if learner.epsilon == 0:
                total_action_displacement += (learner.chosen_action-1)
            learner.update_Q(env)
            step += 1
        steps.append(step)

    kappa = total_action_displacement/np.abs(total_action_displacement)
    kappas.append(kappa)
    print(f"Kappa: {kappa}")


plt.rcParams.update({"font.size": 12})
fig = plt.figure(figsize=(8, 5), layout="constrained")
plt.plot(P_obstacle_values, kappas, marker='o', linestyle='-', color='b', label=r'Gemessenes $\kappa$')

plt.title(r'Bestimmung des Übergangs: $\kappa$ als Funktion von $P_{obstacle}$', fontsize=12)
plt.xlabel('$P_{obstacle}$ (Wahrscheinlichkeit für Hindernis)', fontsize=11)
plt.ylabel(r'$\kappa$ (Übergangsparameter)', fontsize=11)

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.show()


#alter plot
#ratio = np.array(steps) / np.array(min_actions)
#plt.rcParams.update({"font.size": 12})
#fig = plt.figure(figsize=(8, 5), layout="constrained")
#plt.semilogy(range(len(ratio)), ratio, label='Steps / Minimal steps')
#plt.xlabel('Episode')
#plt.ylabel('Verhältnis (log scale)')
#plt.title('Verhältnis ausgeführte Schritte zu minimalen Schritten')
#plt.grid(True, which='both')
#plt.legend()
#graf_filename = f"Aufgabe_3_HYPERPARAMETER_stepVergleich_a0.999999_e500{now}.png"
##graf_filename = f"Aufgabe_3_HYPERPARAMETER_stepVergleich_a1{now}.png"
#fig.savefig(graf_filename, dpi=150, bbox_inches=None)
#plt.show()

im_agent = plt.imread("../IMG/Lasse_head.png")
im_treat = plt.imread("../IMG/treat.png")
imagebox = OffsetImage(im_agent, zoom=0.02)
imagebox2 = OffsetImage(im_treat, zoom=0.5)

trajfig,trajax = plt.subplots()
camera = Camera(trajfig)


f = open('Q_MATRIX_' + now + '.txt', 'w')
f.write(str(learner.Q))
f.close()
#--------------------------------------------------------------------
#IM FOLGENDEN CODE BLOCK WIRD DIE TRAJEKTORIE ALS GIF AUSGEGEBEN
#WENN DIESER TEIL ZU LANGE DAUERT ODER SIE DIE NÖTIGEN LIBRARIES NICHT INSTALLIEREN KÖNNEN,
#KOMMENTIEREN SIE DIESEN CODE BLOCK AUS
#--------------------------------------------------------------------


LAST_TRAJECTORY_WITHOUT_PBJUMPS = []
for i in range(len(LAST_TRAJECTORY)-1):
    dx = np.abs(LAST_TRAJECTORY[i+1] - LAST_TRAJECTORY[i])
    LAST_TRAJECTORY_WITHOUT_PBJUMPS.append(LAST_TRAJECTORY[i])
    if dx > 0.5*env.N_states:
        LAST_TRAJECTORY_WITHOUT_PBJUMPS.append(None)
        

for i,x in enumerate(LAST_TRAJECTORY_WITHOUT_PBJUMPS):
    trajax.plot(np.array(LAST_TRAJECTORY_WITHOUT_PBJUMPS)[:i],np.linspace(0,0,len(np.array(LAST_TRAJECTORY_WITHOUT_PBJUMPS)[:i])),color="blue") 
    ab = AnnotationBbox(imagebox2, (env.target_position, 0.0),frameon=False)
    trajax.add_artist(ab)
    if x:
        ab = AnnotationBbox(imagebox, (x, 0.0),frameon=False)
        #trajax.add_artist(ab)
    else:
        ab = AnnotationBbox(imagebox, (LAST_TRAJECTORY_WITHOUT_PBJUMPS[i+1], 0.0),frameon=False)
    trajax.add_artist(ab)
            

    trajax.set_xlim(0.0,env.N_states)
    trajax.set_ylim(-1.0,1.0)
    trajax.set_yticks([])
    camera.snap()
animation = camera.animate()        
animation.save("LAST_TRAJECTORY_" + now + ".gif")
