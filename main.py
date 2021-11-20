import numpy as np
from gridgame import GridgameEnv



def state_value_iteration(env, theta=0.0001, discount_factor=0.8):

    def one_step_action_choice(state, V):
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A


    V = np.zeros(env.nS)
    while True:

        delta = 0

        for s in range(env.nS):
            # find the best action
            A = one_step_action_choice(s, V)
            best_action_value = np.max(A)
            # Calculate terminate condition
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function
            V[s] = best_action_value        
        # Check if we can stop 
        if delta < theta:
            break

    policy = np.zeros([env.nS, env.nA])

    for s in range(env.nS):
        A = one_step_action_choice(s, V)
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0
    return policy, V


if __name__ == '__main__':
	env = GridgameEnv()
	policy , V = state_value_iteration(env)
	print("Grid Policy (0=up, 1=right, 2=down, 3=left):")
	print(np.argmax(policy, axis=1))
	print("")

