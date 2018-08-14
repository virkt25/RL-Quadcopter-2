import numpy as np
from physics_sim import PhysicsSim

class TakeoffTask():
    def __init__(self, init_pose=None, init_velocities=None, init_angle_velocities=None, runtime=5., target_pos=None):
        self.init_pose = init_pose if init_pose is not None else np.array([0., 0., 0., 0., 0., 0.])
        self.sim = PhysicsSim(self.init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3
        
        ## State Size = Action Repeat * Size of Position Vector
        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 100.])
        
    def get_reward(self):
        distance = abs(np.linalg.norm(self.sim.pose[:3] - self.target_pos))
        reward = 10. - (0.1 * distance) + min(self.sim.pose[2], 100) + np.tanh(self.sim.v[2])
#         angle = abs(np.linalg.norm(self.sim.pose[3:6]))
#         reward += -0.01*angle
#         heightBonus = abs(self.sim.pose[2] - self.target_pos[2]) // 10
#         reward += min(heightBonus, 10)
#         reward = 2. - (self.sim.pose[0]**2 + self.sim.pose[1]**2)**0.5 + abs(self.sim.pose[2])
        return reward
    
    def step(self, rotor_speeds):
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # Update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done
    
    def reset(self):
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state

    

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 100.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        ## Take off -- avoid x,y deviation. promote z proximity. Promote z speed? Maintain Euler angles?
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state