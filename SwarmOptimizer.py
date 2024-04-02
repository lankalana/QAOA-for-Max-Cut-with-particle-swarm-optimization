import numpy as np

class SwarmOptimizer:
    # Initialize some key parameter. Optionally store the range for the optimized parameters and maximum velocity for the particles
    def __init__(self, func, varRanges = None, Vmax = None, seed=None):
        self.func = func
        self.rng = np.random.default_rng(seed=seed)

        self.varRanges = varRanges
        self.Vmax = Vmax

    # Initialize the swarm
    def InitSwarm(self):
        numParams = len(self.bestGlobalPos)
        # Initialize some array for storing the positions, velocities, etc.
        self.particlePos = np.zeros((self.N, numParams))
        self.particleVelocity = np.zeros((self.N, numParams))
        self.bestParticlePos = np.zeros((self.N, numParams))
        self.bestParticleVal = np.zeros(self.N)
        self.bestGlobalPos = np.zeros(numParams)
        self.bestGlobalVal = 0
        # Store the iterated range for the optimization step
        self.particleRange = range(self.N)

        # Initialize all particles to a random position within the given variable range or 0 and 1
        for i in self.particleRange:
            if self.varRanges is None:
                self.particlePos[i] = self.rng.random(numParams)
            else:
                self.particlePos[i] = self.rng.random(numParams) * np.copy(self.varRanges[1, :])

            # Give particles random velocities between -Vmax and Vmax or -0.5 and 0.5
            if self.Vmax is None:
                self.particleVelocity[i] = self.rng.random(numParams) - 0.5
            else:
                self.particleVelocity[i] = self.rng.random(numParams) * self.Vmax - self.Vmax / 2
            self.bestParticlePos[i] = np.copy(self.particlePos[i])
    
    def OptimizationStep(self, iter):
        for i in self.particleRange:
            r = 2 * self.rng.random(2)

            # Calculate new velocity
            self.particleVelocity[i] = self.inertia[iter] * self.particleVelocity[i] + self.C1 * r[0] * (self.bestParticlePos[i] - self.particlePos[i]) + self.C2 * r[1] * (self.bestGlobalPos - self.particlePos[i])

            # If Vmax was given, clip the speed to be within Vmax
            if (self.Vmax != None):
                absoluteV = np.sqrt(self.particleVelocity[i].dot(self.particleVelocity[i]))
                if (absoluteV > self.Vmax):
                    self.particleVelocity[i] = (self.particleVelocity[i] / absoluteV) * self.Vmax

            # Update the position of the particle
            self.particlePos[i] = self.particlePos[i] + self.particleVelocity[i]
            
            # If variable limits were given, check is the particle within them. If not, invert the velocity and clamp the position to the limits
            if not (self.varRanges is None):
                invertV = False
                for j, pj in enumerate(self.particlePos[i]):
                    if (pj > self.varRanges[1, j]):
                        self.particlePos[i,j] = self.varRanges[1, j]
                        invertV = True
                    elif (pj < self.varRanges[0, j]):
                        self.particlePos[i,j] = self.varRanges[0, j]
                        invertV = True
                if (invertV):
                    self.particleVelocity[i,j] *= -1

            # Compute the function value
            val = self.func(self.particlePos[i])

            # Check if the current position is a new best
            if (val > self.bestParticleVal[i]):
                self.bestParticleVal[i] = val
                self.bestParticlePos[i] = np.copy(self.particlePos[i])

            # Check if the current position is a new global best
            if (val > self.bestGlobalVal):
                self.bestGlobalVal = val
                self.bestGlobalPos = np.copy(self.particlePos[i])

    def Optimize(self, initialParams, N, iters, inertiaStart, inertiaEnd, C1, C2):
        # Store linearly decreasing intertia values
        self.inertia = np.linspace(inertiaStart, inertiaEnd, iters)
        self.C1 = C1
        self.C2 = C2
        self.N = N
        self.bestGlobalPos = initialParams
        # Initialize the swarm
        self.InitSwarm()
        calls = 0
        
        # Optimize iters times
        for i in range(iters):
            self.OptimizationStep(i)
            calls += self.N

        # Print and return the result
        print(f'Best value of {self.bestGlobalVal:.3f} found at:\n\t{self.bestGlobalPos}')
        return self.bestGlobalPos