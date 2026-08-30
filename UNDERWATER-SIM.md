# About

Simulating stuff in Pinocchio:
- Hydrodynamic Drag forces
- Buoyancy effects

## Steps

https://gemini.google.com/app/9dcf0c6c5af19b2a

1. Get Link Velocities:
Use pinocchio.forwardKinematics and pinocchio.getFrameVelocity to find the Cartesian velocity ($v_{link}$) and position ($z_{link}$) of the center of mass of each UR5 link (upper arm, forearm, wrists).

2. Calculate Fluid Velocity:
Based on the link's height ($z_{link}$), calculate $v_{fluid}$ using either your Uniform or Shearing equations.

3. Calculate Relative Velocity:
Subtract fluid velocity from link velocity.

4. Compute Drag Wrench:
Calculate the drag force. Because a robot link is a 3D object, drag acts as a "Wrench" (3 linear forces, 3 angular moments). You can approximate the UR5 links as cylinders.

5. Inject into Pinocchio:
Create a list of pinocchio.Force objects (one for each joint). Pass this list as the fext (external forces) argument into pinocchio.aba().