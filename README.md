# project_robotic_farkas

<b> Initial commit </b>

Run *simpleTest.py* with V-REP open to see if this works for you.

<b> First scene </b>

In this commit the first scene is added featuring simple sphere, robotic arm and main_camera (*test.py*).

The following questions seem relevant:
- What exactly is the input that our system will recieve? Is it necessary to have 2 cameras on scene, or is our goal 
just to learn the arm based on simple input (e.g. sphere coordinates relative to camera (eyes)) and visual preprocessing (that we don't 
have to do) will take care fo the rest?
- Is feedback arm or camera driven? What I mean by that is if we should observe arm location based on camera input or 
arm position (or maybe both)?
- Should we use the actual robotic arm that will be used in experiments?

<b> Changes in scene & short TO-DO list </b>

There is a robotic arm on the scene now, that resembles the real arm in the lab more accurately.

TO-DO list:
- implement test arm movements. Take a look at RemoteAPI functions *simxSetJointPosition* and *simxSetJointTargetPosition*.
Checking script that is prewritten for the robotic arm on scene could also be useful. One thing to figure out 
might also be the principle upon which arm actually moves. Is it necessary to start a simulation first or not? How would the 
simulation workflow look like? Checking the code provided by Farkaš might also help answer some of these questions.
- check out the papers Farkaš sent and implement the algorithm. It might be a good idea to check out some libraries 
that are capable of reinforcement learning.

<b> New questions </b>

There are 2 new files available. File *cacla.py* consists of main program and learning algorithm, while *arm.py*
implements code needed for manipulation of the arm. 

While reading and transforming the CACLA code, following questions emerged:
- There may be a need to clarify what discount function is. 
- State vs. Change distinction?

I haven't fully gotten the code yet, so I don't know when TODOs added will need to be implemented. 
I also removed all 1D and 2D functions to remove clutter. Additional code cleaning might still be necessary.

