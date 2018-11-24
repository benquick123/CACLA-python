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

<b> Some new literature </b>

- https://web.stanford.edu/group/pdplab/pdphandbook/handbookch10.html
- https://github.com/dennybritz/reinforcement-learning
- https://github.com/keras-rl/keras-rl
- http://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
- https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69

But so far this has been most useful considering manually updating network weights:
- https://medium.com/@dhruvp/how-to-write-a-neural-network-to-play-pong-from-scratch-956b57d4f6e0
- https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69

Questions:
- When exactly does training occur? (After arm gets to terminal state?)
- Are the delta and a_t1 - A_t0 basically just loss functions?

<b> Training works again, but still has some issues </b>

So, training is working again, but I still have few issues and improvements:
- Make exploration smaller as the arm gets closer to the object. That way, arm will be able to fine-tune faster and won't lose so much 
time with exploration while the object is right next to it.
- Make sure the arm doesn't fold into itself. That happends because the arm finds the local optima that way and can't 
get out of it no matter how many trials it makes.
- Should I perform an explored action or predicted action after actor.fit()?
- Should the arm get closer to the object with every iteration. If I, for instance have a perfect critic, then the 
value it returns is only positive when future state (facilitated by action) is better then the current state. 
Or does critic have a different role? Maybe it is just evaluating if random explored action is better than default
action actor would execute. But in that case, how does it acomplish that? Giving it another thought, I would
say it's the second one.
- Does the actor perform action every time, and the action solely depends on which one it makes; explored one if 
exploration was better, instead the default one?

- Penalizing worse movements
- implementing -1 rewards
- relative movements
- tanh activation function
- Plotting of positions 