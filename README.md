# CACLA, OpenAI environment & forward kinematics

## Requirements

- Python 3.6
- keras 2.2.4
- openai-gym 0.10.9
- skinematics 0.8.0

## Description

Code under /CACLA code implements CACLA algorithm (Hasselt, 2012). Run main.py to start training/testing. When initializing environment with argument *simulation=True*, V-rep instance is also started, opening one of the scenes in folder /Scenes. Otherwise, model is trained with forward kinematics only, which speeds up the training process significantly.

Whole framework uses OpenAI environment, so it should be fairly easy to extend it to other environments/problems, without having to change core CACLA learning process. 

## References

Van Hasselt, H. (2012). Reinforcement learning in continuous state and action spaces. In Reinforcement learning (pp. 207-251). Springer, Berlin, Heidelberg.
