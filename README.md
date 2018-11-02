# Reinforcement-Learning-Stocks
Applying Reinforcement learning models for stock price predictions. We are using very basic Deep Q-Learning algorithms to train our model.

## Data Used
All of the data I am using for training is publicly available on Yahoo! Finance. I have already uploaded some of the files in   `data_raw` folder. We will be using `Closing Prices` for our evaluation. As more complex algorithms and models are build, I will resources to those too.

## Packages used
Tensorflow is the major library that we are using to make and run our deep learning models. All of the code uses tensorflow for operation while using numpy and pandas as secondary libraries.

## Files
Download the .zip for this repo, and run the command `python3 skeleton_stock_dqn.py`. This is the most basic file for beginners to understand what the code does. There is also a blog [here]() that explains the code per line. There are more files in the repo, and can be run with same commands. 

## Models
For now I have trained three simple models, DQN, Double DQN, Dueling DQN and also uploaded the skeleton code for those. You can go through it for improvements.

## Vision
To make DQN models capable of handling three various aspects of stock markets, to buy or not to buy (already achieved), how much to buy (more tricky, still doable) and what to buy (AI requires a strong knowledge background here). Working on how much to buy.

The bucket method looks promising for now.
