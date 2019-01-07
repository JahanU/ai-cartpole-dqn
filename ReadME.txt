To run the program you first require a range of dependencies:
	
	Keras
	Tensorflow
	OpenAI Gym (From openAI Gym you will require all the dependencies, to do this, inside the directory, do the following command: pip install gym[all]
	Python 3.5 virtual enviroment
	pip - will help in installing all the required dependencies
-
-
-

How I installed them:
	Install pip:
		sudo easy_install pip

Create New directory 
	Git clone cd gym:
		git clone https://github.com/openai/gym.git
		cd gym
		pip install gym[all]

May need to install virtualenv:
	sudo pip install virtualenv

create virtual enviroment within gym
	virtualnv venv 

create python 3.5 enviroment within the already made venv
	virtualenv -p python3.5 venv
	python --version - "Checks python version"

install tensorflow:
	pip install tensorflow

install keras:
	pip install keras
-
-
-

Here is how I ran the program:
	Jahan-MacBook-Pro:~ Jahan$ cd A2-CartPoleDQN
	Jahan-MacBook-Pro:A2-DQN Jahan$ cd gym
	Jahan-MacBook-Pro:gym Jahan$ source venv/bin/activate
	(venv) Jahan-MacBook-Pro:gym Jahan$ python CartPoleDQN.py


Running python code on command line (Windows):
	py CartPoleDQN.py

(Mac):
	python CartPoleDQN.py