# Cartpole
In this project Gym Cartpole-v0 challenge from OpenAI is solved. The task here is to prevent the pendulum from falling over. "CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials". 
  ![](https://miro.medium.com/max/600/1*Q9gDKBugQeYNxA6ZBei1MQ.png)  
There are two types of implementations in this project. Please note that these implementations are completely independent and there is no relation between them. The implementations are:


*   **Q-learning** implementation
*   **Deep Neural Network** implementation

In the first implementation Q-learning technique is used. There is a saved q-table called *q_table_test.npy* to test the score of implementation. To test the implementation run the following command in the terminal:   
**>>> python src/qlearning.py --test True --qtable q_table_test.npy** 

The second implementation is a **Deep Q-Network (DQN)** and is implemented in the *Cartpole_DQN.ipynb* notebook. There is also a google colab link in the notebook. To run the notebook in google colab click on the link and run the notebook.
