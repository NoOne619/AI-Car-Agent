# **AI Car TORCS Agent using Q-Learning** 🏎️

**An autonomous driving agent built with Q-Learning and trained in the TORCS (The Open Racing Car Simulator) environment.**

---

## **📌 Project Summary**

**This project implements a reinforcement learning agent that learns to drive a car in the TORCS simulator using the Q-Learning algorithm.**  
TORCS (The Open Racing Car Simulator) is an open-source car racing game and research simulator used for AI experimentation. The goal of this project is to enable an AI-controlled car to navigate racing tracks efficiently by learning from trial-and-error interactions.

The Q-Learning agent is trained to make decisions such as **accelerating, braking, and steering** based on the car’s current state (e.g., speed, angle, distance from track center). Over time, it learns an optimal driving policy through rewards and penalties.  

The agent was trained on **three different maps** in TORCS, enabling it to generalize its driving behavior across various track layouts, including curves, straight sections, and sharp turns.

This project demonstrates how classical reinforcement learning can be applied to simulation-based autonomous driving, even without deep learning models.

---

## **🧠 Key Features**

- **Q-Learning-based Driving Agent**
- **Discrete State-Action Space**
- **Custom Reward Function for Track Performance**
- **Training on 3 Distinct TORCS Tracks**
- **Q-Table Saving and Reusability**
- **Modular Python Code (Easy to Extend or Modify)**

---

## **📦 Requirements**

Before running the project, make sure the following software is installed:

### ✅ Python Dependencies

```bash
pip install numpy
