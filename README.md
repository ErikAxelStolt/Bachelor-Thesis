# Finite-Time Lyapunov Exponents in Recurrent Neural Networks

This repository contains the code and files for my bachelor's thesis:

## Finite-Time Lyapunov Exponents in Recurrent Neural Networks: Analyzing Chaos in RNN Forecasts of Dynamical Systems
Erik Stolt & Abbe Tanndal

Uppsala University, Engineering Physics (2025)

Supervisor: Magdalena Larfors

Reviewer: Sayantani Bhattacharya

Examiner: Martin Sjödin


# Abstract
something


This project investigates how chaos, a hallmark of nonlinear dynamical systems, manifests in machine learning models trained to predict such systems. Specifically, we analyze Recurrent Neural Networks (RNNs) trained to forecast the motion of a simple pendulum (regular) and a double pendulum (chaotic).

To study this, we define and compute two types of Finite-Time Lyapunov Exponents (FTLEs) for the RNN:

Prediction-based FTLE (PLE): Measures how small perturbations in the input sequence grow in the predicted output.

Hidden-state FTLE (HSLE): Tracks sensitivity to initial conditions within the RNN's internal dynamics using Jacobians.

We found that:

* The FTLEs of the trained RNNs converge and reflect the underlying physical system's behavior.

* There is a positive correlation between the FTLEs of the RNN and the true Lyapunov exponent of the system it models.

These results suggest that RNNs not only approximate trajectories but also inherit the chaotic structure of the dynamical systems they are trained on—offering a new perspective on how complexity is encoded in neural models.




