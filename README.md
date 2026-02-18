# Personalized BEV energy consumption estimation framework
-This repository contains code files for a framework that integrates learned driver behavior with map data to esimate future vehicle velocity.
-An LSTM model along with a State-space control system is used to predict a driver-specific velocity profile for a route selected using OSM and populated with node and edge data from Valhalla routing engine
-The LSTM model based on the encoder-decoder architecture suggested in https://github.com/lpaparusso/DriVe-forecast/
-The BEV Energy consumption model is based on the paper “Power-based electric vehicle energy con￾sumption model: Model development and validation” by Chiara Fiori, Kyoungho Ahn, and Hesham A Rakha
-The code for the inference framework to predict velocity profiles for a selected route and calculate the energy consumption is given the \src folder
-The code used for training the LSTM model is provided in \training
