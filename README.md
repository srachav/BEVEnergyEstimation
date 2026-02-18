# Personalized BEV Energy Consumption Estimation Framework

This repository contains the implementation of a personalized Battery Electric Vehicle (BEV) energy‑consumption estimation framework that integrates **driver‑specific behavior modeling** with **map‑based route data** to predict future vehicle velocity and energy usage.

---

## Overview

This project combines machine learning and map‑based contextual information to estimate how a specific driver will navigate a selected route and how much energy their BEV will consume. The system leverages:

- **Driver‑specific velocity prediction** using an LSTM encoder–decoder model  
- **State‑space vehicle kinematic simulation**  
- **Map data from OpenStreetMap (OSM)**  
- **Node and edge attributes from the Valhalla routing engine**  
- **A physics‑based BEV energy consumption model**

---

## Key Components

### 1. Driver‑Specific Velocity Prediction
The framework predicts future vehicle velocity by integrating:

- A **Bidirectional LSTM encoder–decoder architecture**, based on the model proposed in:  
  **DriVe‑forecast** — https://github.com/lpaparusso/DriVe-forecast  
- A **state‑space control system** to generate a baseline velocity and acceleration profile  
- Map‑derived contextual features such as curvature, slope, and speed limits  

This enables modeling of how an individual driver behaves across different road environments using road geometr. Traffic is not modeled. 

---

### 2. Map‑Based Route Processing
Routes are selected using **OpenStreetMap (OSM)** and enriched with:

- Road geometry  
- Elevation  
- Speed limits  
- Curvature  
- Traffic‑control indicators  

The **Valhalla routing engine** is used to extract these features and generate a high‑resolution representation of the route.

---

### 3. BEV Energy Consumption Model
Energy consumption is computed using a physics‑based model derived from:

**Fiori, Ahn, and Rakha — “Power-based electric vehicle energy consumption model: Model development and validation.”**

The model estimates:

- Tractive power  
- Regenerative braking  
- Battery energy usage  
- State‑of‑Charge (SOC) evolution  

---

## Usage

- The **src** folder contains the full inference pipeline for:
  - 
  - Route processing  
  - Vehicle simulation  
  - LSTM‑based velocity prediction  
  - BEV energy consumption estimation  

- The **training** folder contains the scripts used to train the LSTM model on driver‑specific datasets.

---

## Additional Documentation

For a complete technical description of the methodology, system architecture, experiments, and results, refer to the thesis document:

**GRAD699_90_O_Applied_Project_SreechakraRachavelpula**

