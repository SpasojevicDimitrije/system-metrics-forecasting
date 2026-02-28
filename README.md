# System Metrics Forecasting

## Overview

This project investigates next-state prediction of computer system metrics using time series modeling techniques.

The dataset consists of virtual machine performance traces from the Bitbrains Cloud VM dataset. Each VM contains multivariate time series data including:

- CPU usage (%)
- Memory usage
- Disk read throughput
- Disk write throughput
- Network received throughput
- Network transmitted throughput

The objective is to predict the next system state:


\[
\hat{x}_{t+1} = f(x_t, x_{t-1}, ..., x_{t-k})
\]

where \( x_t \in \mathbb{R}^d \) represents the system metrics at time \( t \).

---

## Project Goals

The project compares:

1. Linear statistical models (VAR)
2. Linear regression with lag features
3. Recurrent neural networks (LSTM / GRU)
4. Latent state forecasting using Autoencoders

We evaluate:

- One-step prediction accuracy
- Multi-step rollout stability
- Latent vs direct forecasting performance

---

## Dataset

Bitbrains VM performance traces.

Each CSV file contains approximately 10,000 time steps sampled at regular intervals.

---

## Repository Structure

data/
    raw/
    processed/

notebooks/

src/
    data/
    models/
evaluation/
reports/

---

## Setup

Create virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
