Football Match Prediction System with Deep Learning & Live Statistics

A comprehensive football match prediction system that combines machine learning (LightGBM), deep learning (TCN - Temporal Convolutional Network), Poisson modeling with Dixon-Coles adjustment, dynamic Elo ratings, and live match statistics integration.
Table of Contents

    Overview

    System Architecture

    Features

    Installation

    Data Sources

    Core Components

    Usage Guide

    Output Visualizations

    API Integration

    Configuration

    Performance Metrics

Overview

This system predicts football match outcomes (Home Win, Draw, Away Win) using:

    Historical EPL data from Understat API (xG, goals, match dates)

    Dynamic Elo ratings with margin-of-victory weighting

    Temporal Convolutional Network (TCN) for sequence learning

    LightGBM with Bayesian hyperparameter optimization

    Platt scaling & Isotonic regression for probability calibration

    Monte Carlo simulations for score distribution

    Live match statistics via API-Football (v3)


Features
Data Processing

    Fetch EPL match data from 2022-2025 seasons

    Dynamic Elo rating system (base K=20, margin-of-victory multiplier)

    Rolling window features (5 matches, exponentially weighted)

    Head-to-head statistics

    Fixture congestion analysis

    Home advantage calculation

Machine Learning

    LightGBM: 30+ engineered features including:

        Weighted average goals, xG, xGA

        Clean sheet rates

        Points accumulation rates

        ELO differentials

        Poisson-derived probabilities

    TCN Model:

        Dilated causal convolutions (dilations 1, 2, 4)

        Residual connections

        Weight normalization

        Receptive field: 10 matches

        Parameters: ~30K (vs 70K for Bi-LSTM)

Calibration Methods

    Platt scaling (sigmoid)

    Isotonic regression

    Time-series aware train/calibration split

Ensemble Strategy

    Dynamic gating based on TCN performance

    Optimal weight search via validation set

    KL divergence-based blending with Poisson priors

Monte Carlo Simulation

    Dixon-Coles adjustment (rho=0.15)

    10,000+ simulations per match

    Score distribution analysis

    Convergence monitoring

Live Statistics (API-Football)

    Real-time possession, shots, passes

    Live xG approximation

    On-the-fly probability updates

Installation
Prerequisites
bash

Python 3.8+
pip install --upgrade pip

Dependencies
bash

pip install numpy pandas matplotlib seaborn scipy
pip install torch torchvision
pip install lightgbm scikit-learn
pip install understatapi
pip install optuna  # optional, for Bayesian tuning
pip install requests  # for live API

Quick Install
bash

git clone <repository-url>
cd match-intel-v4
pip install -r requirements.txt

Data Sources
Source	Data Type	Access Method
Understat	xG, goals, match dates, teams	understatapi Python client
API-Football	Live statistics, fixtures	REST API (requires key)
API Keys

    Understat: No authentication required

    API-Football: Free tier available at https://www.api-football.com/

Core Components
1. Dynamic Elo System
python

compute_elo(df, k_base=20, mov_cap=4)

    Base K-factor: 20

    Dynamic adjustment: k = k_base * log1p(min(goal_diff, mov_cap))

    Expected score: E = 1 / (1 + 10^((Ra - Rh)/400))

2. Feature Engineering (build_features)

Generates 30+ features:

    Rolling averages (exponential weighting)

    Efficiency metrics (actual vs expected goals)

    Defensive metrics

    Form indicators

3. Temporal Convolutional Network

    Channel-first architecture (Batch, Channels, Sequence)

    Dilated residual blocks

    No future information leakage (causal padding)

4. Poisson Model with Dixon-Coles

Adjusts for low-scoring correlation:
python

tau(i,j) = 1 - hx * ax * rho   # for (0,0)
tau(i,j) = 1 + ax * rho        # for (1,0)
tau(i,j) = 1 + hx * rho        # for (0,1)
tau(i,j) = 1 - rho             # for (1,1)

5. Ensemble Blending

    If TCN is 5%+ worse than LGBM: gate to LGBM-only

    Otherwise: weighted average optimized on validation set

    Final blend: p_final = α * p_ml + (1-α) * p_pois where α = max(0.5, 1 - KL/0.05)

Usage Guide
Basic Prediction
python

# Configure teams
HOME_TEAM = "Burnley"
AWAY_TEAM = "Manchester City"

# Run prediction
prob, h_xg, a_xg, p_ml_raw, p_pois, kl_div, alpha_ml = predict_match(HOME_TEAM, AWAY_TEAM)

# View results
for outcome, probability in prob.items():
    print(f"{outcome}: {probability*100:.1f}%")

Monte Carlo Simulation
python

scores, counts, hw, dr, aw, mat = run_monte_carlo(h_xg, a_xg, n_sims=10000)
print(f"Home Win Probability: {hw/10000*100:.1f}%")

Live Match Statistics
python

# Fetch live data
fix_id, actual_home, actual_away, status = get_fixture_id(HOME_TEAM, AWAY_TEAM)
get_full_statistics(fix_id, actual_home, actual_away, status)

# Calculate live xG
h_xg, a_xg = calculate_live_xg(live_stats)
live_probs, score_mat = run_live_monte_carlo(h_xg, a_xg)

Generate Full Dashboard
python

# Automatically creates comprehensive visualization
# Saved as: Match_Intel_v4_{home}_vs_{away}_{timestamp}.png

Output Visualizations

The system generates a comprehensive dashboard with 13 panels:
Panel	Content
Probability Donut	Final blended probabilities
Score Heatmap	Most likely scorelines
xG Distribution	Poisson goal distributions
Alignment Chart	ML vs Poisson comparison
Reliability Diagram	Calibration quality
Monte Carlo Bar	Simulation outcomes
Bubble Chart	Score frequency
Convergence Plot	Simulation stability
Top 10 Scores	Most probable results
Team Form	Last 10 matches goals
Feature Importance	LGBM top predictors
Elo Evolution	Rating history
TCN Summary	Deep learning performance
Recent Tables	Last 10 matches per team
Live Dashboard (6 panels)

    Possession donut

    Shot analysis

    Passing accuracy

    Discipline & set-pieces

    Live W/D/L probability

    Live score prediction heatmap

API Integration
API-Football Endpoints Used
python

# Fixtures (live)
GET https://v3.football.api-sports.io/fixtures
params: {"live": "all"}

# Fixtures (historical)
GET https://v3.football.api-sports.io/fixtures
params: {"team": team_id, "last": 20}

# Statistics
GET https://v3.football.api-sports.io/fixtures/statistics
params: {"fixture": fixture_id}

# Team search
GET https://v3.football.api-sports.io/teams
params: {"search": team_name}

Headers
python

HEADERS = {"x-apisports-key": YOUR_API_KEY}

Configuration
Adjustable Parameters
python

# Data
SEASONS   = ["2022", "2023", "2024", "2025"]
ROLL_N    = 5           # Rolling window size
SEQ_LEN   = 10          # Sequence length for TCN
SEQ_FEATS = 5           # Features per timestep

# Elo
K_BASE = 20
MOV_CAP = 4

# TCN
N_FILTERS = 32
DROPOUT = 0.2
DILATIONS = [1, 2, 4]

# Ensemble
TCN_GATE_THRESHOLD = 0.05  # 5% relative improvement required

# Monte Carlo
N_SIMS = 10000
MAX_GOALS = 8
RHO = 0.15  # Dixon-Coles correlation

Performance Metrics
LogLoss Comparison (EPL validation)
Model	LogLoss
LightGBM (uncalibrated)	~0.98
LightGBM (calibrated)	~0.94
TCN	~0.96
Ensemble	~0.92
Calibration Methods

    Isotonic regression: Better for non-linear miscalibration

    Platt scaling: More stable with limited calibration data

    System automatically selects best based on validation LogLoss

Feature Importance Top Predictors

    Poisson home win probability

    Expected goals differential

    ELO differential

    Points rate differential

    Clean sheet rates

File Outputs
File	Description
Match_Intel_v4_{home}_vs_{away}_{timestamp}.png	Full prediction dashboard
{home}_vs_{away}_{timestamp}.png	Live match dashboard
Limitations & Assumptions

    Data availability: Understat xG data may have delays

    Live API: Requires internet connection and valid API key

    Small sample warning: Rolling features require minimum 3 matches

    Time-series split: Assumes chronological order is preserved

    Poisson independence: Dixon-Coles only adjusts for 0-0, 1-0, 0-1, 1-1

    Home advantage: Calculated from last 20 matches only

Troubleshooting
Common Issues

"Team not found in ELO dictionary"

    Team name must exactly match Understat format (e.g., "Manchester City", not "Man City")

"API-Football returns 429"

    Free tier limited to 100 requests/day

    Add delay between requests: time.sleep(1)

"Insufficient data for rolling features"

    Ensure at least 3 matches in selected seasons

    Reduce ROLL_N parameter

"TCN gated (LGBM only)"

    TCN performance was >5% worse than LGBM

    Normal behavior; LGBM more robust with small data

Performance Optimization
python

# Reduce computation time
ROLL_N = 3
N_SIMS = 5000
SEQ_LEN = 8

# Use CPU for inference
tcn_model.cpu()

License

For educational and research purposes only. Not for commercial betting use.
Acknowledgments

    Understat for xG data

    API-Football for live statistics

    Dixon & Coles (1997) for correlation adjustment

    Bai et al. (2018) for TCN architecture
