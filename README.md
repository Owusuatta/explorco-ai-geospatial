# EXPLORCO AI — Geospatial Exploration Screening Assistant

## Overview

**EXPLORCO AI** is a prototype geospatial decision-support system designed to assist exploration teams in screening and prioritizing areas for potential field investigation.

The system combines:

- GIS and remote sensing data
- Terrain analysis
- Road accessibility analysis
- Machine learning
- Field-image classification
- Interactive visualization

The main objective is to demonstrate how geospatial data and artificial intelligence can support faster and more structured exploration planning.

> **Important:** This prototype is a field-investigation screening system. It does not predict the presence of oil, gas, or mineral deposits and should not be interpreted as a hydrocarbon prospectivity model.

---

# Project Objectives

The project was developed to demonstrate how multiple geospatial datasets can be integrated into an AI-assisted exploration workflow.

The main objectives are to:

1. Prepare and process satellite and terrain datasets.
2. Derive useful geospatial features for exploration screening.
3. Analyse accessibility using distance to roads.
4. Develop a field-investigation priority score.
5. Train a machine-learning model to reproduce the screening priority classes.
6. Generate a spatial exploration-priority map.
7. Classify field photographs into visible surface-condition categories.
8. Provide an interactive interface for exploring the results.

---

# System Architecture

The system consists of two complementary AI components.

### 1. GIS-Based Exploration Screening

The GIS component uses:

- Sentinel-2 imagery
- Near-infrared reflectance (B08)
- NDVI
- Elevation
- Slope
- Distance to roads

These variables are processed into a geospatial feature dataset.

A screening score is then generated using road accessibility and terrain accessibility.

The resulting areas are divided into:

- **Check First**
- **Check Next**
- **Check Later**

These categories represent relative field-investigation priority within the study area.

---

### 2. Field Image Classification

The second component uses a MobileNetV2-based image classification model to analyse uploaded field photographs.

The model classifies visible surface conditions into:

- Bare Ground
- Disturbed Ground
- Road Access
- Rocky Surface
- Vegetation

The field-image model provides additional visual evidence for field teams.

It does **not** modify the GIS screening result.

Instead, the system treats the two models as complementary:

**GIS AI → Where should we investigate?**

**Field AI → What visible condition is present?**

**Decision Support → What should the field team consider next?**

---

# Data Sources

The prototype uses the following geospatial datasets:

### Sentinel-2

Sentinel-2 imagery was used to obtain multispectral information, including the B08 near-infrared band.

NDVI was derived from the satellite imagery to provide vegetation-related information.

### Digital Elevation Model

A 30 m SRTM-derived Digital Elevation Model was used to generate elevation and slope information.

### OpenStreetMap

Road data from OpenStreetMap was used to estimate accessibility.

Road features were converted into a raster representation and used to calculate distance to the nearest road.

### Geology

Geological data were inspected during project development. However, the available geology layer did not spatially overlap the final Sentinel-2 study area and therefore was **not included in the final machine-learning feature stack**.

---

# Geospatial Processing Workflow

The overall processing workflow was:

text
Raw Geospatial Data
        ↓
Data Preparation
        ↓
Coordinate System Alignment
        ↓
Clipping to Study Area
        ↓
Feature Generation
        ↓
Feature Stack
        ↓
Exploration Screening Score
        ↓
Priority Classes
        ↓
Spatial Machine Learning
        ↓
Model Evaluation
        ↓
Full-Raster Prediction
        ↓
Exploration Priority Map
