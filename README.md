# 24-Hour Continuous Holter Cardiac Telemetry Dataflow

An institutional-grade, multi-format clinical data engineering pipeline that ingests raw, high-fidelity microvolt Electrocardiogram (ECG) signals from the PhysioNet database, structures telemetry streams into an optimized relational SQLite data warehouse, processes biological waveforms via topological peak detection, and auto-generates diagnostic analytics across corporate Excel sheets and multi-page patient dashboards (PDF).

---

## 📊 Visual Diagnostic Dashboard

The pipeline automatically compiles raw time-series metrics into high-speed scannable diagnostic visuals, completely removing text clipping or overlapping errors:

### 1. Global Cardiac Rhythm & Heart Rate Spectrum

| 24-Hour Timeline Spectrum Variation | Overall Heart Rhythm Composition |
| :---: | :---: |
| ![Heart Rate Spectrum Variation](assets/patient_heart_rate_bar.png) | ![Heart Rhythm Composition](assets/patient_health_pie.png) |

### 2. Standardized ECG Strip Paper Simulation
The rendering engine maps authentic, light-pink clinical multi-grid layouts (`#FFF5F5`) with fine major and minor axes, precise millivolt bounds, and calibrated 3-second diagnostic windows matching hospital thermal printout sheets:
<img width="1100" height="900" alt="image" src="https://github.com/user-attachments/assets/ec8a4e7d-9908-4538-9a81-e45ce44eceb3" />
<img width="1400" height="760" alt="image" src="https://github.com/user-attachments/assets/84d5dfbd-3ed9-4763-9767-7abee9a851be" />


---

## 🚀 Architectural Dataflow Overview

The analytical pipeline is structured across four decoupled data engineering layers:

# holter-cardiac-telemetry-dataflow

[STEP 1] Scan/Load Local Cache (MIT-BIH Record 100 Analytics Core)
│
[STEP 2] Bulk SQL Ingestion with Optimized Indication Keys (B-Tree Indexing)
│
[STEP 3 & 5] Resampling Telemetry Streams & Running Inline OpenAI Analytics
├── Deploy SciPy Peak Finder (Filter Noise/Calculate Heart Rates)
├── Inject Calibrated 3s ECG Thermal Strip Plots into Excel Tabs (Cell E3)
└── Active OpenAI LLM Interface (Unique Diagnostic Text Per Block Context)
│
[STEP 4] Compile Standalone Executive Charts (Spectrum Bar + Composition Pie)
│
[STEP 6] Compile Multi-Page ReportLab PDF Assembly Canvas -> [SUCCESS]
