# EchoRide

EchoRide is a simulation based audio hazard detection framework for cyclist safety. The project explores whether directional vehicle movement can be approximated using environmental audio alone using a Unity generated synthetic traffic environment and a Python machine learning pipeline.

Cyclists often need to monitor nearby vehicles while also paying attention to road conditions, traffic signals, pedestrians, and balance. Existing cyclist awareness systems frequently rely on radar, cameras, or other dedicated hardware that may increase cost or require specialized equipment.

EchoRide explores a lower cost, smartphone inspired alternative using directional audio classification. The system generates synthetic traffic audio scenarios in Unity, processes WAV recordings through a Python machine learning pipeline, and predicts directional vehicle movement using extracted audio features.

# How It Works

EchoRide uses a multi-stage simulation and machine learning pipeline:

1. Unity generates labeled directional traffic scenarios
2. WAV recordings are generated automatically
3. Audio recordings are preprocessed and standardized
4. Audio features are extracted
5. A machine learning classifier predicts vehicle direction
6. A simulated cyclist vibration alert is generated

Current direction classes include:
- FrontPass
- RearPass
- LeftPass
- RightPass
- LeftTurn
- RightTurn
- RearCrash

# Project Structure

EchoRide/
├── src/
│   ├── guiApp.py
│   ├── predictor.py
│   ├── preprocessing.py
│   ├── feature_extract.py
│   ├── direction_model.py
│   ├── vibration.py
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── tests/
├── unity/
│   └── BikeAudioSim/
├── requirements.txt
└── README.md

# Unity Simulation

The Unity project is located in:

unity/BikeAudioSim/


The simulation generates synthetic cyclist traffic audio scenarios using:
- spatial audio
- directional vehicle movement
- randomized ambient sounds
- automatic WAV recording
- automatic file naming

Ambient conditions currently include:
- heavy wind
- idle engines
- dog barking
- children playing
- horns
- sirens
- construction sounds
- night ambience

Generated recordings are saved as WAV files and later processed through the Python classification pipeline.

# Direction Classes

| Label | Description |
|---|---|
| FrontPass | Vehicle passes in front of cyclist |
| RearPass | Vehicle approaches from behind |
| LeftPass | Vehicle passes on left side |
| RightPass | Vehicle passes on right side |
| LeftTurn | Left turning vehicle scenario |
| RightTurn | Right turning vehicle scenario |
| RearCrash | Rear collision inspired scenario |


# Installation

## Prerequisites

- Python 3.9+
- Unity 2022+ (only required for simulation generation)

## Clone Repository

```bash
git clone https://github.com/Chance-Krueger/EchoRide.git
cd EchoRide
```

## Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

# Usage

## Run the Main Pipeline

```bash
cd src
python main.py
```

This:
- builds the dataset
- preprocesses audio
- extracts features
- trains the classifier
- evaluates predictions
- generates simulated vibration outputs

---

## Run the GUI

```bash
cd src
python guiApp.py
```

GUI workflow:
1. Upload a WAV file or select a preset
2. Press the power button
3. The classifier predicts vehicle direction
4. Simulated vibration patterns are displayed

Note: the power button must be turned off after each prediction.

Vibration Paterns:
    RightTurn: 1 blink
    LeftTurn: 2 blinks
    FrontPass: 3 blinks
    RearPass: 4 blinks
    LeftPass: 5 blinks
    RightPass: 6 blinks
    RearCrash: rapid/continuous

## Generate Audio Using Unity

1. Open Unity Hub
2. Open:
   
   audio simulator/

3. Enter Play Mode
4. Trigger scenario generation
5. WAV recordings are automatically saved

Generated WAV files should be placed into:

```text
data/raw/<DirectionClass>/


Example:

```text
data/raw/RearPass/
```

---

# Data Format

Audio recordings:
- WAV format
- organized by direction class folders
- resampled to 16kHz during preprocessing
- padded or cropped to fixed duration

# Machine Learning Pipeline

Current pipeline stages:
- audio preprocessing
- feature extraction
- dataset construction
- direction classification
- vibration mapping

The project currently supports experimentation with:
- synthetic audio datasets
- direction classification
- cyclist alert mapping

---

# Current GUI Features

The GUI currently supports:
- WAV file upload
- preset audio recording
- direction prediction display
- vibration visualization



# Known Limitations

- Current dataset is synthetic only
- Real world cyclist recordings have not yet been integrated
- Unity and Python are not directly connected in real time
- Current vibration output is visual simulation only


# Future Work

Planned future improvements include:
- real time Unity → Python integration
- live microphone input
- smartphone application deployment
- haptic feedback hardware
- larger datasets
- real world recordings
- real time processing
- improved prediction accuracy 

# Authors

- Kianny Calvo
- Chance Krueger


# Acknowledgments

CSC 396 — University of Arizona — Spring 2026


# License

To be determined.