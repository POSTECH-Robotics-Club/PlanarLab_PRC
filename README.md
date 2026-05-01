# PlanarLab

**PlanarLab** is a lightweight 2D simulation lab designed for rapid prototyping and validation of planning and control algorithms.

It provides a simple yet flexible environment for experimenting with both model-based and learning-based methods, including trajectory optimization, sampling-based planners, and reinforcement learning approaches such as MPPI and TD-MPC.

## Key Features

* **Lightweight & Fast**
  Built for quick iteration and idea validation without heavy simulation overhead.

* **2D Planar Environments**
  Minimal yet expressive environments focused on planar dynamics (e.g., car-like systems).

* **Planning-Centric Design**
  Designed to support trajectory optimization and predictive control workflows.

* **Modular Architecture**
  Clean separation between environments, dynamics, and controllers.

* **Hybrid Methods Ready**
  Easily combine model-based planning (e.g., MPPI) with learning-based approaches (e.g., TD-MPC).

## Philosophy

PlanarLab is not intended to be a high-fidelity simulator.
Instead, it acts as a **sandbox for fast experimentation**, where ideas can be tested, iterated, and validated before scaling to more complex systems.

## Use Cases

* Rapid prototyping of planning algorithms
* Benchmarking model-based vs. learning-based control
* Testing hybrid control pipelines
* Educational and research experiments in control and RL

## Getting Started

Coming soon.

### 1. Clone the repository

```bash
git clone https://github.com/TaehwanWon/PlanarLab.git
cd planarlab
```

### 2. Create a virtual environment (recommended)
```bash
conda create -n planarlab python=3.12 -y
```

### 3. Activate the environment
```bash
conda activate planarlab
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run a example code
Launch Jupyter:

```bash
jupyter notebook
```
Then open:
```bash
scripts/mppi/runner.ipynb
```


## Attribution

This project includes code derived from kohonda (2023), originally licensed under the MIT License.

## Contact

**Author:** Taehwan Won  
**Affiliation:** Mechanical Engineering, POSTECH  
**Email:** wonth1375@postech.ac.kr 

