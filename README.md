# TeleICU Monitoring System 🖥️🎦

- TeleICU empowers intensivists, the backbone of ICUs, to manage more patients.  This remote monitoring system leverages AI:
  - YOLOv10: This powerful deep learning model identifies patients and tracks vitals in real-time video feeds.
  - Deep SORT: It builds on YOLOv10's work, tracking patient movements to create a detailed health picture.
  
- With AI handling the heavy lifting, a single intensivist can:
  - Monitor multiple patients simultaneously, freeing up time for deeper care.
  - Intervene faster thanks to real-time monitoring, potentially improving patient outcomes.
  
TeleICU with AI is revolutionizing critical care by boosting efficiency and effectiveness.

## Demo of The Project
https://github.com/Wydoinn/TeleICU-Monitoring-System/assets/120785316/82fc5ca4-63f4-4489-8aab-ebe7e5697443

# Installation

**1. Clone This Repository**

```bash
git clone https://github.com/Wydoinn/TeleICU-Monitoring-System.git
cd TeleICU-Monitoring-System
```

**2. Create New Environment**

- Using Conda

```bash
conda env create -f conda.yml
conda activate teleicu-monitoring-system
pip install -r requirements.txt
```

- Using Pip

```bash
python -m virtualenv -p python3.11.7 teleicu-monitoring-system
source teleicu-monitoring-system/bin/activate
pip install -r requirements.txt
```

**3. Clone YOLOv10 Repository**

```bash
git clone https://github.com/THU-MIG/yolov10.git
pip install .
```

**4. TeleICU Monitoring System Application**

- For Web Application

```bash
python app.py
http://127.0.0.1:5000 # Navigate to the address on your browser
```

- For Window Application

```bash
python monitor.py
```
