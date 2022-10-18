# Hand-Gesture-Identification
This project presents an innovative approach to how static hand gestures can be classified with wrist-worn barometric pressure sensors.

## Hardware
- **Program:** Altium Designer
- The Sensor-PCBs is used as a sensing board mounted on a velcro strap to measure deformations of the wrist

### Design Decisions
- The design was chosen as simple and small as possible to enable a fast prototyping process and the usage of up to 10 Sensor-PCBs around the wrist.
- Due to poor availability of the LIS2DH 3-axes accelerometer, we provided two footprints for accelerometers (MC3479, LIS2DH).

## Software
- **Language:** Python
- **Environment:** JupyterLab 3.2.2

### Folders and Files
- **Data:** This is the database containing some relevant data recordings. The folder contains the recorded data along with a file called *description* which explains what the data contain and what they are used for.
- **main.py:** This file implements the entire hand gesture processing pipeline. The cells need to be executed in the right order (from top to bottom) to perform hand gesture recognition. Almost every cell has a part called *CONFIGURATION* where variables need to be adjusted according to the users needs. After adjusting the *CONFIGURATION* every cell works "out of the box".
