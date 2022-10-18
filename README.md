# Hand-Gesture-Identification

## Software
- **Language:** Python
- **Environment:** JupyterLab 3.2.2

### Folders and Files
- **Data:** This is the database containing all relevant data recordings. The folders are named after the creation date except the *Study* folder, which holds all data associated with the conducted study. Every folder contains the recorded or processed data along with a file called *description* which explains what the data contain and what they are used for.
- **main.py:** This file implements the entire hand gesture processing pipeline. The cells need to be executed in the right order (from top to bottom) to perform hand gesture recognition. Almost every cell has a part called *CONFIGURATION* where variables need to be adjusted according to the users needs. After adjusting the *CONFIGURATION* every cell works "out of the box".

## Hardware
This PCB is used as a sensing board mounted on a velcro strap to measure deformations of the wrist.

### Design Decisions
- The design was chosen as simple and small as possible to enable a fast prototyping process and the usage of up to 10 Sensor-PCBs around the wrist.
- Due to poor availability of the LIS2DH 3-axes accelerometer, we provided two footprints for accelerometers (MC3479, LIS2DH).

### Future Improvements
- Since we had some problems with the communication, different PCB-Headers should be chosen.
- Delete one of the two accelerometer footprints.
- Remove the footprints of the test points. They were added for mounting the Sensor-PCB on the velcro strap but not used.
- Increase the distance between the mounting holes and the PCB-Header to ease the mounting of the board on the velcro strap.
- Place the accelerometer further away from the pressure sensor to ease the process of casting the pressure sensor into silicone.
