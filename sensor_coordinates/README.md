# Sensor Coordinate Calculation

Phillip has provided the JPEG image which details roughly where each of the sensor locations are. I have created a python script that hard codes the approximate locations of the sensors, and takes in the coordinates of the field. From this we can get approximate coordinates of each sensor, which is saved in sensor_coordinates.json

He's also provided this table in notion mapping sensor ids to location:

|  | C1  | C2 | C3 |
| --- | --- | --- | --- |
| R4 | 2, 26, 29, 37, 39, 60, 79, 96 |  |  |
| R3 | 18, 38, 49, 59, 71, 76, 84, 85) | 9, 24, 31, 33, 52, 92, 99, 100 | 5, 20, 34, 42, 43, 47, 50, 72 |
| R2 | 10, 13, 45, 55, 56, 64, 87, 97 | 3, 14, 25, 30, 36, 53, 86, 90 | 19, 23, 32, 54, 57, 80, 81, 94 |
| R1 | 15, 21, 28, 46, 63, 69, 75, 93 | 11, 12, 35, 61, 70, 88, 89, 91 | 4, 22, 40, 44, 48, 67, 78, 98 |
|  |  |  |  |
| Control 1 (NW) | 1, 17, 58, 65, 66, 74, 82, 83 |  |  |
| Control 2 (NE) | 6, 7, 8, 16, 27, 62, 77, 95 |  |  |