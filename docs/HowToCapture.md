# How to Capture: Motion Sequence Recording Guidelines

## Phase 1: Preparation

### Material 
1. BLK2GO
2. Aria Glasses
3. Motion Caputre Suit
4. Extension Cable
5. Power Supplies
6. Small Power Bank
7. Measurement Device
8. Optional: Router

### Environmental and Subject Readiness
* **Measurements:** Record the subject's exact height, weight, and arm span prior to suiting up.
* **Static Scene:** Ensure the target environment is entirely static. Do not proceed if moving objects or pedestrians are expected to cross the capture area.
* **Magnetic Interference:** Verify the location is free from heavy magnetic interference (e.g., large metallic structures, heavy machinery), which will corrupt the IMU tracking.

### Leica BLK2GO Scanner
1. Fully charge the scanner's batteries.
2. Download the **BLK2GO Live** application onto your mobile device.
3. Open the application settings and change the initialization mode to **Stationary**.

### Project Aria Glasses (Subject & Observer)
1. Fully charge both pairs of glasses.
2. Download the **Aria Studio** application onto your mobile device.
3. Set the recording profile to **Profile 15** for both pairs of glasses.

### Egocentric Motion Capture Suit
1. Charge a compatible [power bank](https://support.rokoko.com/hc/en-us/articles/4561524515345-Best-power-banks-for-Smartsuit-Pro-II) to 100%.
2. Secure a dedicated [Wi-Fi router](https://support.rokoko.com/hc/en-us/articles/4410409171985-Best-Wi-Fi-routers-for-Smartsuit-Pro-II) if your mobile device cannot provide a reliable hotspot connection.
3. Download the **Rokoko Studio** application on your main processing machine.
4. Generate digital avatars of the subjects according to the manual measurements. 

---

## Phase 2: Recording Procedure

### Step 1: 3D Scene Capture
1. Place the scanner on a stable surface facing a feature-rich direction. The scenery must not change during the recording phase.
2. Connect the scanner to your mobile device.
3. Start the scan in **Stationary Mode** and capture the initial reference picture.
4. Pick up the scanner, traverse the environment to capture the geometry, and then stop the scan.

### Step 2: Device Synchronization Setup
1. Instruct the Subject to stand completely still, facing the exact same direction as the scanner's initial reference picture.
2. Position the Observer so they are facing the Subject, ensuring the Subject's full body is clearly visible.
3. Power on the Rokoko suit, connect it to your network, and perform the straight-pose calibration via Rokoko Studio.
4. Verify the calibration by having the Subject perform a few full-body movements. Repeat the calibration if the digital avatar behaves unnaturally.

### Step 3: The Action Sequence
1. Start the recording on the **Observer's** Aria Glasses.
2. Start the recording on the **Subject's** Aria Glasses. Confirm the recording LED is illuminated and ensure absolute silence in the room.
3. Start the recording in **Rokoko Studio**.
4. Instruct the Subject to perform a single, hard, visible **clap** (essential for audio-visual synchronization).
5. Instruct the Subject to take one distinct **sidestep to the left**, followed by one **sidestep to the right**.
6. Execute the planned motion sequence. **Keep the total recording time strictly under 5 minutes** to prevent visible IMU sensor drift.
7. Once the sequence is complete, stop the recordings on all devices.

---

## Phase 3: Post-Processing

### Point Cloud & Mesh Generation (Scanner)
1. Download the scanned scene (`.b2g`) from the BLK2GO.
2. Import the file into the off-the-shelf **Leica Cyclone REGISTER 360 PLUS** software to generate the point cloud (`.e57` format) and align the data if using multiple scans.
3. Import the generated point cloud into **Leica Cyclone 3DR** to clean up and generate the final mesh file (e.g., `.obj`).

### Aria Glasses SLAM Processing
1. Download the recording files (`.vrs`) from both glasses.
2. Process the SLAM data on the Meta servers using the [MPS CLI](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/mps_summary) or the Aria Studio application.
3. **Note:** When processing the data from both glasses simultaneously for absolute spatial alignment, you *must* use the **Multi-SLAM CLI** (this feature is not possible via the mobile app).

### Suit Motion Extraction
1. Open the recorded take in the **Rokoko Studio** software and manually process the foot-locks to mitigate floor sliding.
2. Export the HumanIK skeleton data as a `.bvh` file at **30 FPS**.
3. Export the raw joint data as a `.csv` file at **100 FPS**.