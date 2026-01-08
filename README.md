# Sound Attack

The repository for Offensive AI course final assignment.

## Setup

1. Download the Room Impulse Response and Noise Database from [here](https://openslr.trmal.net/resources/28/rirs_noises.zip)
2. Extract the zip to `data/RIRS_NOISES`
3. Install packages for `uv sync`

## Building the dataset

### Running the dataset creation

1. Configure the parameters in [src/dataset.py](src/dataset.py):
   - `target_snr_db`: Signal-to-Noise Ratio in dB (default: 30). Higher values = quieter noise
   - `random_inject`: Boolean flag for signal placement (default: False)
     - `True`: Inject signal at a random position in the noise
     - `False`: Inject signal at a fixed position (determined by `min_start_time`)
   - `min_start_time`: Time in seconds from the start before the signal appears (default: 0.4)
     - This parameter applies regardless of the `random_inject` setting
     - When `random_inject=True`, this is the minimum time before random placement
     - When `random_inject=False`, the signal starts exactly at this time
   - `time_before_after`: Time in seconds to keep before and after the interesting sound (default: 2.0)
     - The interesting sound is the convolved result (windows sound + RIR)
     - Example: If windows sound is 1s, RIR is 2s (interesting sound ~3s), and `time_before_after=2`, the output will be ~7s (2 + 3 + 2)
     - This parameter cuts the final output to focus on the region around the target sound

2. Run the dataset creation script:
   ```bash
   python src/dataset.py
   ```

The script will generate audio files in the `output/` directory. Each output file combines:
- A source audio signal convolved with a room impulse response (RIR)
- Background noise at the specified SNR level
- The signal placed according to the parameters

### Output naming convention

Output files are named: `{source}_{room}_{background}.wav`

For example: `Windows_sound_in_room_rir_00001_with_cafe_noise.wav`

