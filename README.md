# Neural Network for Music Exploration (NeuroNMusE)

### Overview

Meet **NeuroNMusE**: a compact and efficient Neural Network for Music Exploration, specifically designed to generate the next 10 seconds of an audio track based on its initial 10 seconds — all while operating with minimal computational resources.

Are you a musician seeking inspiration or struggling to compose your next masterpiece? Let NeuroNMusE be your creative muse. Simply play any segment of your melody—even an unfinished or imperfect one—and let the network suggest alternative rhythms and melodies to spark your creativity.

NeuroNMusE is built on three core components, making it both intuitive and powerful for music generation and exploration.

1. **Encoder**: Encodes the input waveformat files (numpy analog of an audio file, see **Data Preprocessing**) into a compact representation using convolutional and max-pooling layers.
2. **Transformer**: Processes the encoded representation using multi-head attention layers to capture temporal patterns in the audio signal and model temporal dependencies.
3. **Decoder**: Decodes the processed representation back into waveformat files. Thes outputs are converted back into .mp3 files.

The model has two outputs:
- **Output 1**: The original input audio, passed through the encoder and decoder, is reconstructed. This ensures the network's ability to encode and decode the signal effectively without losing important features.
- **Output 2**: The encoded signal is passed through the transformer and then decoded into the predicted continuation of the audio track.

---

### Example Output

Below is an example of the network's 10-second music output generated from the first 10-seconds of soundtrack "Bablo Pobezhdaet Zlo" created by "Yndervud" music band. Although the input is a low telephonic-quality (12000 kHz sample frequency) sound track, the network excels at capturing rhythmic patterns and reproducing drum and bass guitar sections. However, due to computational constraints (e.g., a single GPU with 15GB of RAM), the number of layers and transformer heads is limited, and the training dataset is relatively small (400 songs). As a result, the network struggles with reproducing complex elements like vocals, percussions and rhythm guitar. Here are the examples of neural network input and output .mp3 tracks. 

&nbsp;

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 🎵 **[Download the .mp3 input](
https://github.com/Vlasenko2006/RockNet/blob/main/input_epoch_120.mp3)** 🎵

&nbsp;



&nbsp;

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 🎵 **[Download the .mp3 output](https://github.com/Vlasenko2006/Lets_Rock/blob/main/output_example.mp3)** 🎵

&nbsp;


### Advantages 

1. The neural network is simple and trainable within one hour (120 epochs is pretty enough).
2. The neural network correctly captures the mood and style of the song distinguishing reggae, rock, salsa, etc. rythms in the input without explicit specifications.
3. The neural network creates its own rythmic and instrumental pattern in the output which is similar but not the same as in the corresponding target (original continuation of the input.)
4. Wave format music representation results in memory efficiency.
---

### Caviats

1. Price for efficency in memory usage (Advantage No. 4) is the output quality. 
2. The NN creates a melody in a bass section with drums and bass guitar only (even if the original track has none of these instruments), failing to reproduce complex instruments, guitar riffs, and vocals.
3. Complete failure in reproducing middle-high frequencies - giving a constant telephone-like tone on the background.
4. The NN was trained on low-quality (telephonic quality) audio files. Low-quality inputs result in low-quality outputs.

### Data Preprocessing

Before training, the audio data undergoes preprocessing. The original `.mp3` audio file is converted into a **waveform format**, which represents the amplitude of the sound wave over time. This conversion enables the model to work with numerical data suitable for neural network training.

The waveform format captures:
- The **loudness** (amplitude) of the audio signal at each time step.
- Temporal patterns essential for modeling rhythms, beats, and melodies.

Below is a plot of a converted `.mp3` audio file in waveform format (orange curve) and the predicted one by the NN (violet). The x-axis represents time (microseconds), and the y-axis represents the amplitude of the signal.

![Sample Output](https://github.com/Vlasenko2006/Lets_Rock/blob/main/waveformat.png)

---

### Wave Format or (Mel)Spectrogram?

Indeed, spectrogrgrams give a 2D representation of audio signal and they better picture its pecularities than 1D Wave format. Still, extra dimention gives you extra memory costs and complicates the problem of music reproduction (dimension curse).   

### Training Process

1. **Data Preprocessing**:
   - Audio tracks are converted into **NumPy arrays**.
   - Each waveform is split into 10-second chunks.
   - Pairs are formed where the first chunk serves as the input, and the second chunk serves as the target.

2. **Cost Function**:
   - The cost function consists of two components:
     - **Reconstruction Loss**: Compares the input audio with the reconstructed audio from the encoder-decoder.
     - **Prediction Loss**: Measures the difference between the predicted continuation and the target continuation using Mean Squared Error (MSE).

3. **Training**:
   - The model is trained in two stages:
     - In the first epochs, the encoder and decoder are trained together to minimize the reconstruction loss.
     - Next, the transformer is added, and the full network is trained to predict the continuation of audio tracks.

---

### Installation Instructions

To set up the necessary environment, follow these steps:

1. Install Python 3.9 or higher.
2. Clone the repository and navigate to its directory:
   ```bash
   git clone https://github.com/your-repo/audio-prediction-model.git
   cd audio-prediction-model
   ```
3. Install the required dependencies using either `pip` or `conda`:
   - Using `pip`:
     ```bash
     pip install -r requirements.txt
     ```
   - Using `conda`:
     ```bash
     conda env create -f environment.yml
     conda activate audio-prediction
     ```
4. Ensure `ffmpeg` is installed and available in your system's PATH. You can install it via a package manager:
   - On Ubuntu: 
     ```bash
     sudo apt update
     sudo apt install ffmpeg
     ```
   - On macOS:
     ```bash
     brew install ffmpeg
     ```
   - On Windows: Download the binaries from [FFmpeg's official website](https://ffmpeg.org/) and configure your PATH.

---

### Example Usage

To preprocess audio files, create a dataset, and train the model, use the following commands:

1. **Preprocess Audio Files**:
   ```bash
   python preprocess.py --input_folder /path/to/audio --output_folder /path/to/output
   ```
2. **Create Dataset**:
   ```bash
   python create_dataset.py --output_folder /path/to/output --dataset_folder /path/to/dataset
   ```
3. **Train the Model**:
   ```bash
   python train.py --dataset_folder /path/to/dataset --checkpoint_folder /path/to/checkpoints
   ```

---

### Limitations and Future Improvements

- **Computational Constraints**: The current setup is optimized for a single GPU with 15GB of RAM. Expanding the model's capacity (e.g., more layers and transformer heads) would improve its ability to capture complex musical structures like vocals and rhythm guitars.
- **Dataset Size**: A larger and more diverse dataset would significantly enhance the model's generalization and output quality.

---

### Requirements Files

Below are the `requirements.txt` and `environment.yml` files for setting up the Python environment.

#### `requirements.txt` (For `pip`)

```plaintext name=requirements.txt
librosa==0.10.0.post2
numpy==1.23.5
scipy==1.10.1
torch==2.0.1
soundfile==0.12.1
pydub==0.25.1
tqdm==4.65.0
```

#### `environment.yml` (For `conda`)

```yaml name=environment.yml
name: audio-prediction
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.9
  - librosa=0.10.0
  - numpy=1.23.5
  - scipy=1.10.1
  - pytorch=2.0.1
  - soundfile=0.12.1
  - pydub=0.25.1
  - tqdm=4.65.0
  - ffmpeg
```

---

### Acknowledgments

This project was created by **Andrey Vlasenko** and is an ongoing experiment in music generation. The work demonstrates the potential of compact neural networks for music composition and prediction under constrained computational resources.
