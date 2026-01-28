# DiCoW: Diarization-Conditioned Whisper for Target Speaker Automatic Speech Recognition

DiCoW (Diarization-Conditioned Whisper) enhances OpenAI’s Whisper ASR model by integrating **speaker diarization** for multi-speaker transcription. The app leverages `BUT-FIT/diarizen-wavlm-large-s80-md` to segment speakers and provides diarization-conditioned transcription for long-form audio inputs.

Training and inference source codes can be found here: [TS-ASR-Whisper](https://github.com/BUTSpeechFIT/TS-ASR-Whisper)

> **Note:** For the original v1 model, see the [`v1` branch](https://github.com/BUTSpeechFIT/DiCoW/tree/v1).

## Features

- **Multi-Speaker ASR**: Handles multi-speaker audio using diarization-aware transcription.  
- **Flexible Input Sources**:  
  - **Microphone**: Record and transcribe live audio.  
  - **Audio File Upload**: Upload pre-recorded audio files for transcription.  
  - **Folder Batch Processing** – Process multiple .wav files from a directory via the command line.
- **Diarization Support**: Powered by `BUT-FIT/diarizen-wavlm-large-s80-md` for accurate speaker segmentation.  
- **Built with 🤗 Transformers**: Uses the latest Whisper checkpoints for robust transcription.  

## Demo

![DiCoW-v1 Demo](img.png)  

### Online Usage
Run the app directly in your browser with [Gradio app](https://pccnect.fit.vutbr.cz/gradio-demo).

## Installation

### Requirements

Before running the app, ensure you have the following installed:

- **Python 3.11**  
- **FFmpeg**: Required for audio processing.
- Python Libraries:  
  - `gradio`  
  - `transformers`  
  - `pyannote.audio`  
  - `torch`
  - `librosa`
  - `soundfile`

### Setup

1. Clone the repository:  
    ```bash 
   git clone https://github.com/BUTSpeechFIT/DiCoW.git
   cd DiCoW  
    ```

2. Clone DiariZen submodule:
   ```bash
   git submodule init
   git submodule update
   cd DiariZen
   git submodule init
   git submodule update
   cd ..
   ```

3. Setup dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Export DiariZen to path
    ```bash
    export PYTHONPATH=$PWD/DiariZen:$PYTHONPATH 
    ```
## Usage

### Web Interface

Run the application locally:  
```bash
  python app.py  
```

Once the server is running, access the app in your browser at `http://localhost:7860`.

### Processing a Folder of WAV Files (Command Line)

To process multiple `.wav` files at once, run:

```bash
python inference.py --input-folder /path/to/wav/files
```

### Linux service

If you want to run this demo on background, it may be good to make a service out of it. (some distros kill the background jobs when user logs out, hence kill the demo).

To register the demo as service, first edit `./run_server.sh` and `./DiCoW-background.service` and set proper paths and users. It is important to set the conda correctly in `./run_server.sh` 
as the service is started out of the userspace (`.profile`).

Then register and start the service (run as root):
```
systemctl enable ./DiCoW-background.service #register the service
systemctl start DiCoW-background.service #start
systemctl status DiCoW-background.service #check if it is running
systemctl stop DiCoW-background.service #stop
systemctl disable DiCoW-background.service #will not start on restart anymore
```

### Modes

1. **Microphone**: Use your device's microphone for live transcription.  
2. **Audio File Upload**: Upload pre-recorded audio files for diarization-conditioned transcription.
3. **Folder Batch Processing**: Process multiple WAV files from command line for automated workflows.

## Contributing
We welcome contributions! If you’d like to add features or improve the app, please open an issue or submit a pull request.

## License

This project combines multiple components, each with its own license:

* **DiCoW** (this repository): Licensed under the [Apache License 2.0](LICENSE).
* **DiCoW Model Weights**: Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) – attribution required for usage.
* **Diarizen (BUT-FIT/diarizen-wavlm-large-s80-mlc)**: Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) – free for research and non-commercial use only.

Please ensure compliance with the respective licenses when using, modifying, or redistributing these components.

## Citation
If you use our model or code, please, cite:
```
@INPROCEEDINGS{polok2026sedicow,
  author={Polok, Alexander and Klement, Dominik and Cornell, Samuele and Wiesner, Matthew and Černocký, Jan and Khudanpur, Sanjeev and Burget, Lukáš},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={SE-DiCoW: Self-Enrolled Diarization-Conditioned Whisper}, 
  year={2026},
}

@article{POLOK2026101841,
    title = {DiCoW: Diarization-conditioned Whisper for target speaker automatic speech recognition},
    journal = {Computer Speech & Language},
    volume = {95},
    pages = {101841},
    year = {2026},
    issn = {0885-2308},
    doi = {https://doi.org/10.1016/j.csl.2025.101841},
    url = {https://www.sciencedirect.com/science/article/pii/S088523082500066X},
    author = {Alexander Polok and Dominik Klement and Martin Kocour and Jiangyu Han and Federico Landini and Bolaji Yusuf and Matthew Wiesner and Sanjeev Khudanpur and Jan Černocký and Lukáš Burget},
    keywords = {Diarization-conditioned Whisper, Target-speaker ASR, Speaker diarization, Long-form ASR, Whisper adaptation},
}

@INPROCEEDINGS{10887683,
  author={Polok, Alexander and Klement, Dominik and Wiesner, Matthew and Khudanpur, Sanjeev and Černocký, Jan and Burget, Lukáš},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Target Speaker ASR with Whisper}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Transforms;Signal processing;Transformers;Acoustics;Speech processing;target-speaker ASR;diarization conditioning;multi-speaker ASR;Whisper},
  doi={10.1109/ICASSP49660.2025.10887683}
}
```

## Contact
For more information, feel free to contact us: [ipoloka@fit.vut.cz](mailto:ipoloka@fit.vut.cz), [xkleme15@vutbr.cz](mailto:xkleme15@vutbr.cz).
