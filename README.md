# Vision-Based Cycling Fatigue Detection

This repository contains the code and data for the paper:  
**"The Fatigue Status Feature of Bicycle Movement Based on Deep Learning and Signal Processing Technology"**  
*Authors: Yingchun He, Yi-haw Jan, Fan Yang, Yunru Ma, Xin-Yuan Chen, Chun Pei*  
*Published in: []*

---

## Abstract
Cycling is a common and effective home-based rehabilitation exercise. Accurate and accessible assessment of the onset of fatigue is essential to achieving optimal exercise benefits and preventing overuse injuries. To obtain fatigue-related parameters in different age groups, we applied deep learning algorithms and signal processing technology to analyze cycling movement features for the people aged over 45. 20 healthy adults aged over 45 and 20 aged 18-30 were recruited. Participants were asked to ride a stationary exercise bike at their self-regulated pedaling speeds for 10 minutes and wear a COSMED K5 device to collect physiological signals. The Keypoint RCNN (KR) algorithm and three signal processing methods (Fourier transform(FT), short-time Fourier transform(STFT), and multiscale entropy analysis(MSE) were used to analyze the cycling movement data. Based on time-frequency analysis, subjects’ movement status change points were identified when fatigue occurred. Four movement status parameters were calculated,  including the peak frequency before/after the movement status change point and the complexity index average (CIA) before/after the movement status change point. Inter-group and intra-group movement features, movement status, and physiological data were compared to determine fatigue-related features.  Results showed that the peak frequency (p=0.005), the peak frequency before/after the change point (p=0.008/0.019), the CIA after the change point (p=0.014), the maximum heart rate, maximal oxygen consumption, metabolic equivalents, and energy efficiency exhibited significant inter-group differences. The KR algorithm demonstrated outstanding performance in keypoint detection, achieving an accuracy of 86.5%, significantly outperforming OpenPose. With an inference speed of 30 FPS, it fulfills the demands for real-time monitoring. In addition, CIA valuses before and after change points showed significant differences in the the middle-aged and elderly people group. After the change point, the CIA can identify movement status changes in inter-group and intra-group comparisons, suggesting it can be used as a indicator of fatigue status, especially for people aged over 45

## Repository Structure
```
Vision-Based-Fatigue-Detection/
├── data/                 # Preprocessed and anonymized study data
│   ├── kinematic/        # Processed joint trajectory data (.csv)
│   └── physiological/    # HR, VO2max, RPE, etc. data (.csv)
├── src/                  # Source code
│   ├── inference.py              # Keypoint RCNN pose estimation
│   ├── signal_processing.py      # Fourier Transform, STFT, MSE analysis
│   ├── analysis.py               # Statistical tests and figure generation
│   └── requirements.txt          # Python package dependencies
├── models/               # (Optional) Pre-trained model weights
├── LICENSE
└── README.md
```

## Installation & Usage
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/Vision-Based-Fatigue-Detection.git
    cd Vision-Based-Fatigue-Detection
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r src/requirements.txt
    ```
3.  **Run the analysis:**
    Execute the scripts in `src/` in the following order:
    - `inference.py`: To generate kinematic data from video (if raw video is available).
    - `signal_processing.py`: To perform FT, STFT, and MSE analysis.
    - `analysis.py`: To reproduce all statistical results and figures from the paper.

## Data Description
The data in `data/` directory is anonymized. It includes:
- `kinematic/`: Right knee vertical (Y-direction) acceleration data and derived features (PF, CIA, etc.).
- `physiological/`: Physiological parameters (HRmax, VO2max, METS, EEm, RPE) for all participants.

## Citation
If you use this code or data in your research, please cite both our paper and the archived version of this code:
```bibtex
@article{he2025fatigue,
  title={The Fatigue Status Feature of Bicycle Movement Based on Deep Learning and Signal Processing Technology},
  author={He, Yingchun and Jan, Yi-haw and Yang, Fan and Ma, Yunru and Chen, Xin-Yuan and Pei, Chun},
  journal={Scientific Reports},
  year={2025},
  publisher={Nature Publishing Group}
}
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions, please open an issue on GitHub or contact [yingchun01] at [orange4hyc@163.com].
