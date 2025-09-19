# SimpleVQA: A Simple Video Quality Assessment Model

This is the official implementation of the paper "SimpleVQA: A Simple Video Quality Assessment Model".

## Getting Started

### Prerequisites

* Python 3.8+
* PyTorch
* CUDA

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/SimpleVQA-main.git
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Testing

To test the model on a single video, run the following command:

```bash
python src/test_demo.py --method_name single-scale --dist <path-to-video> --output result.txt --is_gpu
```

### Training

To train the model, run the following command:

```bash
bash scripts/train.sh
```

## Project Structure

```
.
├── .idea
├── LICENSE
├── LSVQ
├── LSVQ_image
├── README.md
├── __pycache__
├── ckpts
├── data
├── docs
├── logs
├── output.txt
├── requirements.txt
├── result.txt
├── scripts
│   ├── extract_features.sh
│   ├── extract_frames.sh
│   ├── test.sh
│   └── train.sh
└── src
    ├── data_loader.py
    ├── extract_frame_LSVQ.py
    ├── extract_frame_konvid1k.py
    ├── model
    │   ├── UGC_BVQA_model.py
    │   └── __pycache__
    ├── test_demo.py
    ├── train.py
    └── utils.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
# VQA
