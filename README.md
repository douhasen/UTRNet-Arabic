# UTRNet-Arabic

This project focuses on enhancing the OCR (Optical Character Recognition) accuracy for Arabic documents, particularly handwritten text. 

This project was inspired by the [UTRNet High-Resolution Urdu Text Recognition project](https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition).

## Installation
1. Clone the repository
```
git clone https://github.com/douhasen/UTRNet-Arabic.git
```

2. Install the requirements
```
conda create -n ocr_arabic python=3.7
conda activate ocr_arabic
pip3 install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Dataset
1. Download KHATT dataset  

In this project, we use the KHATT dataset for training and evaluating our model. The KHATT dataset is a comprehensive collection of handwritten Arabic text.
You can access the KHATT dataset [here](https://khatt.ideas2serve.net/).

2. Create your lmdb dataset
```
python create_lmdb_dataset.py --inputPath data/train --gtFile data/train/gt.txt --outputPath result/train
```
Ensure that the LMDB dataset is created for the train, valid, and test datasets.

## Running the Project

1. Training
```
python train.py --train_data result/train --valid_data result/valid --FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC --exp_name UTRNet-Arabic --num_epochs 100 --batch_size 8 --device_id 0
```

2. Testing
```
CUDA_VISIBLE_DEVICES=0 python test.py --eval_data result/test/ --FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC --saved_model saved_models/UTRNet-Arabic/model.pth
```

3. Character-wise Accuracy Testing
* To create character-wise accuracy table in a CSV file, run the following command

```
CUDA_VISIBLE_DEVICES=0 python3 char_test.py --eval_data result/test/ --FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC  --saved_model saved_models/UTRNet-Arabic/model.pth
```

* Visualize the result by running ```char_test_vis```

4. Reading images
* To read a single image, run the following command:

```
CUDA_VISIBLE_DEVICES=0 python3 read.py --image_path path/to/image.png --FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC  --saved_model saved_models/UTRNet-Arabic/model.pth
```

* To read a folder containing multiple images, run the following command:

```
CUDA_VISIBLE_DEVICES=0 python3 read_all.py --image_folder path/to/folder --FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC --saved_model saved_models/UTRNet-Arabic/model.pth --output_file read_all_outputs.txt
```

