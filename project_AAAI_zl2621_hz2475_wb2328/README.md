# Enhancing SeqGAN for Text Generation

This code was built upon the original [SeqGAN](https://github.com/LantaoYu/SeqGAN).

## Directory Structure

├── seqgan-obama # Codes for Obama's speeches generation

│   ├── checkpoints # Trained models

│   ├── get_1nn_mapping.py # Compute nearest neighbour of words

│   │   ├── id2tok.py # Can convert the generated files with word ids to words

│   ├── sequence_gan.py # Entry point of training

│   ├── test_gan.py # Entry point of testing

│   └── test_glob_bleu.py # Only for unit test purpose

└── seqgan-trump # Codes for style transfer to Trump's speeches 
	── seqgan_transfer.py # Entry point of Transfer training


## How to run

### Obama's speeches generation
+ For training: ```python sequence_gan.py``` (running this will overwrite the saved model)
+ For testing: ```python test_gan.py``` (run this directly to test our model)

### Trump's speeches generation
+ Before training, you need to copy the pretrained Obama model ```checkepoints``` to this directory
+ For training: ```python seqgan_transfer.py``` (running this will overwrite the saved model)
+ For testing: ```python test_gan.py``` (run this directly to test our model)