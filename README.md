# DeCoMa: Detecting and Purifying Code Dataset Watermarks through Dual Channel Code Abstraction

<img src="framework.png" width="1000px"/>


## Preparation 

### Setup Environments 
1. Install Anaconda Python [https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/)
2. `conda create --name DeCoMa python=3.8 -y` ([help](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))
3. `conda activate DeCoMa`

    1. `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge` 
    2. `pip install transformers==4.33.2`

### Download Dataset 
CodeSearchNet dataset can be downloaded through the following links: [https://github.com/microsoft/CodeXGLUE](https://github.com/microsoft/CodeXGLUE) 

## DeCoMa
1. Dataset Preprocessing
```bash
python preprocess.py
```

2. Run `DeCoMa` on the preprocessed dataset
    
```bash
python DeCoMa.py    
``` 

## Other Baselines

1. Run SS and AC
    ```bash
    cd SS_AC
    python defense_ss_ac.py
    ```
2. Run CodeDetector
    ```bash
    cd CodeDetector
    python CodeDetector.py
    ```

-------------------------------------------------
## Hyperparameters
To evaluate the impact of DeCoMa on model performance and to determine whether watermark verification can be bypassed after removing watermarked samples, we train a model for verification, CodeT5, which is a commonly used NCM. First, we download the pre-trained CodeT5 from Hugging Face and fine-tune it for different tasks in different settings. Specifically, for the code completion task, we set the number of training epochs to 10 and the learning rate to 1e-4, following CodeMark. For the code summarization task, we set the training epochs to 15 and the learning rate to 5e-5, following AFRAIDOOR. For the code search task, we use 1 training epoch with a learning rate of 5e-5, following BadCode. All models are trained using the Adam optimizer. Our experiments are implemented using PyTorch 1.13.1 and Transformers 4.38.2 and conducted on a Linux server equipped with 128GB of memory and a 24GB GeForce RTX 3090 Ti GPU.
