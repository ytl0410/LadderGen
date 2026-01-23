# LadderGen
Well-trained models and generative outcomes for the paper "LadderGen: A Large-Scale Generative Library of Ladder Polymers for Membrane Separation." 
# Generation results
All generation results can be found at: https://zenodo.org/records/18275495
# Well-trained models
Well-trained CharRNN, REINVENT, minGPT, and Diffusion models can be found at: https://zenodo.org/records/18275968
## CharRNN
Download the Moses Docker image.
```
docker pull molecular sets/moses
```
Create a container.
```
nvidia-docker run -it --name moses --network="host" --shm-size 10G molecularsets/moses
```
Copy well-trained model files from the local machine to the Moses Docker container.
```
docker cp /Your address/CharRNN-config.pt [CONTAINER ID]:/moses/char_rnn_ladder/CharRNN-config.pt
docker cp /Your address/CharRNN-model.pt [CONTAINER ID]:/moses/char_rnn_ladder/CharRNN-model.pt
docker cp /Your address/CharRNN-vocab.pt [CONTAINER ID]:/moses/char_rnn_ladder/CharRNN-vocab.pt
```
Load the Moses Docker container.
```
docker start moses
docker attach moses
```
Use the well-trained CharRNN model to generate hypothetical polymer structures.
```
python scripts/sample.py char_rnn --model_load char_rnn_ladder/CharRNN-model.pt --vocab_load char_rnn_ladder/CharRNN-vocab.pt --config_load char_rnn_ladder/CharRNN-config.pt --n_samples 2000000 --gen_save char_rnn_ladder/char_rnn_gene_2m.csv
```
## REINVENT
Please refer to https://github.com/MolecularAI/Reinvent to install REINVENT.

Use the trained REINVENT model to generate hypothetical polymer structures.
```
python ./sample_from_model.py -m models/REINVENT.agent -o reinvent_gene_2m.csv -n 2000000
```
## minGPT
Please refer to https://github.com/TRI-AMDD/PolyGen to install minGPT.

Use the trained minGPT model to generate hypothetical polymer structures.
```
generate_config.ckpt_path = "./minGPT/MinGPT.pt"
```

## diffusion1D
Please refer to https://github.com/TRI-AMDD/PolyGen to install minGPT.

Use the trained diffusion1D model to generate hypothetical polymer structures.
```
train_config = pipeline.get_default_train_config()
train_config.ckpts_path = "./diffusion1D"
train_config.task = "unconditional"
print(train_config.device)
train_config.num_steps = 1
```
Use the provided model to replace the generated one. 
```
generate_config.model_index = 1
```
