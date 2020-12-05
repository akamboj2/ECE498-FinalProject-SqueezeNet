Template for neural network development

To train a neural network from scratch,

run the following command: "python main_train.py /path/to/yaml file"

The script will generate a pt.tar file that contains:
- model state
- optimizer state
- learning rate scheduler state
- training&testing accuracy evolution

To evaluate the generated model, run:

"python main_eval.py /path/to/.pt.tar file"

Since the given model is already trained, you just need to run the command for model evaluation
