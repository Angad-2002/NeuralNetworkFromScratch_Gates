# Neural Network from Scratch - Logic Gates

Implementation of a Neural Network in Python from scratch for plotting the decision boundary of XOR gate and Majority Function (3 Inputs).

## Features
- Supports training for AND, OR, XOR gates, and Majority Function.
- Uses a simple feedforward neural network with adjustable layers.
- Implements backpropagation and gradient descent.
- Visualizes decision boundaries with Matplotlib.

## Installation

Clone the repository:
```bash
git clone https://github.com/Angad-2002/NeuralNetworkFromScratch_Gates.git
cd NeuralNetworkFromScratch_Gates
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the neural network training script:
```bash
python train.py
```

Modify `train.py` to change the logic gate being trained and adjust the number of neurons in each layer accordingly. Example:
```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR Gate

# Adjust the number of neurons for XOR
neural_network = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
```
For the Majority Function (3 inputs), update the input size and hidden layer configuration:
```python
X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
              [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([[0], [0], [0], [1], [0], [1], [1], [1]])  # Majority Function

# Adjust the number of neurons for Majority Function
neural_network = NeuralNetwork(input_size=3, hidden_size=6, output_size=1)
```

## Results
After training, the model will output:
- The learned weights and biases.
- The accuracy of predictions on the given dataset.
- A scatter plot showing the decision boundary.

## Dependencies
- Python 3.x
- NumPy
- Matplotlib

## Contributing
Feel free to submit issues or pull requests to improve the project.

## License
This project is licensed under the MIT License.

## Author
Developed by **Angad Singh**
