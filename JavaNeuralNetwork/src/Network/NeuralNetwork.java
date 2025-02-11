package Network;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    private List<Layer> layers = new ArrayList<>();
    private double learningRate = 0.1;

    public void addLayer(int neuronCount, ActivationFunction activation) {
        int inputSize = (layers.isEmpty()) ? neuronCount : layers.get(layers.size() - 1).neuronCount;
        layers.add(new Layer(inputSize, neuronCount, activation));
    }

    public double[] forward(double[] inputs) {
        double[] outputs = inputs;
        for (Layer layer : layers) {
            outputs = layer.forward(outputs);
        }
        return outputs;
    }

    public void train(double[][] trainingInputs, double[][] trainingOutputs, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0;
            for (int i = 0; i < trainingInputs.length; i++) {
                double[] outputs = forward(trainingInputs[i]);
                backward(trainingInputs[i], outputs, trainingOutputs[i]);

                for (int j = 0; j < outputs.length; j++) {
                    totalError += Math.pow(trainingOutputs[i][j] - outputs[j], 2);
                }
            }
            if (epoch % 1000 == 0) {
                System.out.println("Epoch " + epoch + " - Error: " + totalError);
            }
        }
    }

    private void backward(double[] inputs, double[] outputs, double[] expectedOutputs) {
        double[] errors = new double[outputs.length];
        double[] gradients = new double[outputs.length];

        for (int i = 0; i < outputs.length; i++) {
            errors[i] = expectedOutputs[i] - outputs[i];
            gradients[i] = errors[i] * layers.get(layers.size() - 1).activation.derivative(outputs[i]);
        }

        for (int l = layers.size() - 1; l >= 0; l--) {
            Layer layer = layers.get(l);
            double[] prevOutputs = (l == 0) ? inputs : layers.get(l - 1).outputs;

            for (int j = 0; j < layer.neuronCount; j++) {
                for (int i = 0; i < layer.inputSize; i++) {
                    layer.weights[i][j] += learningRate * gradients[j] * prevOutputs[i];
                }
                layer.biases[j] += learningRate * gradients[j];
            }

            if (l > 0) {
                double[] newGradients = new double[layer.inputSize];
                for (int i = 0; i < layer.inputSize; i++) {
                    for (int j = 0; j < layer.neuronCount; j++) {
                        newGradients[i] += layer.weights[i][j] * gradients[j];
                    }
                    newGradients[i] *= layers.get(l - 1).activation.derivative(prevOutputs[i]);
                }
                gradients = newGradients;
            }
        }
    }
}