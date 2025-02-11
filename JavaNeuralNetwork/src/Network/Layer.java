package Network;

import java.util.Random;

public class Layer {
    int inputSize, neuronCount;
    double[][] weights;
    double[] biases;
    double[] outputs;
    ActivationFunction activation;

    public Layer(int inputSize, int neuronCount, ActivationFunction activation) {
        this.inputSize = inputSize;
        this.neuronCount = neuronCount;
        this.activation = activation;
        this.weights = new double[inputSize][neuronCount];
        this.biases = new double[neuronCount];
        this.outputs = new double[neuronCount];

        initWeights();
    }

    private void initWeights() {
        Random rand = new Random();
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < neuronCount; j++) {
                weights[i][j] = rand.nextDouble() - 0.5;
            }
        }
        for (int j = 0; j < neuronCount; j++) {
            biases[j] = rand.nextDouble() - 0.5;
        }
    }

    public double[] forward(double[] inputs) {
        double[] sums = new double[neuronCount]; // Stocke les sommes pondérées

        for (int j = 0; j < neuronCount; j++) {
            double sum = biases[j];
            for (int i = 0; i < inputSize; i++) {
                sum += inputs[i] * weights[i][j];
            }
            sums[j] = sum; // Stocke le résultat
        }

        // Appliquer Softmax si c'est la dernière couche
        if (activation instanceof SoftmaxActivation) {
            //cast pour pas avoir d'erreur
            SoftmaxActivation a = (SoftmaxActivation) this.activation;
            outputs = a.activate(sums);
        } else {
            for (int j = 0; j < neuronCount; j++) {
                outputs[j] = activation.activate(sums[j]); // Applique sur chaque neurone
            }
        }
        return outputs;
    }
}