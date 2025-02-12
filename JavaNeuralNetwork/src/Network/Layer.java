package Network;

import java.util.Arrays;
import java.util.Random;

public class Layer {
    int inputSize;
    int neuronCount;
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
        // He initialization for ReLU
        double scale = Math.sqrt(2.0 / inputSize);

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < neuronCount; j++) {
                weights[i][j] = rand.nextGaussian() * scale * 0.5; // Reduce scaling
            }
        }

        for (int j = 0; j < neuronCount; j++) {
            biases[j] = 0; // Start with zero bias
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

    public String saveToString() {
        StringBuilder sb = new StringBuilder();
        sb.append(inputSize).append(" ").append(neuronCount).append("\n");

        // Ajouter le nom de la fonction d'activation
        sb.append(activation.getClass().getSimpleName()).append("\n");

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < neuronCount; j++) {
                sb.append(weights[i][j]).append(" ");
            }
            sb.append("\n");
        }

        for (int j = 0; j < neuronCount; j++) {
            sb.append(biases[j]).append(" ");
        }
        sb.append("\n");

        return sb.toString();
    }

    public static Layer loadFromString(String data) {
        String[] lines = data.split("\n");
        String[] sizes = lines[0].split(" ");
        int inputSize = Integer.parseInt(sizes[0]);
        int neuronCount = Integer.parseInt(sizes[1]);

        // Lire la fonction d'activation
        String activationName = lines[1].trim();
        ActivationFunction activation = getActivationFunctionByName(activationName);

        Layer layer = new Layer(inputSize, neuronCount, activation);

        // Lire les poids correctement, ligne par ligne
        for (int i = 0; i < inputSize; i++) {
            String[] w = lines[i + 2].split(" "); // Décalage de 2 (1 pour tailles, 1 pour activation)
            for (int j = 0; j < neuronCount; j++) {
                layer.weights[i][j] = Double.parseDouble(w[j]);
            }
        }

        // Lire les biais qui sont à la dernière ligne
        String[] b = lines[inputSize + 2].split(" ");
        for (int j = 0; j < neuronCount; j++) {
            layer.biases[j] = Double.parseDouble(b[j]);
        }

        return layer;
    }

    private static ActivationFunction getActivationFunctionByName(String name) {
        switch (name) {
            case "ReLUActivation": return new ReLUActivation();
            case "SigmoidActivation": return new SigmoidActivation();
            case "SoftmaxActivation": return new SoftmaxActivation();
            case "LeakyReLUActivation": return new LeakyReLUActivation();
            default: throw new IllegalArgumentException("Unknown activation function: " + name);
        }
    }
}