package Network;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NeuralNetwork {
    private List<Layer> layers = new ArrayList<>();
    private double learningRate = 0.1;

    public NeuralNetwork() {
    }

    public NeuralNetwork(String path) {
        load(path); // Charge le réseau à partir du fichier
    }

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

    public void train(double[][] trainingInputs, double[][] trainingOutputs, int epochs, boolean info) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0;
            for (int i = 0; i < trainingInputs.length; i++) {
                double[] outputs = forward(trainingInputs[i]);

                // Calculate cross-entropy loss
                for (int j = 0; j < outputs.length; j++) {
                    totalError -= trainingOutputs[i][j] * Math.log(outputs[j] + 1e-15);
                }

                backward(trainingInputs[i], outputs, trainingOutputs[i]);
            }

            if (info || epoch % 1000 == 0) {
                System.out.println("Epoch " + epoch + " - Loss: " + (totalError / trainingInputs.length));
            }
        }
    }

    public void setLearningRate(double rate) {
        this.learningRate = Math.max(rate, 1e-6); // Prevent zero or negative values
    }

    public double getLearningRate() {
        return this.learningRate;
    }

    private void backward(double[] inputs, double[] outputs, double[] expectedOutputs) {
        double[] gradients = new double[outputs.length];

        // Compute loss gradient
        for (int i = 0; i < outputs.length; i++) {
            gradients[i] = outputs[i] - expectedOutputs[i];
        }

        // Gradient Clipping to prevent explosions
        double maxGradient = Arrays.stream(gradients).map(Math::abs).max().orElse(1.0);
        double clipThreshold = 5.0; // Set a reasonable threshold

        if (maxGradient > clipThreshold) {
            for (int i = 0; i < gradients.length; i++) {
                gradients[i] = (gradients[i] / maxGradient) * clipThreshold;
            }
        }

        // Backpropagation
        for (int l = layers.size() - 1; l >= 0; l--) {
            Layer layer = layers.get(l);
            double[] prevOutputs = (l == 0) ? inputs : layers.get(l - 1).outputs;

            for (int j = 0; j < layer.neuronCount; j++) {
                for (int i = 0; i < layer.inputSize; i++) {
                    layer.weights[i][j] -= learningRate * gradients[j] * prevOutputs[i];
                }
                layer.biases[j] -= learningRate * gradients[j];
            }

            // Backpropagate gradients
            if (l > 0) {
                double[] newGradients = new double[layer.inputSize];
                for (int i = 0; i < layer.inputSize; i++) {
                    for (int j = 0; j < layer.neuronCount; j++) {
                        newGradients[i] += layer.weights[i][j] * gradients[j];
                    }
                }
                gradients = newGradients;
            }
        }
    }

    public void save(String path) {
        for(int i = 0; i < layers.size(); i++) {
            // Extraire le dossier du chemin (supprime tout après le dernier '/')
            File directory = new File(path).getParentFile();

            // Vérifier si le dossier existe, sinon le créer
            if (directory != null && !directory.exists()) {
                directory.mkdirs(); // Crée le dossier et ses parents si nécessaire
            }

            Layer layer = layers.get(i);
            String layerPath = path + "_layer" + i + ".txt";
            try (PrintWriter writer = new PrintWriter(layerPath)) {
                writer.println(layer.saveToString());
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
    }

    public void load(String path) {
        layers.clear();
        int i = 0;
        while (true) {
            String layerPath = path + "_layer" + i + ".txt";
            try {
                String content = new String(Files.readAllBytes(Paths.get(layerPath)));
                Layer layer = Layer.loadFromString(content);
                layers.add(layer);
            } catch (FileNotFoundException e) {
                break;
            } catch (IOException e) {
                break;
            }
            i++;
        }
    }
}