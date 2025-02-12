import Network.*;
import Utils.DataSet;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainMnist {
    public static void main(String[] args) {
        // Create network with modified parameters
        NeuralNetwork nn = new NeuralNetwork();
        nn.setLearningRate(0.0005); // Much smaller learning rate

        // Modified architecture
        nn.addLayer(784, new LeakyReLUActivation());
        nn.addLayer(128, new LeakyReLUActivation()); // Larger middle layer
        nn.addLayer(64, new LeakyReLUActivation());  // Additional layer for stability
        nn.addLayer(10, new SoftmaxActivation());

        // Load smaller batches initially
        DataSet trainData = loadData("data/MNIST/mnist_train.csv", 100000000);

        // Smaller batch size for better stability
        int batchSize = 16;

        // Training with additional monitoring
        for (int epoch = 0; epoch < 1; epoch++) {
            System.out.println("\n=== Epoch " + epoch + " ===");

            for (int i = 0; i < trainData.inputs.length; i += batchSize) {
                int end = Math.min(i + batchSize, trainData.inputs.length);
                double[][] batchInputs = Arrays.copyOfRange(trainData.inputs, i, end);
                double[][] batchOutputs = Arrays.copyOfRange(trainData.outputs, i, end);

                nn.train(batchInputs, batchOutputs, 1, false);
            }
        }

        nn.save("saves/mnistfiles/mnist8");
        /*
        String csvFile = "data/MNIST/mnist_train.csv";
        List<double[]> inputs = new ArrayList<>();
        List<double[]> outputs = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line;
            br.readLine(); // Ignore la première ligne (labels)
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");

                // Label (converti en one-hot encoding)
                double[] output = new double[10];
                int label = Integer.parseInt(values[0]); // Premier élément = chiffre (0-9)
                output[label] = 1.0;

                // Pixels normalisés entre 0 et 1
                double[] input = new double[784];
                for (int i = 0; i < 784; i++) {
                    input[i] = Integer.parseInt(values[i + 1]) / 255.0;
                }

                inputs.add(input);
                outputs.add(output);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Conversion en tableau pour l'entraînement
        double[][] trainingInputs = inputs.toArray(new double[0][]);
        double[][] trainingOutputs = outputs.toArray(new double[0][]);

        //delete les listes
        inputs = null;
        outputs = null;


        // Création et entraînement du réseau de neurones
        NeuralNetwork nn = new NeuralNetwork();

        // Architecture optimized for MNIST
        nn.setLearningRate(0.001); // Smaller learning rate for stability
        nn.addLayer(784, new ReLUActivation());  // Input layer
        nn.addLayer(256, new ReLUActivation());  // Hidden layer 1
        nn.addLayer(128, new ReLUActivation());  // Hidden layer 2
        nn.addLayer(10, new SoftmaxActivation()); // Output layer

        */
        /*
        nn.addLayer(784, new SigmoidActivation());
        nn.addLayer(128, new SigmoidActivation());
        nn.addLayer(10, new SoftmaxActivation());
        */
        //NeuralNetwork nn = new NeuralNetwork("saves/mnistfiles/mnistSigmo");
        //nn.train(trainingInputs, trainingOutputs, 1, true);
        //nn.save("saves/mnistfiles/mnist7");

        /*

        nn.addLayer(2, new SigmoidActivation());
        //nn.addLayer(4, new SigmoidActivation());
        nn.addLayer(1, new SigmoidActivation());

        double[][] trainingInputs = {
                {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };
        double[][] trainingOutputs = {
                {0}, {1}, {1}, {0}
        };

        nn.train(trainingInputs, trainingOutputs, 100000);

        nn.save("saves/xorfiles/xor");
        System.out.println("Prédictions :");
        for (double[] input : trainingInputs) {
            System.out.println(input[0] + " XOR " + input[1] + " = " + nn.forward(input)[0]);
        }
        */
    }

    private static DataSet loadData(String csvFile, int maxSamples) {
        List<double[]> inputs = new ArrayList<>();
        List<double[]> outputs = new ArrayList<>();
        int count = 0;

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line;
            // Skip header if it exists
            if (br.readLine().contains("label")) {
                System.out.println("Skipped header row");
            }

            while ((line = br.readLine()) != null && count < maxSamples) {
                String[] values = line.split(",");

                // One-hot encoding for output
                double[] output = new double[10];
                int label = Integer.parseInt(values[0]);
                output[label] = 1.0;

                // Normalize pixel values to [0, 1]
                double[] input = new double[784];
                for (int i = 0; i < 784; i++) {
                    input[i] = Integer.parseInt(values[i + 1]) / 255.0;
                }

                inputs.add(input);
                outputs.add(output);
                count++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.printf("Loaded %d samples from %s\n", count, csvFile);
        return new DataSet(
                inputs.toArray(new double[0][]),
                outputs.toArray(new double[0][])
        );
    }
}