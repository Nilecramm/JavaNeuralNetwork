import Network.LeakyReLUActivation;
import Network.NeuralNetwork;
import Network.ReLUActivation;
import Network.SoftmaxActivation;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class MainMnistDiagnostic {
    private static class DataSet {
        double[][] inputs;
        double[][] outputs;

        DataSet(double[][] inputs, double[][] outputs) {
            this.inputs = inputs;
            this.outputs = outputs;
        }
    }

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
        DataSet trainData = loadData("data/MNIST/mnist_train.csv", 1000);
        DataSet testData = loadData("data/MNIST/mnist_test.csv", 100);

        // Smaller batch size for better stability
        int batchSize = 16;

        // Training with additional monitoring
        for (int epoch = 0; epoch < 5; epoch++) {
            System.out.println("\n=== Epoch " + epoch + " ===");

            // Shuffle data at start of each epoch
            //shuffleData(trainData);

            double epochLoss = 0;
            int batchCount = 0;

            for (int i = 0; i < trainData.inputs.length; i += batchSize) {
                int end = Math.min(i + batchSize, trainData.inputs.length);
                double[][] batchInputs = Arrays.copyOfRange(trainData.inputs, i, end);
                double[][] batchOutputs = Arrays.copyOfRange(trainData.outputs, i, end);

                // Check for NaN in inputs
                if (hasNaN(batchInputs)) {
                    System.out.println("Warning: NaN detected in inputs at batch " + (i/batchSize));
                    continue;
                }

                nn.train(batchInputs, batchOutputs, 1, false);

                if (i % (batchSize * 5) == 0) {
                    runDiagnostics(nn, batchInputs[0], batchOutputs[0]);

                    // Early stopping if we detect NaN
                    double[] output = nn.forward(batchInputs[0]);
                    if (hasNaN(output)) {
                        System.out.println("NaN detected in outputs - reducing learning rate");
                        nn.setLearningRate(nn.getLearningRate() * 0.1);
                        if (nn.getLearningRate() < 1e-6) {
                            System.out.println("Learning rate too small, stopping training");
                            return;
                        }
                        continue;
                    }
                }
            }

            double accuracy = evaluateAccuracy(nn, testData.inputs, testData.outputs);
            System.out.printf("Epoch %d Complete - Test Accuracy: %.2f%%\n", epoch, accuracy * 100);
        }
    }

    private static boolean hasNaN(double[] array) {
        for (double value : array) {
            if (Double.isNaN(value)) return true;
        }
        return false;
    }

    private static boolean hasNaN(double[][] array) {
        for (double[] row : array) {
            if (hasNaN(row)) return true;
        }
        return false;
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

    private static void runDiagnostics(NeuralNetwork nn, double[] input, double[] expectedOutput) {
        double[] output = nn.forward(input);

        // Print network output distribution
        System.out.println("Network outputs:");
        for (int i = 0; i < output.length; i++) {
            System.out.printf("Class %d: %.6f ", i, output[i]);
        }
        System.out.println();

        // Check if outputs sum to approximately 1 (for softmax)
        double sum = Arrays.stream(output).sum();
        System.out.printf("Output sum: %.6f\n", sum);

        // Print predicted vs expected class
        int predictedClass = getMaxIndex(output);
        int expectedClass = getMaxIndex(expectedOutput);
        System.out.printf("Predicted: %d, Expected: %d\n", predictedClass, expectedClass);

        // Check for dead neurons (all zeros) or saturated neurons (all ones)
        double outputMax = Arrays.stream(output).max().getAsDouble();
        double outputMin = Arrays.stream(output).min().getAsDouble();
        System.out.printf("Output range: [%.6f, %.6f]\n", outputMin, outputMax);
    }

    private static void evaluateDetailedAccuracy(NeuralNetwork nn, double[][] inputs, double[][] expectedOutputs) {
        int[] confusionMatrix = new int[10];
        int[] classCounts = new int[10];

        for (int i = 0; i < inputs.length; i++) {
            double[] output = nn.forward(inputs[i]);
            int predicted = getMaxIndex(output);
            int expected = getMaxIndex(expectedOutputs[i]);

            classCounts[expected]++;
            if (predicted == expected) {
                confusionMatrix[predicted]++;
            }
        }

        System.out.println("Per-class accuracy:");
        for (int i = 0; i < 10; i++) {
            if (classCounts[i] > 0) {
                double accuracy = (double) confusionMatrix[i] / classCounts[i] * 100;
                System.out.printf("Digit %d: %.2f%% (%d/%d)\n",
                        i, accuracy, confusionMatrix[i], classCounts[i]);
            }
        }
    }

    private static double evaluateAccuracy(NeuralNetwork nn, double[][] inputs, double[][] expectedOutputs) {
        int correct = 0;
        for (int i = 0; i < inputs.length; i++) {
            double[] output = nn.forward(inputs[i]);
            int predicted = getMaxIndex(output);
            int expected = getMaxIndex(expectedOutputs[i]);
            if (predicted == expected) {
                correct++;
            }
        }
        return (double) correct / inputs.length;
    }

    private static int getMaxIndex(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}