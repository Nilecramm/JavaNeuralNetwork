import Network.NeuralNetwork;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MainMnistLoad {
    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork("saves/mnistfiles/mnist8");

        String csvFile = "data/MNIST/mnist_test.csv";
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
        double[][] testingInputs = inputs.toArray(new double[0][]);
        double[][] testingOutputs = outputs.toArray(new double[0][]);

        //delete les listes
        inputs = null;
        outputs = null;

        System.out.println("Prédictions :");
        int correct = 0;
        for (int i = 0; i < testingInputs.length; i++) {
            double[] input = testingInputs[i];
            double[] output = testingOutputs[i];

            double[] prediction = nn.forward(input);
            int predictedLabel = 0;
            for (int j = 1; j < prediction.length; j++) {
                if (prediction[j] > prediction[predictedLabel]) {
                    predictedLabel = j;
                }
            }
            int trueLabel = 0;
            for (int j = 1; j < output.length; j++) {
                if (output[j] > output[trueLabel]) {
                    trueLabel = j;
                }
            }

            if (predictedLabel == trueLabel) {
                correct++;
            }
            System.out.println("Prediction: " + predictedLabel + " - True: " + trueLabel);
        }
        System.out.println("Accuracy: " + (double) correct / testingInputs.length);
    }
}
