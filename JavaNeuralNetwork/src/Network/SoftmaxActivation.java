package Network;

import java.util.Arrays;

public class SoftmaxActivation implements ActivationFunction {
    private double[] lastOutput; // Pour stocker la sortie et éviter de la recalculer

    public double[] activate(double[] x) {
        double max = Arrays.stream(x).max().getAsDouble(); // Évite les overflows
        double sum = 0;
        double[] expValues = new double[x.length];

        for (int i = 0; i < x.length; i++) {
            expValues[i] = Math.exp(x[i] - max);
            sum += expValues[i];
        }

        for (int i = 0; i < x.length; i++) {
            expValues[i] /= sum;
        }

        lastOutput = expValues; // Sauvegarde pour la dérivée
        return expValues;
    }

    public double[] derivative(double[] x) {
        double[] gradients = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            gradients[i] = lastOutput[i] * (1 - lastOutput[i]); // Dérivée de softmax
        }
        return gradients;
    }

    // Implémentation non utilisée ici
    @Override
    public double activate(double x) {
        return 0;
    }

    @Override
    public double derivative(double x) {
        return 0;
    }
}