package Network;

import java.util.Arrays;

public class SoftmaxActivation implements ActivationFunction {
    public double[] activate(double[] inputs) {
        double maxInput = Arrays.stream(inputs).max().orElse(0); // Prevent large exponentials
        double[] expValues = new double[inputs.length];
        double sum = 0;

        for (int i = 0; i < inputs.length; i++) {
            expValues[i] = Math.exp(inputs[i] - maxInput); // Subtract maxInput
            sum += expValues[i];
        }

        for (int i = 0; i < inputs.length; i++) {
            expValues[i] /= sum;
        }

        return expValues;
    }

    @Override
    public double activate(double x) {
        throw new UnsupportedOperationException("Softmax requires array input");
    }

    @Override
    public double derivative(double output) {
        return output * (1 - output);
    }
}