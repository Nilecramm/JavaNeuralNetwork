package Network;

public class SigmoidActivation implements ActivationFunction {
    public double activate(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public double derivative(double x) {
        return x * (1 - x); // dérivée de sigmoid(x) = x * (1 - x)
    }
}