package Network;

public class LeakyReLUActivation implements ActivationFunction {
    public double activate(double x) {
        return (x > 0) ? x : 0.01 * x; // LeakyReLU
    }

    public double derivative(double x) {
        return (x > 0) ? 1 : 0;
    }
}