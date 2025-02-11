package Network;

public class ReLUActivation implements ActivationFunction {
    public double activate(double x) {
        return Math.max(0, x);
    }

    public double derivative(double x) {
        return (x > 0) ? 1 : 0;
    }
}