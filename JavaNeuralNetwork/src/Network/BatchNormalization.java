package Network;

public class BatchNormalization {
    private double epsilon = 1e-5;
    private double[] gamma;  // Scale parameter
    private double[] beta;   // Shift parameter
    private double[] mean;
    private double[] variance;
    private double momentum = 0.9;

    public BatchNormalization(int size) {
        gamma = new double[size];
        beta = new double[size];
        mean = new double[size];
        variance = new double[size];

        // Initialize gamma to 1 and beta to 0
        for (int i = 0; i < size; i++) {
            gamma[i] = 1.0;
            beta[i] = 0.0;
        }
    }

    public double[] normalize(double[] input, boolean training) {
        double[] output = new double[input.length];

        if (training) {
            // Calculate mean
            double batchMean = 0;
            for (double value : input) {
                batchMean += value;
            }
            batchMean /= input.length;

            // Calculate variance
            double batchVar = 0;
            for (double value : input) {
                batchVar += (value - batchMean) * (value - batchMean);
            }
            batchVar /= input.length;

            // Update running statistics
            for (int i = 0; i < input.length; i++) {
                mean[i] = momentum * mean[i] + (1 - momentum) * batchMean;
                variance[i] = momentum * variance[i] + (1 - momentum) * batchVar;
            }

            // Normalize
            for (int i = 0; i < input.length; i++) {
                output[i] = (input[i] - batchMean) / Math.sqrt(batchVar + epsilon);
                output[i] = gamma[i] * output[i] + beta[i];
            }
        } else {
            // Use running statistics for inference
            for (int i = 0; i < input.length; i++) {
                output[i] = (input[i] - mean[i]) / Math.sqrt(variance[i] + epsilon);
                output[i] = gamma[i] * output[i] + beta[i];
            }
        }

        return output;
    }
}