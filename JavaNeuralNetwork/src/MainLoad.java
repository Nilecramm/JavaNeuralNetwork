import Network.NeuralNetwork;
import Network.ReLUActivation;
import Network.SigmoidActivation;
import Network.SoftmaxActivation;

public class MainLoad {
    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork("saves/xorfiles/xor");

        double[][] trainingInputs = {
                {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };
        double[][] trainingOutputs = {
                {0}, {1}, {1}, {0}
        };

        System.out.println("Prédictions :");
        for (double[] input : trainingInputs) {
            System.out.println(input[0] + " XOR " + input[1] + " = " + nn.forward(input)[0]);
        }
    }
}