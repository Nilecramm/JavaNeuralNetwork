import Network.NeuralNetwork;
import Network.ReLUActivation;
import Network.SigmoidActivation;
import Network.SoftmaxActivation;

public class Main {
    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork();

        nn.addLayer(2, new SigmoidActivation());
        //nn.addLayer(4, new SigmoidActivation());
        nn.addLayer(1, new SigmoidActivation());

        double[][] trainingInputs = {
                {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };
        double[][] trainingOutputs = {
                {0}, {1}, {1}, {0}
        };

        nn.train(trainingInputs, trainingOutputs, 100000, false);

        nn.save("saves/xorfiles/xor");
        System.out.println("Prédictions :");
        for (double[] input : trainingInputs) {
            System.out.println(input[0] + " XOR " + input[1] + " = " + nn.forward(input)[0]);
        }
    }
}