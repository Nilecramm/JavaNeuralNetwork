import Network.NeuralNetwork;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;

public class MainMnistLoadImage {
    public static void main(String[] args) {
        // Charger le modèle
        NeuralNetwork nn = new NeuralNetwork("saves/mnistfiles/mnist8");

        // Charger et tester une image
        String imagePath = "data/test1.png";
        int predictedLabel = predictFromImage(nn, imagePath);
        System.out.println("Prédiction pour l'image : " + predictedLabel);
    }

    /**
     * Charge une image 28x28, la convertit en tableau normalisé et retourne la prédiction du réseau.
     */
    public static int predictFromImage(NeuralNetwork nn, String imagePath) {
        try {
            // Charger l'image en niveaux de gris
            BufferedImage image = ImageIO.read(new File(imagePath));

            if (image.getWidth() != 28 || image.getHeight() != 28) {
                image = convertTo28x28(imagePath);
            }

            double[] input = new double[784]; // 28x28 = 784
            for (int y = 0; y < 28; y++) {
                for (int x = 0; x < 28; x++) {
                    int color = image.getRGB(x, y);
                    if(color != 0)
                        color = 1;
                    else
                        color = 0;
                    input[y * 28 + x] = color;
                }
            }
            drawImageInConsole(input);

            // Faire la prédiction
            double[] prediction = nn.forward(input);
            return getMaxIndex(prediction);

        } catch (IOException e) {
            System.out.println("Erreur lors du chargement de l'image !");
            e.printStackTrace();
            return -1;
        }
    }

    /**
     * Retourne l'index du plus grand élément du tableau (utilisé pour obtenir la classe prédite).
     */
    private static int getMaxIndex(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static BufferedImage convertTo28x28(String imagePath) {
        try {
            BufferedImage original = ImageIO.read(new File(imagePath));
            BufferedImage resized = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
            Graphics2D g = resized.createGraphics();
            g.drawImage(original, 0, 0, 28, 28, null);
            g.dispose();
            return resized;
        } catch (IOException e) {
            System.out.println("Erreur : Impossible de charger l'image !");
            e.printStackTrace();
            return null;
        }
    }

    public static void drawImageInConsole(double[] input) {
        for (int i = 0; i < 784; i++) {
            System.out.print(input[i] == 1 ? 'x' : ' '); // Seulement X pour les pixels noirs
            if ((i + 1) % 28 == 0) {
                System.out.println();
            }
        }
    }
}