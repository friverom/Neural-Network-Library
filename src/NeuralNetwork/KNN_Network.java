package NeuralNetwork;

import Matrix.Matrix;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

/**
 * This class implements K nearest neighbor algorithm Inputs are organize in a
 * List of Matrix and Labels as well
 *
 * @author Federico
 */
public class KNN_Network implements Serializable {

    private String name;
    private List<Matrix> train_inputs;
    private List<Matrix> train_targets;
    private int k_factor; //K nearest factor
    private int target_matrix_size;

    /**
     * This constructor instantiates a KNN object
     *
     * @param train_inputs List<Matrix> of inputs
     * @param train_targets List<Matrix> of targets
     * @param k_factor K factor
     */
    public KNN_Network(String name, List<Matrix> train_inputs, List<Matrix> train_targets, int k_factor) throws FileNotFoundException, IOException {
        //Copy training data into local training lists.
        this.train_inputs = new ArrayList<>(train_inputs);
        Collections.copy(this.train_inputs, train_inputs);
        this.train_targets = new ArrayList<>(train_targets);
        Collections.copy(this.train_targets, train_targets);
        this.k_factor = k_factor;
        //Save the KNN object
        this.name = name + ".bin";
        target_matrix_size = ((int) getMax(this.train_targets) - (int) getMin(this.train_targets)) + 1;
        File file = new File(this.name);
        FileOutputStream fos = new FileOutputStream(file);
        ObjectOutputStream oos = new ObjectOutputStream(fos);

        oos.writeObject(this);
        fos.flush();
        fos.close();

    }

    /**
     * This method makes a prediction of the input variable according to the KNN
     * algorithm. The result is a double number representing the predicted
     * label.
     *
     * @param input Matrix input
     * @return label value
     */
    public double makeGuess(Matrix input) {

        List<Euclidian_distance> results = new ArrayList<>();

        Iterator it_inputs = train_inputs.iterator();
        Iterator it_targets = train_targets.iterator();
        //Compute euclidian distance of the input vector to the training list
        while (it_inputs.hasNext()) {
            Matrix in = (Matrix) it_inputs.next();
            Matrix tg = (Matrix) it_targets.next();
            double sum = 0;
            for (int i = 0; i < in.getRows(); i++) {
                sum += Math.pow(in.getValue(i + 1, 1) - input.getValue(i + 1, 1), 2);
            }
            //add distance to list and save targets
            Euclidian_distance dist = new Euclidian_distance(Math.sqrt(sum), tg.getValue(1, 1));
            results.add(dist);
        }
        //Sort distance list and targets
        Collections.sort(results, new Comparator<Euclidian_distance>() {
            @Override
            public int compare(Euclidian_distance o1, Euclidian_distance o2) {
                return Double.compare(o1.distance, o2.distance);
            }
        });
        //Select the nearest neighbors
        List<Euclidian_distance> list = new ArrayList<>();
        list = results.subList(0, k_factor);
        //Select the mode of the list
        int maxCount = 0;
        double label = 0;
        for (Euclidian_distance d : list) {
            int count = 0;
            for (int i = 0; i < list.size(); i++) {
                if (compare(d.target, list.get(i).target) == 1) {
                    count++;
                }
            }
            if (count > maxCount) {
                label = d.target;
                maxCount = count;
            }
        }
        return label;
    }

    //This method compares two double values
    //returns 1 if a>b
    //return 0 if a<b
    private int compare(double a, double b) {
        double c = a - b;
        int result;

        if (Math.abs(c) < 0.0001) {
            return 1;
        } else {
            return 0;
        }
    }

    /**
     * This method takes known inputs and targets and evaluates the performance
     * of the KNN network. The results are return as a matrix representing the
     * confusion matrix.
     *
     * @param inputs
     * @param targets
     * @return Confusion Matrix
     */
    public Matrix evalList(List<Matrix> inputs, List<Matrix> targets) {
        //Compute the size of the confusion matrix
        int matrixSize = ((int) getMax(targets) - (int) getMin(targets)) + 1;
        Matrix result = new Matrix(matrixSize, matrixSize);

        Iterator it_inputs = inputs.iterator();
        Iterator it_targets = targets.iterator();
        //Evaluate inputs
        while (it_inputs.hasNext()) {
            Matrix in = (Matrix) it_inputs.next();
            Matrix tg = (Matrix) it_targets.next();
            double guess = makeGuess(in);
            //actualize the confusion matrix
            int row = (int) guess;
            int col = (int) tg.getValue(1, 1);
            result.setValue(row, col, result.getValue(row, col) + 1);
        }
        return result;
    }

    //This method vectorize label (double) value to a matrix type.
    //if label values are 1,2,3 the vector will be of size 3.
    //if value is 1, matrix will be [1,0,0]. if label is 2, vector will be
    //[0,1,0]
    public Matrix toMatrix(double value) {
        //get the size of vector
        Matrix matrix = new Matrix(target_matrix_size, 1);
        double minValue = getMin(train_targets);
        //Set corresponding row according to label value
        matrix.setValue((int) (value - minValue + 1), 1, 1);

        return matrix;
    }

    //This method return the minimum value of labels.
    //Input is a Matrix 1,1 and label is a double value
    private double getMin(List<Matrix> m) {

        double min = 10;
        int label = 0;
        int index = 0;
        Iterator it_m = m.iterator();

        while (it_m.hasNext()) {
            Matrix tg = (Matrix) it_m.next();
            double value = tg.getValue(1, 1);
            if (value < min) {
                min = value;
                label = index;
            }
            index++;
        }
        //    System.out.println("Label: "+label);
        return min;

    }

    //This method return the maximum value of labels.
    //Input is a Matrix 1,1 and label is a double value
    private double getMax(List<Matrix> m) {

        double max = 0;
        int label = 0;
        int index = 0;
        Iterator it_m = m.iterator();

        while (it_m.hasNext()) {
            Matrix tg = (Matrix) it_m.next();
            double value = tg.getValue(1, 1);
            if (value > max) {
                max = value;
                label = index;
            }
            index++;
        }
        System.out.println("Label: " + label);
        return max;

    }

    //This class holds the Euclidian distance and corresponding target
    //for finding the nearest points
    private class Euclidian_distance {

        int index = 0;
        double distance;
        double target;

        private Euclidian_distance(double distance, double target) {
            this.distance = distance;
            this.target = target;
        }

    }
}
