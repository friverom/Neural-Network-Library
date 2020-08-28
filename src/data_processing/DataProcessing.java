
package data_processing;

import Matrix.Matrix;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 * This class implements several methods to process and manipulate data files
 *
 * @author Federico
 */
public class DataProcessing {

    /**
     * This method takes a CSV file and creates a training data files and
     * testing data files. Test data files contains the last 20% of the list.
     * Training data file will be name as "filename"_train.txt Test data file
     * will be name as "filename"_test.txt
     *
     * @param filename CSV file containing the data
     * @throws FileNotFoundException
     * @throws IOException
     */
    public static void createFiles(String filename) throws FileNotFoundException, IOException {

        List<String> sourceFile = readLines(filename);
        List<String> train_data = new ArrayList<>();
        List<String> test_data = new ArrayList<>();

        int listSize = (int) (sourceFile.size() * 0.8);

        for (int i = 0; i < listSize; i++) {
            train_data.add(sourceFile.get(i));
        }

        for (int i = listSize; i < sourceFile.size(); i++) {
            test_data.add(sourceFile.get(i));
        }

        writeLines(filename + "_train", train_data);
        writeLines(filename + "_test", test_data);

    }

    /**
     * This method takes a CSV file and selects the numerical values of labels
     * and convert them to vectors
     *
     * @param filename CSV file containing the data
     * @param startCol Colum number of labels
     * @throws IOException
     */
    public static void vectorizeLabels(String filename, int startCol) throws IOException {

        List<String> list = new ArrayList<>();
        List<Integer> labels = toInteger(getColData(filename, startCol));
        int maxValue = getMaxInt(labels);
        int minValue = getMinInt(labels);

        for (Integer s : labels) {
            String str = "";
            for (int i = 0; i < (maxValue - minValue + 1); i++) {
                if (s == i) {
                    str += "1";
                } else {
                    str += "0";
                }
                if (i != maxValue - minValue) {
                    str += ",";
                }
            }
            list.add(str);
        }
        replaceCol(filename, list, startCol);
    }

    /**
     * This method takes a CSV file and randomize the lines
     *
     * @param filename CSV file with data
     * @throws IOException
     */
    public static void randomize(String filename) throws IOException {
        Random random = new Random();

        List<String> lines = readLines(filename);

        for (int i = 0; i < lines.size(); i++) {
            int rand = random.nextInt(lines.size());
            String st1 = lines.get(i);
            String temp = st1;
            String st2 = lines.get(rand);
            lines.set(i, st2);
            lines.set(rand, temp);
        }

        writeLines(filename, lines);

    }

    /**
     * This method takes a CSV file and extract the labels. Those labels are
     * then categorized with numeric values and write back to the CSV file.
     *
     * @param filename
     * @param startCol Labels field
     * @param label starting numerical value for labels
     * @throws IOException
     */
    public static void setLabels(String filename, int startCol, int label) throws IOException {

        List<String> list = getLabels(filename, startCol);
        List<String> labels = getColData(filename, startCol);
        List<String> newLabels = new ArrayList<>();

        for (String s : labels) {
            newLabels.add(String.valueOf(getLabelIndex(list, s, label)));
        }

        replaceCol(filename, newLabels, startCol);
    }

    /**
     * This method takes a CSV file an normalize the values of each field
     *
     * @param filename
     * @param start starting field
     * @param end end field
     * @throws FileNotFoundException
     * @throws IOException
     */
    public static void normalize(String filename, int start, int end) throws FileNotFoundException, IOException {

        for (int i = 0; i < (end - start) + 1; i++) {
            List<String> list = getColData(filename, start + i);
            List<String> newList = toString(norm(toDouble(list)));
            replaceCol(filename, newList, start + i);
        }
    }

    /**
     * This method reads a CSV file and returns as a String all lines in the
     * file
     *
     * @param filename
     * @return
     * @throws FileNotFoundException
     * @throws IOException
     */
    private static List<String> readLines(String filename) throws FileNotFoundException, IOException {

        File file = new File(filename + ".txt");
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);

        List<String> lines = new ArrayList<>();
        String line;
        while ((line = br.readLine()) != null) {
            lines.add(line);
        }
        return lines;
    }

    /**
     * This method takes a String and writes a file
     *
     * @param filename
     * @param lines
     * @throws IOException
     */
    private static void writeLines(String filename, List<String> lines) throws IOException {

        File file = new File(filename + ".txt");
        FileWriter fw = new FileWriter(file);
        BufferedWriter bw = new BufferedWriter(fw);

        for (String l : lines) {
            bw.write(l + "\n");
        }

        bw.flush();
        bw.close();
        fw.close();
    }

    /**
     * This method gets the labels list and returns an int with the number of
     * different labels on the list
     *
     * @param list
     * @param label
     * @param startNum
     * @return
     */
    private static int getLabelIndex(List<String> list, String label, int startNum) {

        int index = startNum;
        for (String s : list) {
            if (!s.equals(label)) {
                index++;
            } else {
                break;
            }
        }
        return index;
    }

    /**
     * This method takes a CSV file and extract the label field indicated.
     *
     * @param filename
     * @param startCol Labels Field
     * @return List of labels
     * @throws FileNotFoundException
     * @throws IOException
     */
    private static List<String> getLabels(String filename, int startCol) throws FileNotFoundException, IOException {

        FileReader fr = new FileReader(filename + ".txt");
        BufferedReader br = new BufferedReader(fr);

        String line;
        String label = "";
        List<String> list = new ArrayList<>();

        while ((line = br.readLine()) != null) {
            line = line.replace(" ", "");
            String items[] = line.split(",");
            String item = items[startCol - 1];

            if (!list.contains(item)) {
                list.add(item);
            }
        }
        br.close();
        fr.close();

        return list;
    }

    /**
     * This method replace the field column of data with the List
     *
     * @param filename
     * @param list
     * @param start Field column to replace
     * @throws FileNotFoundException
     * @throws IOException
     */
    private static void replaceCol(String filename, List<String> list, int start) throws FileNotFoundException, IOException {

        File sourceFile = new File(filename + ".txt");
        FileReader fr = new FileReader(sourceFile);
        BufferedReader br = new BufferedReader(fr);

        File tempFile = new File(filename + ".tmp");
        FileWriter fw = new FileWriter(tempFile);
        BufferedWriter bw = new BufferedWriter(fw);

        Iterator it = list.iterator();
        String line;

        while ((line = br.readLine()) != null) {
            line = line.replace(" ", "");
            String[] items = line.split(",");
            items[start - 1] = (String) it.next();
            String l = "";
            for (int i = 0; i < items.length; i++) {
                l += items[i];
                if (i != (items.length - 1)) {
                    l += ",";
                } else {
                    l += "\n";
                }
            }
            bw.write(l);
        }
        bw.flush();
        bw.close();
        br.close();

        sourceFile.delete();
        new File("iris.tmp").renameTo(new File("iris.txt"));

    }

    /**
     * This method returns a Column field indicated
     *
     * @param name
     * @param start Field column number to extract
     * @return
     * @throws FileNotFoundException
     * @throws IOException
     */
    private static List<String> getColData(String name, int start) throws FileNotFoundException, IOException {

        File file = new File(name + ".txt");
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);

        String data;
        List<String> list = new ArrayList<>();

        while ((data = br.readLine()) != null) {
            data = data.replace(" ", "");
            String[] items = data.split(",");
            list.add(items[start - 1]);
        }

        br.close();
        fr.close();
        return list;
    }

    /**
     * This method converts a list of doubles to String values
     *
     * @param list
     * @return
     */
    private static List<String> toString(List<Double> list) {

        List<String> newList = new ArrayList<>();

        for (double n : list) {
            newList.add(String.valueOf(n));
        }
        return newList;
    }

    /**
     * This method converts a List of Integers to String values
     *
     * @param list
     * @return
     */
    private static List<Integer> toInteger(List<String> list) {
        List<Integer> newList = new ArrayList<>();

        for (String s : list) {
            int val = Integer.parseInt(s);
            newList.add(val);
        }
        return newList;
    }

    /**
     * This method converts a list of String to double values
     *
     * @param list
     * @return
     */
    private static List<Double> toDouble(List<String> list) {

        List<Double> newlist = new ArrayList<>();

        for (String s : list) {
            double val = Double.parseDouble(s);
            newlist.add(val);
        }

        return newlist;
    }

    /**
     * This method computes normalization of a list of values
     *
     * @param list
     * @return
     */
    private static List<Double> norm(List<Double> list) {

        List<Double> norm = new ArrayList<>();
        double max = getMax(list);
        double min = getMin(list);
        double range = max - min;

        for (double l : list) {
            norm.add((l - min) / range);
        }
        return norm;
    }

    /**
     * This method returns the max Value in the list
     *
     * @param list
     * @return
     */
    private static double getMax(List<Double> list) {

        double max = 0;
        for (double l : list) {
            if ((max - l) < 0) {
                max = l;
            }
        }
        return max;
    }

    /**
     * This method returns the max value in the list
     *
     * @param list
     * @return
     */
    private static int getMaxInt(List<Integer> list) {

        int max = 0;
        for (Integer l : list) {
            if ((max - l) < 0) {
                max = l;
            }
        }
        return max;
    }

    /**
     * This method returns the min value of the list
     *
     * @param list
     * @return
     */
    private static double getMin(List<Double> list) {

        double min = 1000;
        for (double l : list) {
            if ((l - min) < 0) {
                min = l;
            }
        }
        return min;
    }

    /**
     * This method returns the min value of the list
     *
     * @param list
     * @return
     */
    private static int getMinInt(List<Integer> list) {

        int min = 1000;
        for (Integer l : list) {
            if ((l - min) < 0) {
                min = l;
            }
        }
        return min;
    }

    /**
     * This method converts a CSV file and creates a List of vectors with the
     * values fields indicated on the starting and ending columns
     *
     * @param filename
     * @param startCol
     * @param endCol
     * @return
     * @throws FileNotFoundException
     * @throws IOException
     * @throws ClassNotFoundException
     */
    private static List<Matrix> loadData(String filename, int startCol, int endCol) throws FileNotFoundException, IOException, ClassNotFoundException {

        List<Matrix> list = new ArrayList<>();
        File file = new File(filename);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);

        String data;
        while ((data = br.readLine()) != null) {
            Matrix m = createMatrix(data, startCol, endCol);
            list.add(m);
        }

        return list;
    }

    /**
     * This method creates a vector of values
     *
     * @param data
     * @param start
     * @param end
     * @return
     */
    private static Matrix createMatrix(String data, int start, int end) {

        String[] items = data.split(",");
        Matrix matrix = new Matrix(end - start + 1, 1);

        if (items.length > 1) {
            for (int i = 0; i < matrix.getRows(); i++) {
                double val = Double.parseDouble(items[(start - 1) + i]);
                matrix.setValue(i + 1, 1, val);
            }
        }
        return matrix;
    }

    /**
     * This method save to file a List of vectors
     *
     * @param name
     * @param m
     * @throws FileNotFoundException
     * @throws IOException
     */
    private static void createFile(String name, List<Matrix> m) throws FileNotFoundException, IOException {
        String filename = name + ".bin";
        File file = new File(filename);
        FileOutputStream fos = new FileOutputStream(file);
        ObjectOutputStream oos = new ObjectOutputStream(fos);
        oos.writeObject(m);
        fos.flush();
        fos.close();
    }

}
