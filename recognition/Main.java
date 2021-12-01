package recognition;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Main {
    public static void main(String[] args) {
        String neuronFileName = "neuron.db";
//        File file = new File(neuronFileName);
//        System.out.println(file.getAbsolutePath());
        String dataFileName = "data.db";
        String dataFolderName = "data";
        String smallDataFolderName = "smalldata";

        Neuron neuron = new Neuron();
        if (ReadText.isExist(neuronFileName)) {
            try {
                StoreNeuron storeNeuron = (StoreNeuron) SerializationUtils.deserialize(neuronFileName);
                storeNeuron.toNeuron(neuron);
            } catch (IOException | ClassNotFoundException e) {
                System.out.println(e.getMessage());
            }
        }

        Scanner scanner = new Scanner(System.in);
        Action ac = new Action(scanner, neuronFileName, dataFileName, dataFolderName, smallDataFolderName, neuron);
        while(true) {
            System.out.println("1. Learn the network");
            System.out.println("2. Guess all the numbers");
            System.out.println("3. Guess number from text file");
            System.out.println("Your choice:");
            int menuno = Integer.parseInt(scanner.nextLine());
            switch (menuno) {
                case 1:
                    ac.bigLearn();
                    break;
                case 2:
                    ac.bigGuessAll();
                    break;
                case 3:
                    ac.guessFromText();
                    break;
                case 4:
                    ac.bigMakedata();
                    break;
                case 5:
                    ac.smallMakedata();
                    break;
                case 6:
                    ac.setEpoch();
                    break;
                case 7:
                    ac.setLearningRate();
                    break;
                case 8:
                    ac.smallLearn();
                    break;
                case 9:
                    ac.smallGuessAll();
                    break;
                case 10:
                    ac.continueLearn();
                    break;
                case 11:
                    ac.writeWeight();
                    break;
                case 12:
                    ac.matrixTest();
                    break;
                case 0:
                    break;
            }
            if (menuno == 3) {
                break;
            }
            if (menuno == 0) {
                break;
            }
        }
        scanner.close();
    }
}

class Action {
    Scanner scanner;
    Neuron neuron;
    String neuronFileName;
    String dataFileName;
    String originalDataFolderName;
    String smallDataFolderName;
    int[] layerSizes;
    String dataFolderName;

    Action(Scanner scanner, String neuronFileName, String dataFileName, String dataFolderName, String smallDataFolderName, Neuron neuron) {
        this.scanner = scanner;
        this.neuronFileName = neuronFileName;
        this.dataFileName = dataFileName;
        this.originalDataFolderName = dataFolderName;
        this.smallDataFolderName = smallDataFolderName;
        this.neuron = neuron;
        dataFolderName = originalDataFolderName;
    }

    List<LabelData> loadTrainingDataWrapper() {
        if (ReadText.isExist(dataFileName)) {
            try {
                StoreLabelData storeLabelData = (StoreLabelData) SerializationUtils.deserialize(dataFileName);
                List<LabelData> list = storeLabelData.toLabelData();
                return list;
            } catch (IOException | ClassNotFoundException e) {
                System.out.println(e.getMessage());
            }
        }

        return loadTrainingData();
    }

    List<LabelData> loadTrainingData() {
        List<LabelData> list = new ArrayList<>();
        File dir = new File(dataFolderName);
        String[] files = dir.list();
        for (int i = 0; i < files.length; i++) {
            String text = ReadText.readAll("data/" + files[i]);
            double[] array = Arrays.stream(text.split("\\s+")).mapToDouble(z -> Double.parseDouble(z)).toArray();   
            int label = (int) array[784];      
            double[] data = new double[784];
            IntStream.range(0, 784).forEach(z -> data[z] = array[z] / 255);
            list.add(new LabelData(label, new Vector(data))); 
        }

        StoreLabelData storeLabelData = new StoreLabelData(list);
        try {
            SerializationUtils.serialize(storeLabelData, dataFileName);
            System.out.println("Done! Training data Saved to the file.");
        } catch (IOException err) {
            System.out.println(err.getMessage());
        }

        return list;
    }

    void bigLearn() {
        dataFolderName = originalDataFolderName;
        learn();
    }

    void smallLearn() {
        dataFolderName = smallDataFolderName;
        learn();
    }

    void continueLearn() {
        System.out.println("Learning...");
        List<LabelData> list = loadTrainingDataWrapper();
        neuron.continueLearn(list);
        StoreNeuron storeNeuron = new StoreNeuron(neuron.getLayerSizes(), neuron.getWeight(), neuron.getBiases());
        try {
            SerializationUtils.serialize(storeNeuron, neuronFileName);
            System.out.println("Done! Saved to the file.");
        } catch (IOException err) {
            System.out.println(err.getMessage());
        }
    }

    void learn() {
   
        System.out.println("Enter the sizes of the layers:");
        // 15 12 12 10
        layerSizes = Arrays.asList(scanner.nextLine().split("\\s+")).stream().mapToInt(s -> Integer.parseInt(s)).toArray();
        System.out.println("Learning...");
        List<LabelData> list = loadTrainingDataWrapper();
        neuron.learn(layerSizes, list);
        StoreNeuron storeNeuron = new StoreNeuron(neuron.getLayerSizes(), neuron.getWeight(), neuron.getBiases());
        try {
            SerializationUtils.serialize(storeNeuron, neuronFileName);
            System.out.println("Done! Saved to the file.");
        } catch (IOException err) {
            System.out.println(err.getMessage());
        }
    }

    void guessFromText() {
        System.out.println("Enter filename:");
        String fileName = scanner.nextLine();
        String text = ReadText.readAll(fileName);
        double[] array = Arrays.stream(text.split("\\s+")).mapToDouble(z -> Double.parseDouble(z)).toArray();   
        double[] data = new double[784];
        IntStream.range(0, 784).forEach(z -> data[z] = array[z] / 255);
        System.out.println(String.format("This number is %d", neuron.getAnswer(data)));
    }

    void bigGuessAll() {
        dataFolderName = originalDataFolderName;
        guessAll();
    }

    void smallGuessAll() {
        dataFolderName = smallDataFolderName;
        guessAll();
    }

    void guessAll() {
        System.out.println("Guessing...");
        List<LabelData> list = loadTrainingDataWrapper();
        int accuracyCount = 0;
        for (int i = 0; i < list.size(); i++) {
            LabelData labelData = list.get(i);
            int label = labelData.label;
            Vector x = labelData.v;
            int y = neuron.getAnswer(x);
            if (y == label) {
                accuracyCount++;
            } 
        }
        int accuracyRate = accuracyCount * 100 / list.size();
        String accuracyRate_str = String.valueOf(accuracyRate);
        String accuracyRate_equation = String.valueOf(accuracyCount) + "/" + String.valueOf(list.size());
        System.out.println("The network prediction accuracy: " + accuracyRate_equation + ", " + accuracyRate_str + "%");
    } 

    void bigMakedata() {
        dataFolderName = originalDataFolderName;
        loadTrainingData(); 
    }

    void smallMakedata() {
        dataFolderName = smallDataFolderName;
        loadTrainingData(); 
    }

    void matrixTest() {
        double[] a = {1, 2, 3};
        double[] b = {4, 5};
        Vector v1 = new Vector(a);  //(1, 3)
        Vector v2 = new Vector(b);  //(1 ,2)
        Matrix m = new Matrix(v1, v2);  //(3, 2)
        Matrix mt = m.transport();
        Vector v3 = Vector.dot(v1, m);   //(1, 3) * (3, 2) = (1, 2)
        Vector v4 = m.dot(v2);
        System.out.println(mt);
        System.out.println(v3);
        System.out.println(v4);
    }

    void writeWeight() {
        int k = neuron.getEpoch();
        double eta = neuron.getLearningRate();
        String fileName = "w" + k + "-" + layerSizesToString() + "-" + eta + ".txt";
        Matrix[] w = neuron.getWeight();
        Vector[] b = neuron.getBiases();
        StringBuilder sb = new StringBuilder();
        for (int layer = 0; layer < w.length; layer++) {
            sb.append("weight layer " + layer + "\r\n");
            double[][] m = w[layer].get();
            for (int row = 0; row < m.length; row++) {
                for (int col = 0; col < m[row].length; col++) {
                    sb.append(m[row][col] + ",");
                }
                int len = sb.length();
                sb.delete(len - 1, len);    
                sb.append("\r\n");
            }
        }
        for (int layer = 0; layer < b.length; layer++) {
            sb.append("baias layer " + layer + "\r\n");
            double[] v = b[layer].get();
            for (int row = 0; row < v.length; row++) {
                    sb.append(v[row] + ",");
            }
            int len = sb.length();
            sb.delete(len - 1, len);
            sb.append("\r\n");
        }
        sb.append("y\r\n");
        List<LabelData> list = neuron.getLabelData();
        for (LabelData data: list) {
            Vector x = data.getVector();
            Vector y = neuron.getY(x);
            double[] v = y.get();
            for (int row = 0; row < v.length; row++) {
                    sb.append(v[row] + ",");
            }
            int len = sb.length();
            sb.delete(len - 1, len);
            sb.append("\r\n");
        }
    
        String text = sb.toString();
        WriteText.writeAll(fileName, text);
    }
    
    String layerSizesToString() {
        StringBuilder sb = new StringBuilder();
        IntStream.range(0, layerSizes.length).forEach(i -> sb.append(layerSizes[i] + "-"));
        int len = sb.length();
        sb.delete(len - 1, len);
        return sb.toString();
    }

    void setEpoch() {
        System.out.println("Enter Epoch:");
        int epoch = Integer.parseInt(scanner.nextLine());
        neuron.setEpoch(epoch);
    }

    void setLearningRate() {
        System.out.println("Enter learnig rate:");
        double learningRate = Double.parseDouble(scanner.nextLine());
        neuron.setLearningRate(learningRate);
    }
}

class StoreNeuron implements Serializable {
    private static final long serialVersionUID = 1L;

    int[] layerSizes;
    double[][][] w;
    double[][] b;

    StoreNeuron(int[] layerSizes, Matrix[] m, Vector[] v) {
        this.layerSizes = layerSizes;
        int layerSize = layerSizes.length;
        w = new double[layerSize - 1][][];
        for (int i = 0; i < layerSize - 1; i++) {
            w[i] = m[i].get();
        }
        b = new double[layerSize - 1][];
        for (int i = 0; i < layerSize - 1; i++) {
            b[i] = v[i].get();
        }
    } 

    void toNeuron(Neuron neuron) {
        neuron.setLayerSizes(layerSizes);
        int layerSize = layerSizes.length;
        Matrix[] m = new Matrix[layerSize - 1];
        for (int i = 0; i < layerSize - 1; i++) {
            m[i] = new Matrix(w[i]);
        }
        Vector[] v = new Vector[layerSize - 1];
        for (int i = 0; i < layerSize - 1; i++) {
            v[i] = new Vector(b[i]);
        }
        neuron.setWeights(m);
        neuron.setBiases(v);
    }
}

class Neuron {
    Learn learn;

    Neuron() {
        this.learn = new Learn();
    }

    int[] getLayerSizes() {
        return learn.getLayers();
    }

    Matrix[] getWeight() {
        return learn.getWeights();
    }

    Vector[] getBiases() {
        return learn.getBiases();
    }

    List<LabelData> getLabelData() {
        return learn.getLabelData();
    }

    int getEpoch() {
        return learn.getEpoch();
    }

    double getLearningRate() {
        return learn.getLearningRate();
    }

    void setLayerSizes(int[] layerSizes) {
        learn.setLayers(layerSizes);
    }
    
    void setWeights(Matrix[] w) {
        learn.setWeights(w);
    }

    void setBiases(Vector[] b) {
        learn.setBiases(b);
    }

    void setLabelData(List<LabelData> list) {
        learn.setLabelData(list);
    } 

    void setEpoch(int epoch) {
        learn.setEpoch(epoch);
    }

    void setLearningRate(double learningRate) {
        learn.setLearningRate(learningRate);
    }

    void learn(int[] layerSizes, List<LabelData> list) {
        this.learn = new Learn(layerSizes, list);
        learn.learn(false);
    }

    void learn() {
        learn.learn(false);
    }

    void continueLearn(List<LabelData> list) {
        learn.setLabelData(list);
        learn.learn(true);
    }
   
    int infer(Vector x) {
        Vector y = learn.forward(x);
        return Vector.maxIndex(y);
    }

    Vector getY(Vector x) {
        return learn.forward(x);
    }

    int getAnswer(double[] a) {
        Vector x = new Vector(a);
        return infer(x);
    }

    int getAnswer(Vector x) {
        return infer(x);
    }

    int getAnswer(int[] a) {
        Vector x = new Vector(a);
        return infer(x);
    }
}

class SerializationUtils {
    static void serialize(Object obj, String fileName) throws IOException {
        FileOutputStream fos = new FileOutputStream(fileName);
        BufferedOutputStream bos = new BufferedOutputStream(fos);
        ObjectOutputStream oos = new ObjectOutputStream(bos);
        oos.writeObject(obj);
        oos.close();
    }

    static Object deserialize(String fileName) throws IOException, ClassNotFoundException {
        FileInputStream fis = new FileInputStream(fileName);
        BufferedInputStream bis = new BufferedInputStream(fis);
        ObjectInputStream ois = new ObjectInputStream(bis);
        Object obj = ois.readObject();
        ois.close();
        return obj;
    }
}

class ReadText {

    static boolean isExist(String fileName) {
        File file = new File(fileName);
        if (file.exists()) {
            return true;
        } else {
            return false;
        }
    }

    static String getAbsolutePath(String fileName) {
        File file = new File(fileName);
        return file.getAbsolutePath();
    }

    static String readAllWithoutEol(String fileName) {
        String text = "";
        try {
            BufferedReader br = new BufferedReader(new FileReader(fileName));
            text =  br.lines().collect(Collectors.joining());
            br.close();
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }
        return text;
    }

    static List<String> readLines(String fileName) {
        List<String> lines = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(fileName));
            lines =  br.lines().collect(Collectors.toList());
            br.close();
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }
        return lines;
    }

    static String readAll(String fileName) {
        char[] cbuf = new char[4096];
        StringBuilder sb = new StringBuilder();
        try {
            BufferedReader br = new BufferedReader(new FileReader(fileName));
            while (true) {
                int length = br.read(cbuf, 0, cbuf.length);
                if (length != -1) {
                    sb.append(cbuf, 0, length);
                }
                if (length < cbuf.length) {
                    break;
                }
            }
            br.close();
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }
        return sb.toString();
    }
}

class WriteText {

    static void writeAll(String fileName, String text) {
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(fileName));
            bw.write(text, 0, text.length());
            bw.close();
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }
    }

    static void writeAll(String fileName, String text, String encoding) {
        try {
            BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(fileName), "UTF-8"));
            bw.write(text, 0, text.length());
            bw.close();
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }
    }
}

class Matrix {

    double[][] m;
    Random rand = new Random();

    Matrix(double[][] m) {
        this.m = m;
    }

    Matrix(double[][] m, boolean gaussian) {
        this.m = m;
        if (gaussian == true) {
            IntStream.range(0, m.length).forEach(i -> IntStream.range(0, m[i].length).forEach(j -> m[i][j] = rand.nextGaussian())); 
        }
    }

    Matrix(Vector row, Vector col) {
        double[] r = row.get();
        double[] c = col.get();
        m = new double[r.length][c.length];
        IntStream.range(0, m.length).forEach(i -> IntStream.range(0, m[i].length).forEach(j -> m[i][j] = r[i] * c[j])); 
    }

    int getRowSize() {
        return m.length;
    }

    int getColSize() {
        return m[0].length;
    }
 
    void allone() {
        IntStream.range(0, m.length).forEach(i -> IntStream.range(0, m[i].length).forEach(j -> m[i][j] = 1)); 
    }

    void allzero() {
        IntStream.range(0, m.length).forEach(i -> IntStream.range(0, m[i].length).forEach(j -> m[i][j] = 0)); 
    }

    double[][] get() {
        return m;
    }

    Matrix add(Matrix m2) {
        double[][] a = m2.get();
        IntStream.range(0, m.length).forEach(i -> IntStream.range(0, m[i].length).forEach(j -> m[i][j] += a[i][j])); 
        return this;
    }

    Matrix sub(Matrix m2) {
        double[][] a = m2.get();
        IntStream.range(0, m.length).forEach(i -> IntStream.range(0, m[i].length).forEach(j -> m[i][j] -= a[i][j])); 
        return this;
    }

    Matrix mul(double s) {
        IntStream.range(0, m.length).forEach(i -> IntStream.range(0, m[i].length).forEach(j -> m[i][j] *= s));
        return this; 
    }

    Matrix dot(double s) {
        IntStream.range(0, m.length).forEach(i -> IntStream.range(0, m[i].length).forEach(j -> m[i][j] *= s)); 
        return this;
    }

    Vector dot(Vector v) {
        return new Vector(IntStream.range(0, m.length).mapToDouble(i -> Vector.dot(v, new Vector(m[i]))).toArray());
    }

    Matrix transport() {
        int row = m.length;
        int col = m[0].length;
        double[][] a = new double[col][row];
        IntStream.range(0, row).forEach(i -> IntStream.range(0, col).forEach(j -> a[j][i] = m[i][j]));
        return new Matrix(a);
    }

    public String toString() {
        String string = "";
        for (int i = 0; i < m.length; i++) {
            List<String> list = new ArrayList<String>();
            double[] row = m[i]; 
            IntStream.range(0, row.length).forEach(j -> list.add(String.valueOf(row[j])));
            string += String.join(", ", list) + "\n";
        }
        return string;
    }
} 

class Vector {

    double[] v;
    Random rand = new Random(17);

    Vector(double[] v) {
        this.v = v;
    }

    Vector(double[] v, boolean gaussian) {
        this.v = v;
        if (gaussian == true) {
            IntStream.range(0, v.length).forEach(i -> v[i] = rand.nextGaussian()); 
        }
    }

    Vector(int[] a) {
        v = new double[a.length];
        IntStream.range(0, a.length).mapToDouble(i -> v[i] = a[i]).toArray();
    }

    Vector(Vector b) {
        v = b.get();
    }

    double[] get() {
        return v;
    }

    int[] getInt() {
        return IntStream.range(0, v.length).map(i -> (int) v[i]).toArray();
    }

    int length() {
        return v.length;
    }

    void allone() {
        IntStream.range(0, v.length).forEach(i -> v[i] = 1);
    }

    void allzero() {
        IntStream.range(0, v.length).forEach(i -> v[i] = 0);
    }

    Vector add(Vector v1) {
        double[] a = v1.get();
        IntStream.range(0, a.length).forEach(i -> v[i] += a[i]);  
        return this;  
    }

    Vector sub(Vector v1) {
        double[] a = v1.get();
        IntStream.range(0, a.length).forEach(i -> v[i] -= a[i]);   
        return this; 
    }

    Vector mul(double s) {
        IntStream.range(0, v.length).forEach(i -> v[i] *= s);    
        return this;
    }

    Vector mul(Vector v1) {
        double[] a = v1.get();
        IntStream.range(0, a.length).forEach(i -> v[i] *= a[i]);   
        return this; 
    }

    public String toString() {
        List<String> list = new ArrayList<String>();
        IntStream.range(0, v.length).forEach(i -> list.add(String.valueOf(v[i])));
        return String.join(", ", list);
    }

    static Vector add(Vector v1, Vector v2) {
        double[] a = v1.get();
        double[] b = v2.get();
        return new Vector(IntStream.range(0, a.length).mapToDouble(i -> a[i] + b[i]).toArray()); 
    }

    static Vector sub(Vector v1, Vector v2) {
        double[] a = v1.get();
        double[] b = v2.get();
        return new Vector(IntStream.range(0, a.length).mapToDouble(i -> a[i] - b[i]).toArray()); 
    }

    static double dot(Vector v1, Vector v2) {
        double[] a = v1.get();
        double[] b = v2.get();
        return IntStream.range(0, a.length).mapToDouble(i -> a[i] * b[i]).sum();
    }

    static Vector mul(double s, Vector v1) {
        double[] a = v1.get();
        return new Vector(IntStream.range(0, a.length).mapToDouble(i -> s * a[i]).toArray());
    }

    static Vector mul(Vector v1, Vector v2) {
        double[] a = v1.get();
        double[] b = v2.get();
        return new Vector(IntStream.range(0, a.length).mapToDouble(i -> a[i] * b[i]).toArray());
    }

    static Vector dot(Vector v1, Matrix m1) {
        double[] a = v1.get();
        double[][] b = m1.get();
        double[] c = new double[b[0].length];
        for (int j = 0; j < c.length; j++) {
            for (int i = 0; i < a.length; i++) {
                c[j] += a[i] * b[i][j];
            }
        }
        return new Vector(c);
    }

    static Vector sigmoid(Vector v1) {
        double[] a = v1.get();
        return new Vector(IntStream.range(0, a.length).mapToDouble(i -> sigmoid(a[i])).toArray());
    }

    static double sigmoid(double x) {
        return (1 / (1 + Math.exp(-x)));    
    }

    static Vector sigmoid_prime(Vector v1) {
        double[] a = v1.get();
        return new Vector(IntStream.range(0, a.length).mapToDouble(i -> sigmoid_prime(a[i])).toArray());
    }

    static double sigmoid_prime(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    static Vector relu(Vector v1) {
        double[] a = v1.get();
        return new Vector(IntStream.range(0, a.length).mapToDouble(i -> Math.max(0, a[i])).toArray());
    }

    static double relu(double x) {
        return Math.max(0, x);    
    }

    static Vector dRelu(Vector v1) {
        double[] a = v1.get();
        return new Vector(IntStream.range(0, a.length).mapToDouble(i -> a[i] > 0 ? 1 : 0).toArray());
    }

    static double dRelu(double x) {
        return x > 0 ? 1 : 0;
    }
    
    static Vector identity(Vector v1) {
        double[] a = v1.get();
        return new Vector(IntStream.range(0, a.length).mapToDouble(i -> a[i]).toArray());
    }

    static double identity(double x) {
        return x;    
    }

    static Vector dIdentity(Vector v1) {
        double[] a = v1.get();
        return new Vector(IntStream.range(0, a.length).mapToDouble(i -> 1).toArray());
    }

    static double dIdentity(double x) {
        return 1;
    }

    static Vector crossEntropy(Vector y, Vector t) {
        double[] a = y.get();
        double[] b = t.get();
        return new Vector(IntStream.range(0, a.length).mapToDouble(i -> crossEntropy(a[i], b[i])).toArray());
    }

    static double crossEntropy(double y, double t) {
        return t * Math.log(y) + (1 - t) * Math.log(1 - y);
    }

    static Vector dCrossEntropy(Vector y, Vector t, Vector z) {
        double[] a = y.get();
        double[] b = t.get();
        double[] c = z.get();
        return new Vector(IntStream.range(0, a.length).mapToDouble(i -> dCrossEntropy(a[i], b[i], c[i])).toArray());
    }

    static double dCrossEntropy(double y, double t, double z) {
        return  (y - t) / sigmoid_prime(z);       
    }

    static Vector label(int number, int size) {
        return new Vector(IntStream.range(0, size).mapToDouble(i -> i == number ? 1 : 0).toArray());
    }

    static Vector hadamard(Vector v1, Vector v2) {
        double[] a = v1.get();
        double[] b = v2.get();
        return new Vector(IntStream.range(0, a.length).mapToDouble(i -> a[i] * b[i]).toArray());
    }

    static Vector hadamard(Vector v1, Vector v2, Vector v3) {
        double[] a = v1.get();
        double[] b = v2.get();
        double[] c = v2.get();
        return new Vector(IntStream.range(0, a.length).mapToDouble(i -> a[i] * b[i] * c[i]).toArray());
    }

    static Vector allone(int size) {
        double[] a = new double[size];
        return new Vector(IntStream.range(0, a.length).mapToDouble(i -> 1).toArray());
    }

    static Vector allzero(int size) {
        double[] a = new double[size];
        return new Vector(IntStream.range(0, a.length).mapToDouble(i -> 0).toArray());
    }

    static Vector allone(Vector v1) {
        double[] a = v1.get();
        return new Vector(IntStream.range(0, a.length).mapToDouble(i -> 1).toArray());
    }

    static Vector allzero(Vector v1) {
        double[] a = v1.get();
        return new Vector(IntStream.range(0, a.length).mapToDouble(i -> 0).toArray());
    }

    static int maxIndex(Vector v1) {
        double[] a = v1.get();
        double max = IntStream.range(0, a.length).mapToDouble(i -> a[i]).max().getAsDouble();
        return IntStream.range(0, a.length).filter(i -> a[i] == max).findFirst().orElse(-1);
    }

    static void copy(Vector v1, Vector v2) {
        double[] a = v1.get();
        double[] b = v2.get();
        IntStream.range(0, v1.length()).forEach(i -> b[i] = a[i]);
    }
}

class StoreLabelData implements Serializable {
    private static final long serialVersionUID = 1L;

    int size;
    int[] labels;
    double[][] vs;

    StoreLabelData(List<LabelData> list) {
        size = list.size();
        labels = new int[this.size];
        vs = new double[size][784];
        for (int i = 0; i < size; i++) {
            LabelData labelData = list.get(i);
            double[] v = labelData.v.get();
            labels[i] = labelData.label;
            for (int j = 0; j < 784; j++) {
                vs[i][j] = v[j];
            }
        }
    } 

    List<LabelData> toLabelData() {
        List<LabelData> list = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            int label = labels[i];
            double[] v = vs[i];
            list.add(new LabelData(label, new Vector(v)));
        }
        return list;
    }
}

class LabelData {

    int label;
    Vector v;

    LabelData(int label, Vector v) {
        this.label = label;
        this.v = v;    
    }

    Vector getVector() {
        return v;
    }

    int getLabel() {
        return label;
    }
}

class Learn {
    String costfunction = "quadratic";
    int batchSize = 10; 
    double learningRate = 3.0;
    int epoch = 30;

    List<LabelData> trainingData;
    int[] layers;
    Matrix[] weights;
    Vector[] biases;
    Vector[] z;
    Vector[] a;
    Matrix[] nabla_w;
    Vector[] nabla_b;
    Matrix[] delta_w;
    Vector[] delta_b;
    Vector[] trainingInput;
    Vector[] trainingOutput;
    int[] labels;

    Learn() {}
    
    Learn(int[] layers, List<LabelData> trainingData) {
        this.layers = layers;
        this.trainingData = trainingData;
    }

    void init(int[] layers) {
        this.a = new Vector[layers.length];
        this.z = new Vector[layers.length - 1];
        this.nabla_w = new Matrix[layers.length - 1];
        this.nabla_b = new Vector[layers.length - 1];
        this.delta_w = new Matrix[layers.length - 1];
        this.delta_b = new Vector[layers.length - 1];
        for (int s = 0; s < layers.length - 1; s++) {
            this.a[s] = new Vector(new double[layers[s]]);
            this.z[s] = new Vector(new double[layers[s+1]]);
            this.nabla_w[s] = new Matrix(new double[layers[s+1]][layers[s]]);
            this.nabla_b[s] = new Vector(new double[layers[s+1]]);
            this.delta_w[s] = new Matrix(new double[layers[s+1]][layers[s]]);
            this.delta_b[s] = new Vector(new double[layers[s+1]]);
        }      
        int s = layers.length - 1;
        this.a[s] = new Vector(new double[layers[s]]);
    }

    void initWeightsBiases(int[] layers) {
        this.weights = new Matrix[layers.length - 1];
        this.biases = new Vector[layers.length - 1];
        for (int s = 0; s < layers.length - 1; s++) {
            this.weights[s] = new Matrix(new double[layers[s+1]][layers[s]], true);
            this.biases[s] = new Vector(new double[layers[s+1]], true);
        }      
    }

    void init(List<LabelData> trainingData) {
        this.trainingInput = new Vector[trainingData.size()];
        this.trainingOutput = new Vector[trainingData.size()];
        this.labels = new int[trainingData.size()];
        for (int i = 0; i < trainingData.size(); i++) {
            LabelData labelData = trainingData.get(i);
            this.trainingInput[i] = labelData.getVector();
            this.labels[i] = labelData.getLabel();
            this.trainingOutput[i] = vectorize(labelData.getLabel());
        }
    }

    void learn(boolean continueFlag) {
        if (continueFlag == false) {
            initWeightsBiases(layers);
        }
        init(layers);
        init(trainingData);
        for (int e = 0; e < epoch; e++) {
            System.out.println("epoch: " + (e+1) + " start");
            int batch_index = 0;
            for (int m = batch_index; m < trainingData.size() ; m += batchSize) {
                for (int s = 0; s < layers.length - 1; s++) {
                    this.delta_w[s] = new Matrix(new double[layers[s+1]][layers[s]]);
                    this.delta_b[s] = new Vector(new double[layers[s+1]]);
                }      
                for (int t = m; t < Math.min(m + batchSize, trainingData.size()); t++) {
                    Vector x = trainingInput[t];
                    Vector y = trainingOutput[t]; 
                    forward(x);
                    backward(y);
                }
                update();
            }
            int accuracyCount = evaluate();
            int accuracyRate = accuracyCount * 100 / trainingData.size();
            System.out.println("epoch:" + (e+1) + " " + accuracyCount + "/" + trainingData.size() + " " + accuracyRate + "%");
        }
    }

    Vector forward(Vector x) { 
        a[0] = x;
        for (int s = 0; s < layers.length - 1; s++) {
            z[s] = weights[s].dot(a[s]).add(biases[s]);
            a[s + 1] = Vector.sigmoid(z[s]);
        } 
        return a[layers.length - 1];
    }

    void backward(Vector y) {
        for (int s = layers.length - 2; s >= 0; s--) {
            if (s == layers.length - 2) {
                if (costfunction.equals("cross_entropy")) {
                    nabla_b[s] = Vector.sigmoid(z[s]).sub(y);
                } else {
                    nabla_b[s] = costDerivative(a[s + 1], y).mul(Vector.sigmoid_prime(z[s]));               
                }
            } else {
                nabla_b[s] = weights[s + 1].transport().dot(nabla_b[s + 1]).mul(Vector.sigmoid_prime(z[s]));                   
            }

            double[][] nw = nabla_w[s].get();
            double[] nb = nabla_b[s].get();
            double[] na = a[s].get();
            for (int i = 0; i < layers[s + 1]; i++) {
                for (int j = 0; j < layers[s]; j++) {
                    nw[i][j] = nb[i] * na[j];
                }
            }
            nabla_w[s] = new Matrix(nw);

            delta_b[s].add(nabla_b[s]);
            delta_w[s].add(nabla_w[s]); 
        }
    }

    void update() {

        for (int s = 0; s < layers.length - 1; s++) {
            weights[s].sub(delta_w[s].mul(learningRate / batchSize));
            biases[s].sub(delta_b[s].mul(learningRate / batchSize));
        }
    } 

    int evaluate() {
        int accuracyCount = 0;
        for (int t = 0; t < trainingData.size(); t++) {
            int label = trainingData.get(t).label;
            Vector x = trainingInput[t];
            Vector y = forward(x);
            if (Vector.maxIndex(y) == label) {
                accuracyCount++;
            }
        }
        return accuracyCount;
    }

    Vector costDerivative(Vector z, Vector y) {
        return Vector.sub(z, y);
    }

    Vector vectorize(int label) {
        double[] vlabel = new double[10];
        vlabel[label] = 1.0;
        return new Vector(vlabel);
    }

    int[] getLayers() {
        return this.layers;
    }

    Matrix[] getWeights() {
        return this.weights;
    }

    Vector[] getBiases() {
        return this.biases;
    }

    List<LabelData> getLabelData() {
        return this.trainingData;
    }

    int getEpoch() {
        return epoch;
    }

    double getLearningRate() {
        return learningRate;
    }
        
    void setLayers(int[] layers) {
        this.layers = layers;
        init(layers);
    }

    void setWeights(Matrix[] weights) {
        this.weights = weights;
    }

    void setBiases(Vector[] biases) {
        this.biases = biases;
    }

    void setLableData(List<LabelData> trainingData) {
        this.trainingData = trainingData;
    }

    void setEpoch(int epoch) {
        this.epoch = epoch;
    }

    void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    void setLabelData(List<LabelData> trainingData) {
        this.trainingData = trainingData;
        init(trainingData);
    }
}