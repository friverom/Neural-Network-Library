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
 * This class implements K nearest neighbor algorithm
 * @author Federico
 */
public class KNN_Network implements Serializable{
    
    String name;
    private List<Matrix> train_inputs;
    private List<Matrix> train_targets;
    private int k_factor; //K nearest factor
    
    /**
     * 
     * @param train_inputs List<Matrix> of inputs
     * @param train_targets List<Matrix> of targets
     * @param k_factor K factor
     */
    public KNN_Network(String name,List<Matrix> train_inputs, List<Matrix> train_targets, int k_factor) throws FileNotFoundException, IOException{
        //Copy training data into local training lists.
        this.train_inputs=new ArrayList<>(train_inputs);
        Collections.copy(this.train_inputs, train_inputs);
        this.train_targets=new ArrayList<>(train_targets);
        Collections.copy(this.train_targets, train_targets);
        this.k_factor=k_factor;
        this.name=name+".bin";
        File file=new File(this.name);
        FileOutputStream fos=new FileOutputStream(file);
        ObjectOutputStream oos=new ObjectOutputStream(fos);
        
        oos.writeObject(this);
        fos.flush();
        fos.close();
        
    }
    /**
     * 
     * @param input Matrix input
     * @return label value
     */
    public double makeGuess(Matrix input){
        
        List<Euclidian_distance> results=new ArrayList<>();
        
        Iterator it_inputs=train_inputs.iterator();
        Iterator it_targets=train_targets.iterator();
        //Compute euclidian distance to train data points
        while(it_inputs.hasNext()){
            Matrix in=(Matrix)it_inputs.next();
            Matrix tg=(Matrix)it_targets.next();
            double sum=0;
            for(int i=0;i<in.getRows();i++){
                sum+=Math.pow(in.getValue(i+1, 1)-input.getValue(i+1, 1), 2);
            }
            //add distance to list and save targets
            Euclidian_distance dist=new Euclidian_distance(Math.sqrt(sum),tg.getValue(1,1));
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
        List<Euclidian_distance> list=new ArrayList<>();
        list=results.subList(0, k_factor);
        
        int maxCount=0;
        double label=0;
        for(Euclidian_distance d:list){
            int count=0;
            for(int i=0;i<list.size();i++){
                if(compare(d.target,list.get(i).target)==1){
                    count++;
                }
            }
            if(count>maxCount){
                label=d.target;
                maxCount=count;
            }
        }
        return label;
        }
    public Matrix evalList(List<Matrix> inputs, List<Matrix> targets){
        int matrixSize=3;
        Matrix result=new Matrix(3,3);
        
        Iterator it_inputs=inputs.iterator();
        Iterator it_targets=targets.iterator();
        
        while(it_inputs.hasNext()){
            Matrix in=(Matrix)it_inputs.next();
            Matrix tg=(Matrix)it_targets.next();
            double guess=makeGuess(in);
            int row=(int)guess;
            int col=(int)tg.getValue(1, 1);
            result.setValue(row, col, result.getValue(row, col)+1);
        }
        return result;
    }
    private int compare(double a, double b){
        double c=a-b;
        int result;
        
        if(Math.abs(c)<0.0001){
            return 1;
        }else{
            return 0;
        }
    }
    private class Euclidian_distance {
        int index=0;
        double distance;
        double target;
        
        private Euclidian_distance(double distance, double target){
            this.distance=distance;
            this.target=target;
        }

        
    }
}
