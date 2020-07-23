
package NeuralNetwork;

import Matrix.Matrix;
import java.io.Serializable;
import NeuralNetwork.EnumValues;
import NeuralNetwork.EnumValues.ActivationFunction;
import NeuralNetwork.EnumValues.CostFunction;
import NeuralNetwork.EnumValues.InitializeMethod;
import NeuralNetwork.EnumValues.CostFunction;
import NeuralNetwork.EnumValues.GradientDescent;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Iterator;
import java.util.List;


/**
 *
 * @author Federico
 */
public class NeuralNetwork implements Serializable{
    //This array contains the layers of the network.
    //index [0] is the first hidden layer and the last index is the output layer
    private NN_Layer[] layer;
    private String name;
    private double learning_rate=0.05; //learning rate common to all layers
    private double lambda=0; //Lambda factor for L2 reguralization of weights
    private double n_factor=0; //Nesterov momemtum factor
    private int index=0; //To keep track of initialize layers
    
    private static long serialVersionUID=2L;
    
    
    /**
     * Constructor to setup a Neural Network.
     * Creates a array of layer 
     * @param num_hidden_layers Number of hidden layers including the outputs
     * layer
     */
    public NeuralNetwork(String name,int in,int num_hidden_layers,double lr, double lambda){
        //Set up an array of layers to build the network
        //Last layer is the output layer
        layer=new NN_Layer[num_hidden_layers];
        this.name=name;
        this.learning_rate=lr;
        this.lambda=lambda;
        
    }
    /**
     * Layer Builder Method. 
     * @param layerNum Layer Number in the network.(1..n)
     * @param Inputs Specify the number of inputs to this layers
     * @param Neurons How many nodes contained in this layer 
     * @param act   Specify the Activation Function
     * @param init  Specify initialization Method for Matrices
     * @param grd Gradient Descent Method
     */
    public void layerBuilder(int Inputs, int Neurons, ActivationFunction act, InitializeMethod init,GradientDescent grd){
        if(index<layer.length){
            layer[index++]=new NN_Layer(Inputs,Neurons,act,init,learning_rate,lambda,grd);
        }else{
            throw new RuntimeException("Layer number out of Bounds");
        }
    }
    /**
     * Layer Builder Method. 
     * @param layerNum Layer Number in the network.(1..n)
     * @param Inputs Specify the number of inputs to this layers
     * @param Neurons How many nodes contained in this layer 
     * @param act   Specify the Activation Function
     * @param init  Specify initialization Method for Matrices
     * @param grd Gradient Descent Method
     * @param factor Nesterov momemtum factor
     */
    public void layerBuilder(int Inputs, int Neurons, ActivationFunction act, InitializeMethod init,GradientDescent grd, double factor){
        if(index<layer.length){
            layer[index++]=new NN_Layer(Inputs,Neurons,act,init,learning_rate,lambda,grd,factor);
        }else{
            throw new RuntimeException("Layer number out of Bounds");
        }
    }
    /**
     * Layer Builder Method.
     * Activation Function will be the Sigmoid, Initialization method for
     * matrices will be Gaussian Random with mean = 0 and standard deviation
     * = 1. Learning rate will be 0.05
     * @param layerNum Layer Number in the network.(1..n)
     * @param Inputs Specify the number of inputs to this layers
     * @param Neurons How many nodes contained in this layer
     */
    public void layerBuilder(int Inputs, int Neurons){
        if(index<layer.length){
            layer[index++]=new NN_Layer(Inputs,Neurons);
        }else{
            throw new RuntimeException("Layer number out of Bounds");
        }
    }
    /**
     * Loops through the network and checks if outputs matched inputs in 
     * the next layer.
     * 
     */
    public void checkNetworkIntegrity(){
        
        for(int i=0;i<layer.length-1;i++){
            if(layer[i].getNumOutputs()!=layer[i+1].getNumInputs()){
                String s="Network Integrity error at layer "+(i+1);
                s+=" Outputs: "+layer[i].getNumOutputs();
                s+=" next layer inputs: "+layer[i+1].getNumInputs();
                throw new RuntimeException(s);
            }
        }
    }
    /**
     * Takes input Matrix and loops through the network feedforwarding results
     * @param in Input Matrix
     * @return Outputs Matrix
     */
    public Matrix makeGuess(Matrix in){
        Matrix[] m=new Matrix[layer.length+1];
        m[0]=new Matrix(in);
        for(int i=0;i<m.length-1;i++){
            m[i+1]=new Matrix(layer[i].feedForward(m[i]));
        }
        //Return outputs. Is the last layer of the network
        return m[m.length-1];
    }
    
    /**
     * Evaluates network performance. Computes the errors and produces
     * an error value
     * @param in Inputs to the Network
     * @param target Expected results
     * @param cost QUADRATIC or CROSS_ENTROPY
     * @return error value
     */
    public double evalNet(Matrix in, Matrix target,CostFunction cost){
        Matrix e=makeGuess(in);
        double costError=0;
        //Compute Cost Function
        switch(cost){
            case QUADRATIC:
                costError=quadratic(e,target);
                break;
            case CROSS_ENTROPY:
                costError=cross_entropy(e,target);
                break;
            default:
                break;
        }
        return costError;
    }
    /**
     * Evaluates network performance. Computes the errors and produces
     * an error value
     * @param in List of Inputs to the Network
     * @param target List of Expected results
     * @param cost QUADRATIC or CROSS_ENTROPY
     * @return error value
     */
    public double evalNet(List<Matrix> in, List<Matrix> target,CostFunction cost){
        int batch_size=in.size();
        double costError=0;
        Iterator it_in=in.iterator();
        Iterator it_tg=target.iterator();
        
        while(it_in.hasNext()){
            Matrix inp=(Matrix)it_in.next();
            Matrix targ=(Matrix)it_tg.next();
            Matrix e=makeGuess(inp);
        
        //Compute Cost Function
        switch(cost){
            case QUADRATIC:
                costError+=quadratic(e,targ);
                break;
            case CROSS_ENTROPY:
                costError=+cross_entropy(e,targ);
                break;
            default:
                break;
        }
        }
        return Math.abs(costError/batch_size);
    }

    
    private double convertError(Matrix m){
        double error=0;
        for(int i=0;i<m.getRows();i++){
            error+=m.getValue(i+1, 1);
        }
        return error;
    }
    
    private double cross_entropy(Matrix guess, Matrix target){
        double error=0;
        for(int i=0;i<guess.getRows();i++){
            error+=cross_ent(guess.getValue(i+1, 1),target.getValue(i+1, 1));
        }
        return (error/guess.getRows());
    }
    
    private double cross_ent(double a, double y){
        double error=y*Math.log(a)+(1-y)*Math.log(1-a);
        return error*-1;
    }
    /**
     * Quadratic Cost Error Function
     * error=1/n*SUM[(target-guess)^2]
     * n= number of elements in matrix guess
     * @param guess
     * @param target
     * @return 
     */
    private double quadratic(Matrix guess, Matrix target){
        double sum=0;
        double e=0;
        Matrix error=target.minus(guess);
        for(int i=0;i<error.getRows();i++){
            e=error.getValue(i+1, 1);
            sum+=e*e;
        }
        return sum;
    }
    
    public double trainNetwork(Matrix[] in, Matrix[] targets,CostFunction cost) throws FileNotFoundException, IOException{
        int batch_size=in.length;
        double costError=0;
        Matrix error=new Matrix(in[0].getRows(),1);
        //Compute Output error based on cost function
        for(int i=0;i<batch_size;i++){
        switch(cost){
            case QUADRATIC:
                error=(makeGuess(in[i]).minus(targets[i]));
                error=error.times_hadamard(layer[layer.length-1].getActivationDerivates());
                costError+=convertError(error);
                backPropagateError(error,in[i]);
                break;
            case CROSS_ENTROPY:
                error=makeGuess(in[i]).minus(targets[i]);
                costError+=convertError(error);
                backPropagateError(error,in[i]);
                break;
            default:
                break;
        }
        
        }
        //Save network to file
        File file=new File(name+".bin");
        FileOutputStream fos=new FileOutputStream(file);
        ObjectOutputStream oos=new ObjectOutputStream(fos);
        oos.writeObject(this);
        fos.flush();
        fos.close();
        
        return costError/batch_size;
    }
    
    public double trainNetwork(List<Matrix> in, List<Matrix> targets,CostFunction cost) throws FileNotFoundException, IOException{
        int batch_size=in.size();
        Matrix error=new Matrix(in.get(0).getRows(),1);
        Iterator it_in=in.iterator();
        Iterator it_tg=targets.iterator();
        double costError=0;
        
        
        //Compute Output error based on cost function
        while(it_in.hasNext()){
            Matrix input=(Matrix)it_in.next();
            Matrix target=(Matrix)it_tg.next();
        switch(cost){
            case QUADRATIC:
                error=(makeGuess(input).minus(target));
                costError+=Math.abs(convertError(error));
                error=error.times_hadamard(layer[layer.length-1].getActivationDerivates());
                backPropagateError(error,input);
                break;
            case CROSS_ENTROPY:
                error=makeGuess(input).minus(target);
                costError+=Math.abs(convertError(error));
                backPropagateError(error,input);
                break;
            default:
                break;
        }
 }
        //Save network to file
        File file=new File(name+".bin");
        FileOutputStream fos=new FileOutputStream(file);
        ObjectOutputStream oos=new ObjectOutputStream(fos);
        oos.writeObject(this);
        fos.flush();
        fos.close();
        
        return costError/batch_size;
    }
    
    private void backPropagateError(Matrix error, Matrix in){
        //Actualize output layer errors
        layer[layer.length-1].setErrors(error);
        //Back Propagate errors and compute delta weights
        for(int i=layer.length-1;i>0;i--){
            layer[i-1].setErrors(layer[i].getErrorGradient().times_hadamard(layer[i-1].getActivationDerivates()));
        }
        
        //Compute delta weights
        layer[0].adjustDeltaWeights(in);
        for(int i=1;i<layer.length;i++){
            layer[i].adjustDeltaWeights(layer[i-1].getActivationOutputs());
        }
        //Adjust weight matrices
        for(int i=0;i<layer.length;i++){
            layer[i].gradient_descend();
        }
    }
public void printNetworkInfo(){
    System.out.println("Neural Network Information");
    System.out.println("Inputs: "+layer[0].getNumInputs());
    System.out.println("Hidden Layers: "+(layer.length-1));
    System.out.println("Outputs: "+layer[layer.length-1].getNumOutputs());
    System.out.println();
    for(int i=0;i<layer.length;i++){
        System.out.printf("Layer %d Data\n",i+1);
        layer[i].printLayerInfo();
    }
}        
}

