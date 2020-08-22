
package NeuralNetwork;

import Matrix.Matrix;
import java.io.Serializable;
import NeuralNetwork.EnumValues;
import NeuralNetwork.EnumValues.ActivationFunction;
import NeuralNetwork.EnumValues.CostFunction;
import NeuralNetwork.EnumValues.InitializeMethod;
import NeuralNetwork.EnumValues.CostFunction;
import NeuralNetwork.EnumValues.GradientDescent;
import NeuralNetwork.EnumValues.LearningMethod;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.BlockingQueue;


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
    private boolean queueFlag=false;
        
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
    public String checkNetworkIntegrity() throws FileNotFoundException, IOException{
        String s="0";
        for(int i=0;i<layer.length-1;i++){
            if(layer[i].getNumOutputs()!=layer[i+1].getNumInputs()){
                s="Network Integrity error at layer "+(i+1);
                s+=" Outputs: "+layer[i].getNumOutputs();
                s+=" next layer inputs: "+layer[i+1].getNumInputs();
             //   throw new RuntimeException(s);
            }
        }
        return s; 
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
            error+=Math.abs(m.getValue(i+1, 1));
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
            e=Math.abs(error.getValue(i+1, 1));
            sum+=e;
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
                costError+=convertError(error);
                error=error.times_hadamard(layer[layer.length-1].getActivationDerivates());
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
    /**
     *Train method. it can perform learning with 3 type of algorithm, ONLINE, BATCH, MINI BATCH
     *ONLINE will adjust weights on each training sample.
     *BATCH will adjust the weights once on the average gradients of all samples
     *MINI Batch will adjust the weights once on the average gradients of all mini batches and will
     *loops randomly through all training data.
     *
     * @param in List of train data
     * @param targets List of targets
     * @param cost Cost Function to compute errors
     * @param learningMethod 
     * @param miniBatch
     * @return error
     * @throws FileNotFoundException
     * @throws IOException 
     */
     
    
    public double trainNetwork(List<Matrix> in, List<Matrix> targets,CostFunction cost, LearningMethod learningMethod, int miniBatch) throws FileNotFoundException, IOException, InterruptedException{
        
        double error=0;
        switch(learningMethod){
            case ONLINE:
                error=onlineTraining(in,targets,cost);
                break;
            case BATCH:
                error=batchTraining(in,targets,cost);
                break;
            case MINI_BATCH:
                error=miniBatchTraining(in,targets,cost,miniBatch);
                break;
            default:
                break;
        }
        //Save network to file
        File file=new File(name+".bin");
        FileOutputStream fos=new FileOutputStream(file);
        ObjectOutputStream oos=new ObjectOutputStream(fos);
        oos.writeObject(this);
        fos.flush();
        fos.close();
        return error;
    }
    //Creates a mini List of inputs and targets randomly from the inputs
    private double miniBatchTraining(List<Matrix> in, List<Matrix> targets, CostFunction cost, int miniBatch) throws InterruptedException{
        //Create mini Batch List array
        int miniBatchListSize=in.size()/miniBatch;
        List<Matrix> mini_in_list=new ArrayList<>();
        List<Matrix> mini_tg_list=new ArrayList<>();
        Random random=new Random();
        double error=0;
        
        int index=random.nextInt(in.size());
        
        for(int i=0;i<(miniBatchListSize-miniBatch);i++){
            for(int j=0;j<miniBatch;j++){
                Matrix m_in=new Matrix(in.get(index));
                Matrix m_tg=new Matrix(targets.get(index));
                mini_in_list.add(m_in);
                mini_tg_list.add(m_tg);
                index=random.nextInt(in.size());
            }
            error=batchTraining(mini_in_list, mini_tg_list,cost);
            
            mini_in_list.clear();
            mini_tg_list.clear();
        }
        return error;
    }
    private double batchTraining(List<Matrix> in, List<Matrix> targets, CostFunction cost) throws InterruptedException {
        int data_size = in.size();
        Matrix error = new Matrix(in.get(0).getRows(), 1);
        Iterator it_in = in.iterator();
        Iterator it_tg = targets.iterator();
        double costError = 0;

        //Compute Output error based on cost function
        while (it_in.hasNext()) {
            Matrix input = (Matrix) it_in.next();
            Matrix target = (Matrix) it_tg.next();

            error = computeError(input, target, cost);
            costError += Math.abs(convertError(error));
            backPropagateError(error, input);
        }
        
        gradientDescent();
        
        return costError / data_size;
    }

    private double onlineTraining(List<Matrix> in, List<Matrix> targets, CostFunction cost) throws FileNotFoundException, IOException, InterruptedException {
        int data_size = in.size();
        Matrix error = new Matrix(in.get(0).getRows(), 1);
        Iterator it_in = in.iterator();
        Iterator it_tg = targets.iterator();
        double costError = 0;

        //Compute Output error based on cost function
        while (it_in.hasNext()) {
            Matrix input = (Matrix) it_in.next();
            Matrix target = (Matrix) it_tg.next();

            error = computeError(input, target, cost);
            costError += Math.abs(convertError(error));
            backPropagateError(error, input);
            gradientDescent();
        }
        
        return costError / data_size;
    }
    //Compute error base on cost function
    private Matrix computeError(Matrix input, Matrix target, CostFunction cost ){
        Matrix error=new Matrix(input.getRows(),1);
               
        switch(cost){
            case QUADRATIC:
                error=(makeGuess(input).minus(target));
                error=error.times_hadamard(layer[layer.length-1].getActivationDerivates());
                break;
            case CROSS_ENTROPY:
                error=(makeGuess(input).minus(target));
          //      error=error.times_hadamard(layer[layer.length-1].getActivationDerivates());
                break;
            default:
                break;
        }
        return error;
    }
    //Cross Entropy gradient calculation
    //derivate=(guess-target)/(guess*(1-guess))
    private Matrix cross_entropy_gradient(Matrix inputs,Matrix targets){
        Matrix guess=makeGuess(inputs);
        Matrix error=guess.minus(targets);
        
         Matrix grad=guess;
        for(int i=0;i<guess.getRows();i++){
            grad.setValue(i+1, 1, inv_res(error.getValue(i+1, i),guess.getValue(i+1, 1)));
        }
        return grad;
    }

    private double inv_res(double error, double guess ){
        
        return(error/(guess*(1-guess)));
    }
    //Back propagate the errors
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
        
    }
    
    private void gradientDescent(){
        //Adjust weight matrices
        for(int i=0;i<layer.length;i++){
            layer[i].gradient_descend();
        }
    }
    
    public void setLearningRate(double rate){
        this.learning_rate=rate;
        for(NN_Layer l:layer){
            l.setLearningRate(rate);
        }
    }
    
    public double getLearningRate(){
        return learning_rate;
    }
    
    public void setNesterovFactor(double n_factor){
        this.n_factor=n_factor;
    }
    
    public double getNesterovFactor(){
        return n_factor;
    }
    
    public double getLambda(){
        return lambda;
    }
    public void setLambda(double lambda){
        this.lambda=lambda;
        for(NN_Layer l:layer){
            l.setLambda(lambda);
        }
    }
    
    public String getNetworkName(){
        return name;
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

public List<String> getNetworkInfo(){
    
    List<String> str=new ArrayList<>();
    String s=name+","+index+","+learning_rate+","+lambda;
    str.add(s);
    for(NN_Layer l:layer){
        s=l.getLayerInfo();
        str.add(s);
    }
    return str;
}
}

