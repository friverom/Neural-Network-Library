package NeuralNetwork;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */



import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import Matrix.Matrix;

/**
 *
 * @author Federico
 */
public class Neural_Network implements Serializable{
    
    private List<Matrix> weightsMatrix= new ArrayList<>();
    private List<Matrix> biasWeightsMatrix=new ArrayList<>();
    private List<Matrix> hiddenMatrix=new ArrayList<>();
    private List<Matrix> hiddenErrors=new ArrayList<>();
    
    private final double learningRate=0.05;
    private final double lambda=0.0;
    
    private static final long serialVersionUID = 2L;
   
    /**
     * Creates a Neural Network according to definition string
     * numbers in the definition string are the number of nodes for each layer.
     * The weights matrices are initialized with a random gaussian distribution
     * with mean=0 and standard deviation of 1.
     * @param definition "inputs,hidden1,hidden2,...,hidden n,outputs"
     */
    public Neural_Network(String definition){
        
        //Check correct spelling of string and get nunber of nodes per layer
        int[] layers=checkString(definition);
        //Items is an array where items[0] are the Inputs, items[1..(length-2]
        //are the hidden layers and items[lenght-1] are the Outputs
        //Add inputs vector to hiddenMatrix. First item in the List
        //are the inputs.
        Matrix inputs=new Matrix(layers[0],1);
        hiddenMatrix.add(inputs);
        //Create List of  matrices with random values
        for(int i=0;i<layers.length-1;i++){
            //Create weights Matrices
            Matrix a=Matrix.random_gaussian(layers[i+1], layers[i]);
            weightsMatrix.add(a);
            
            //Create Bias Vectors
            Matrix b=Matrix.random_gaussian(layers[i+1], 1);
            biasWeightsMatrix.add(b);
            
            //Create Hidden Values Matrix
            //Last vector are the Outputs
            Matrix c= new Matrix(layers[i+1],1);
            hiddenMatrix.add(c);
            
            //Create hidden errors Matrix
            //Last matrix is output errors
            Matrix d= new Matrix(layers[i+1],1);
            hiddenErrors.add(d);
        }
        
        
    }
    /**
     * Computes weights and bias with BackProgatation and Gradient Descent
     * Cost function used is Cross Enthropy
     * @param inputs Vector of known inputs
     * @param targets Vector of known outputs
     */
    
    public void trainNeuralNetwork(Matrix inputs, Matrix targets){
        
        //Feed forward inputs and get results
        makeGuess(inputs);
        
        //Compute Output errors.
        //error=(output-target) x (haddamar)(output*(1-output))
        Matrix outputs=hiddenMatrix.get(hiddenMatrix.size()-1);
        Matrix output_errors=outputs.minus(targets);
        //Eliminate next computation to have Cross Enthropy Cost Function
     //   output_errors=output_errors.times_hadamard(sigDerivate(outputs));
        //save output error in last matrix
        hiddenErrors.set(hiddenErrors.size()-1, output_errors);
        
        //Back propagate the errors
        //error(j)=[w(j+1)T x error(j+1)] x (haddamar) activation(j+1)
        
        for(int i=hiddenErrors.size()-1;i>0;i--){
            Matrix weights_T=weightsMatrix.get(i).transpose();
            Matrix delta=hiddenErrors.get(i);
            Matrix values=hiddenMatrix.get(i);
            
            Matrix grad=weights_T.times(delta);
            grad=grad.times_hadamard(values);
            hiddenErrors.set(i-1,grad);
        }    
        //Adjust weights and bias withGradient descent
       //deltas w(j) = learning_rate . hiddenErrors(j) x (hidden Values(j-1))T
       //W = W - delta
        for(int i=0;i<weightsMatrix.size();i++){
            Matrix in=hiddenMatrix.get(i).transpose();
            Matrix d=hiddenErrors.get(i).scale(learningRate);
            Matrix dW=d.times(in);
            
            //Actualize Weights and bias Matrices
            Matrix W=weightsMatrix.get(i);
            //Reguralize L2 weights matrix
          //  W=l2_reguralize(W);
            Matrix B=biasWeightsMatrix.get(i);
            W=W.minus(dW);
            weightsMatrix.set(i, W);
            B=B.minus(d);
            biasWeightsMatrix.set(i,B);
        }
        
    }
    /**
     * Makes a guess using the input vector
     * @param inputs vector
     * @return outputs vector
     */
    public Matrix makeGuess(Matrix inputs){
        
        //Load Inputs into item 0 in the hiddenMatrix
        hiddenMatrix.set(0, inputs);
        
        //Forward Pass
        for(int i=0;i<weightsMatrix.size();i++){
            Matrix h=weightsMatrix.get(i).times(hiddenMatrix.get(i));
            //Add bias
            h=h.plus(biasWeightsMatrix.get(i));
            
            //Calculate activation function values
            Matrix s=sigmoid(h);
            hiddenMatrix.set(i+1,s);
        }
        Matrix outputs=hiddenMatrix.get(hiddenMatrix.size()-1);
        outputs=softMax(outputs);
        return outputs;
    }
    /**
     * This method computes the derivates of vector assuming activation is
     * the sigmoid function
     * @param vector
     * @return 
     */
    private Matrix sigDerivate(Matrix vector){
        //Loop through vector and calculate derivates
        Matrix vec=new Matrix(vector);
        for(int i=0;i<vec.getRows();i++){
            double value=vec.getValue(i+1, 1);
            vec.setValue(i+1, 1, value*(1-value));
        }
        return vec;
    }
    /**
     * Computes the sigmoid activation function
     * @param vector
     * @return 
     */
    private Matrix sigmoid(Matrix vector){
        //Loop through vector and calculate sigmoid of each element
        Matrix vec=new Matrix(vector);
        for(int i=0;i<vec.getRows();i++){
            vec.setValue(i+1, 1, sigF(vec.getValue(i+1, 1)));
        }
        return vec;
    }
    /**
     * This method is the sigmoid activation function
     * @param i
     * @return 
     */
    private double sigF(double i){
        
        double sig=1/(1+Math.exp(-1*i));
        return sig;
    }
    
    /**
     * Checks the Neural Network definition string
     * item[0] is the input vector and the last item[x] is the output vector
     * in between are the hidden layers
     * @param s
     * @return 
     */
    private int[] checkString(String s){
        
        //Remove any space from string definition
        String str=s.replaceAll(" ", "");
        
        String[] items=s.split(",");
        //Convert string values to ints.
        int[] layers=new int[items.length];
        
        for(int i=0;i<items.length;i++){
            layers[i]=Integer.parseInt(items[i]);
        }
        
        return layers;
    }
    /**
     * This method computes the SoftMax of a vector
     * @param m is a Vector of values
     * @return exp is vector of outputs
     */
    private Matrix softMax(Matrix m){
        
        double sumX=0;
        
        for(int i=0;i<m.getRows();i++){
            sumX+=Math.exp(m.getValue(i+1, 1));
        }
        
        Matrix exp=m;
        double e=0;
        for(int i=0;i<m.getRows();i++){
            e=Math.exp(m.getValue(i+1, 1));
            m.setValue(i+1, 1, e/sumX);
        }
        return exp;
    }

private Matrix l2_reguralize(Matrix m){
    Matrix r=new Matrix(m);
    
    for(int i=0;i<r.getRows();i++){
        for(int j=0;j<r.getCols();j++){
            r.setValue(i+1, j+1, (1-learningRate*lambda)*m.getValue(i+1, j+1));
        }
    }
    return r;
} 


private void printList(String msg,List<Matrix> M){
    int i=1;
    System.out.println(msg);
    for (Matrix m : M) {
        System.out.println("Matrix "+i++);
        m.show();
        System.out.println();
    }
}
}
