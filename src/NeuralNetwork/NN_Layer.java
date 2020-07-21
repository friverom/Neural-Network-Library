
package NeuralNetwork;

import Matrix.Matrix;
import NeuralNetwork.EnumValues.ActivationFunction;
import NeuralNetwork.EnumValues.GradientDescent;
import NeuralNetwork.EnumValues.InitializeMethod;
import java.io.Serializable;

/**
 * This class creates a Layer for a neural network and provides methods
 * to easily compute Back Propagation training in the Neural Network.
 * @author Federico
 */
public class NN_Layer implements Serializable{
    
    private Matrix weights; //weights Matrix of size(#neurons x inputs)
    private Matrix delta_w; //Delta weights for Gradient Descend
    private Matrix nesterov_deltas; //Deltas for Nesterov AGD
    private Matrix bias; //bias vector 
    private Matrix delta_b; //Delta bias for Gradient Descend
    private Matrix outputs; //Activation function results
    private Matrix z; //Outputs of neurons before activation
    private Matrix errors; //Vector of size #neurons
    private ActivationFunction activation; //Activation Function Type
    private double learning_rate;
    private double lambda=0; //Regularization factor
    private double n_factor=0;
    private InitializeMethod initialization; //Matrices initialization method
    private GradientDescent gradientMethod;
    private double batch_count=0;
    private static final long serialVersionUID=2L;
    /**
     * Neural Network layer constructor
     * @param definition "#in,#neurons,Activation Function, Initialization, 
     * learning rate, lambda, Gradient Descent, Nesterov factor
     */
    public NN_Layer(int inputs, int neurons, ActivationFunction act, InitializeMethod init,double lr,double lambda, GradientDescent grd, double n){
        
        //Set Matrices Initialization method
        this.initialization=init;
        //Set Activation Function
        this.activation=act;
        
        this.learning_rate=lr;
        this.lambda=lambda;
        this.gradientMethod=grd;
        this.n_factor=n;
        //Initialize Matrices
        createMatrices(inputs,neurons);
        
        }
    
    /**
     * Neural Network layer constructor
     * @param definition "#in,#neurons,Activation Function, Initialization, 
     * learning rate, lambda, Gradient Descent
     */
    public NN_Layer(int inputs, int neurons, ActivationFunction act, InitializeMethod init,double lr,double lambda, GradientDescent grd){
        
        //Set Matrices Initialization method
        this.initialization=init;
        //Set Activation Function
        this.activation=act;
        
        this.learning_rate=lr;
        this.lambda=lambda;
        this.gradientMethod=grd;
        this.n_factor=0;
        //Initialize Matrices
        createMatrices(inputs,neurons);
        
        }
    
    /**
     * Neural Network layer constructor
     * @param definition "#in,#neurons
     */
    public NN_Layer(int inputs, int neurons){
        
        //Set Matrices Initialization method
        this.initialization=InitializeMethod.GAUSSIAN_RANDOM;
        //Set ActivationFunction Function
        this.activation=ActivationFunction.SIGMOID;
        this.gradientMethod=GradientDescent.SGD;
        this.learning_rate=0.05;
        this.lambda=0;
        //Initialize Matrices
        createMatrices(inputs,neurons);
        
        }
    
    private void createMatrices(int in, int hidden){
        
        int inputs=in;
        int neurons=hidden;
        //Initialize weight Matrix depending on Initialization method requested
        switch(initialization){
            //Matrix elements = 0
            case ZEROS:
                weights=new Matrix(neurons,inputs);
                delta_w=new Matrix(neurons,inputs);
                nesterov_deltas=new Matrix(neurons,inputs);
                bias=new Matrix(neurons,1);
                delta_b=new Matrix(neurons,1);
                outputs=new Matrix(neurons,1);
                errors=new Matrix(neurons,1);
                z=new Matrix(neurons,1);
                break;
            //Matrix elements initialize with random number from -0.5 to 0.5    
            case RANDOM:
                weights=Matrix.random(neurons, inputs);
                delta_w=new Matrix(neurons,inputs);
                nesterov_deltas=new Matrix(neurons,inputs);
                bias=Matrix.random(neurons, 1);
                delta_b=new Matrix(neurons,1);
                outputs=new Matrix(neurons,1);
                errors=new Matrix(neurons,1);
                z=new Matrix(neurons,1);
                break;
            //Matrix elements initalize with random number with mean = 0
            // and standard deviation =1
            case GAUSSIAN_RANDOM:
                weights=Matrix.random_gaussian(neurons, inputs);
                delta_w=new Matrix(neurons,inputs);
                nesterov_deltas=new Matrix(neurons,inputs);
                bias=Matrix.random_gaussian(neurons, 1);
                delta_b=new Matrix(neurons,1);
                outputs=new Matrix(neurons,1);
                errors=new Matrix(neurons,1);
                z=new Matrix(neurons,1);
                break;
            //Matrix elements initialize with random number with mean=0 and
            //sd=sqrt(1/inputs)
                case XAVIER:
                weights=Matrix.random_gaussian(neurons, inputs);
                delta_w=new Matrix(neurons,inputs);
                nesterov_deltas=new Matrix(neurons,inputs);
                bias=Matrix.random_gaussian(neurons, 1);
                delta_b=new Matrix(neurons,1);
                outputs=new Matrix(neurons,1);
                errors=new Matrix(neurons,1);
                z=new Matrix(neurons,1);
                break;
            //Matrix elements initialize with random number with mean=0 and
            //sd=sqrt(2/inputs)
                case HE:
                weights=Matrix.random_gaussian(neurons, inputs);
                delta_w=new Matrix(neurons,inputs);
                nesterov_deltas=new Matrix(neurons,inputs);
                bias=Matrix.random_gaussian(neurons, 1);
                delta_b=new Matrix(neurons,1);
                outputs=new Matrix(neurons,1);
                errors=new Matrix(neurons,1);
                z=new Matrix(neurons,1);
                break;
            default:
                weights=Matrix.random(neurons, inputs);
                delta_w=new Matrix(neurons,inputs);
                nesterov_deltas=new Matrix(neurons,inputs);
                bias=Matrix.random(neurons, 1);
                delta_b=new Matrix(neurons,1);
                outputs=new Matrix(neurons,1);
                errors=new Matrix(neurons,1);
                z=new Matrix(neurons,1);
                break;
        }
    }
    
    public Matrix getActivationOutputs(){
        return outputs;
    }
    
    public Matrix getOutputs(){
        return z;
    }
    /**
     * et number of outputs from the layer
     * @return number of outputs
     */
    public int getNumOutputs(){
        return outputs.getRows();
    }
    /**
     * Get number of inputs to the layer
     * @return number of inputs
     */
    public int getNumInputs(){
        return weights.getCols();
    }
    /**
     * Returns weights(Transpose) x errors to be used for backpropagation
     * @return Matrix
     */
    public Matrix getErrorGradient(){
        Matrix m=weights.transpose();
        m=m.times(errors);
        
        return m;
    }
   /**
    * Get Error vector
    * @return errors vector
    */    
    public Matrix getErrors(){
        Matrix e=new Matrix(errors);
        return e;
    }
    
    /**
     * Save errors 
     * @return void
     */
    public void setErrors(Matrix m){
        
        for(int i=0;i<m.getRows();i++){
            errors.setValue(i+1, 1, m.getValue(i+1, 1));
        }
    }
    
    /**
     * Get the weights Matrix
     * @return 
     */
    public Matrix getWeights(){
        Matrix w=new Matrix(weights);
        return w;
    }
    
    /**
     * Get the derivatives of outputs
     * @return derivate vector
     */
    public Matrix getActivationDerivates(){
        return derivates();
    }
    /**
     * Adjust the weights and bias and apply Gradient Descend
     * @param in Inputs to this layer
     */
    public void gradient_descend(){
        //Compute deltas
        delta_w=delta_w.scale(1/batch_count);
        delta_b=delta_b.scale(1/batch_count);
        
        //Compute Reguralization values
        Matrix reg=L2_reguralize(weights);
        reg=reg.scale(1/batch_count);
        
        switch (gradientMethod) {
            case SGD:
                //Compute Gradient Descend
                weights = weights.minus(delta_w);
                weights = weights.minus(reg);
                bias = bias.minus(delta_b);
                break;
            case NESTEROV:
                delta_w=delta_w.minus(nesterov_deltas);
                //Compute Gradient Descend
                weights = weights.minus(delta_w);
                weights = weights.minus(reg);
                bias = bias.minus(delta_b);
                nesterov_deltas=delta_w.scale(n_factor); //copy deltas to create deltas(t-1)
                break;
            default:
                break;
        }
        
        batch_count=0; //reset batch counter for next batch
        
    }
    /**
     * Compute the gradient of the error for back propagation
     * grad=weights(transpose) x errors x learning rate
     * @param in 
     */
    public void adjustDeltaWeights(Matrix in){
        Matrix gradient=errors.scale(learning_rate);
        delta_b=delta_b.plus(gradient);
        delta_w=delta_w.plus(gradient.times(in.transpose()));
        batch_count++;
    }
    
    public Matrix feedForward(Matrix in){
        //Feed forward
        //z = weights x in + bias
        z=weights.times(in);
        z=z.plus(bias);
        
        //output = Activation Function(z)
        switch(activation){
            case SIGMOID:
                outputs=sigmoid(z);
                break;
            case TANH:
                outputs=tanh(z);
                break;
            case ArcTAN:
                outputs=arcTan(z);
                break;
            case ELU:
                outputs=elu(z);
                break;
            case RELU:
                outputs=relu(z);
                break;
            case LEAKY_RELU:
                outputs=l_relu(z);
                break;
            case SOFTMAX:
                break;
            case SOFTPLUS:
                outputs=softplus(z);
                break;
            case LINEAR:
                outputs=linear(z);
            default:
                outputs=sigmoid(z);
                break;
        }
        
        return outputs;
    }
    /**
     * Computes the derivatives of the activation functions
     * @return derivatives vector
     */
    private Matrix derivates(){
        
        Matrix derivatives=new Matrix(outputs);
        
        switch(activation){
            case SIGMOID:
                derivatives=derivate_sigmoid(outputs);
                break;
            case TANH:
                derivatives=derivative_tanh(z);
                break;
            case ArcTAN:
                derivatives=derivative_arcTan(z);
                break;
            case ELU:
                derivatives=derivative_elu(z);
                break;
            case RELU:
                derivatives=derivative_relu(z);
                break;
            case LEAKY_RELU:
                derivatives=derivative_l_relu(z);
                break;
            case SOFTMAX:
                derivatives=derivative_softmax(z);
                break;
            case SOFTPLUS:
                derivatives=derivative_softplus(z);
                break;
            case LINEAR:
                derivatives=derivative_linear(z);
            default:
                derivatives=derivate_sigmoid(outputs);
                break;
        }
        return derivatives;
    }
    /**
     * Linear Activation Function
     * f(x)=x
     * @param m
     * @return 
     */
    private Matrix linear(Matrix m){
        Matrix linear=new Matrix(m);
        return linear;
    }
    
    /**
     * Derivative of Linear Function
     * f'(x)=1
     * @param m
     * @return 
     */
    private Matrix derivative_linear(Matrix m){
        Matrix deriv=new Matrix(m);
        for(int i=0;i<deriv.getRows();i++){
            deriv.setValue(i+1,1,1);
        }
        return deriv;
    }
    /**
     * Exponential Linear Unit activation function
     * elu(x)=alfa*(exp(x)-1) x<0 : x x>=0
     * @param m
     * @return 
     */
    private Matrix elu(Matrix m){
        Matrix elu = new Matrix(m);
        double alfa=0.01;
        double value = 0;
        for (int i = 0; i < elu.getRows(); i++) {
            value = m.getValue(i + 1, 1);
            if (value < 0) {
                elu.setValue(i + 1, 1, ((Math.exp(m.getValue(i+1, 1)))-1)*alfa);
            } else {
                elu.setValue(i + 1, 1, value);
            }
        }
        return elu;
    }
    /**
     * Derivatives of ELU function
     * dELU=elu(x)+alfa x<0 : 1 x>=0
     * @param m
     * @return 
     */
    private Matrix derivative_elu(Matrix m) {
        Matrix deriv = new Matrix(m);
        double value = 0;
        double alfa = 0.01;
        for (int i = 0; i < deriv.getRows(); i++) {
            value = m.getValue(i + 1, 1);
            if (value < 0) {
                deriv.setValue(i + 1, 1, ((Math.exp(m.getValue(i+1, 1)))-1+alfa));
            } else {
                deriv.setValue(i + 1, 1, 1);
            }
        }
        return deriv;
    }
    /**
     * Arc Tangent Activation function
     * arcTan(x)= atan(x)
     * @param m
     * @return 
     */
    private Matrix arcTan(Matrix m){
        Matrix atan=new Matrix(m);
        for(int i=0;i<atan.getRows();i++){
            atan.setValue(i+1, 1, Math.atan(m.getValue(i+1, 1)));
        }
        return atan;
    }
    /**
     * Derivatives of ArcTan activated values
     * derivative atan(x)=1/(1+x^2)
     * @param m
     * @return 
     */
    private Matrix derivative_arcTan(Matrix m){
        Matrix deriv=new Matrix(m);
        for(int i=0;i<deriv.getRows();i++){
            deriv.setValue(i+1, 1, derivate_arcTan(m.getValue(i+1, 1)));
        }
        return deriv;
    }
    /**
     * Derivatives of ArcTan activated values
     * derivative atan(x)=1/(1+x^2)
     * @param value
     * @return 
     */
    private double derivate_arcTan(double value){
        return 1/(1+value*value);
    }
    
    /**
     * SoftPlus Activation Function
     * s(x)=ln(1+exp(x))
     * @param m
     * @return 
     */
    private Matrix softplus(Matrix m){
        Matrix act=new Matrix(m);
        for(int i=0;i<act.getRows();i++){
            act.setValue(i+1, 1, softplus(m.getValue(i+1, 1)));
        }
        return act;
    }
    
    /**
     * SoftPlus Activation Function
     * s(x)=ln(1+exp(x))
     * @param value
     * @return 
     */
    private double softplus(double value){
        return Math.log(1+Math.exp(value));
    }
    
    /**
     *Derivative of SoftPlus activated values
     * derivative softplus(x)=sigmoid(x); 
     */
    private Matrix derivative_softplus(Matrix m){
        Matrix act=new Matrix(m);
        act=sigmoid(m);
        
        return act;
    }
    
    /**
     * Leaky RELU activation function
     * l_relu(x)= alfa*x x<0 : x x>=0
     * @param m
     * @return 
     */
    private Matrix l_relu(Matrix m) {
        Matrix relu = new Matrix(m);
        double alfa=0.01;
        double value = 0;
        for (int i = 0; i < relu.getRows(); i++) {
            value = m.getValue(i + 1, 1);
            if (value < 0) {
                relu.setValue(i + 1, 1, value*alfa);
            } else {
                relu.setValue(i + 1, 1, value);
            }
        }
        return relu;
    }
    
    /**
     * Derivatives of Leaky RELU activated values
     * derivative l_relu(x)= alfa x<0 : 1 x>=0
     * @param m
     * @return 
     */
    private Matrix derivative_l_relu(Matrix m){
        Matrix deriv=new Matrix(m);
        double value=0;
        double alfa=0.01;
        for(int i=0;i<deriv.getRows();i++){
            value=m.getValue(i+1, 1);
            if(value<0){
                deriv.setValue(i+1, 1, alfa);
            }else{
                deriv.setValue(i+1, 1, 1);
            }
        }
        return deriv;
    }
    /**
     * Rectified Linear Unit RELU function
     * f(x): 0 if x<0 : x>=0
     * @param m
     * @return 
     */
    private Matrix relu(Matrix m){
        Matrix relu=new Matrix(m);
        double value=0;
        for(int i=0;i<relu.getRows();i++){
            value=m.getValue(i+1, 1);
            if(value<0){
                relu.setValue(i+1, 1, 0);
            }else{
                relu.setValue(i+1, 1, value);
            }
        }
        return relu;
    }
    
    /**
     * Derivatives of RELU activated values
     * derivate RELU(x)= 0 x<0 : 1 x>=0
     * @param m
     * @return 
     */
    private Matrix derivative_relu(Matrix m){
        Matrix deriv=new Matrix(m);
        double value=0;
        for(int i=0;i<deriv.getRows();i++){
            value=m.getValue(i+1, 1);
            if(value<0){
                deriv.setValue(i+1, 1, 0);
            }else{
                deriv.setValue(i+1, 1, 1);
            }
        }
        return deriv;
    }
    /**
     * Sigmoid activation function
     * 
     * @param m vector of values
     * @return vector of values
     */
    private Matrix sigmoid(Matrix m){
        
        Matrix act=new Matrix(m);
        for(int i=0;i<act.getRows();i++){
            act.setValue(i+1, 1, sigmoid(m.getValue(i+1, 1)));
        }
        return act;
    }
    /**
     * Computes the sigmoid of value x
     * sigmoid(x)=1/(1+e^(-1*x))
     * @param double value
     * @return double sigmoid(value)
     */
    private double sigmoid(double value){
        
        return 1/(1+Math.exp(-1*value));
    }
    /**
     * Computes the derivatives of Sigmoid activated outputs
     * dSigmoid(x)=sigmoid(x)*(1-sigmoid(x))
     * @param vector of values
     * @return vector of derivatives
     */
    private Matrix derivate_sigmoid(Matrix m){
        
        Matrix deriv=new Matrix(m);
        for(int i=0;i<deriv.getRows();i++){
            deriv.setValue(i+1, 1, derivate_sigmoid(m.getValue(i+1, 1)));
        }
        return deriv;
    }
    
    /**
     * Computes the derivative of Sigmoid function
     * dSigmoid(x)=sigmoid(x)*(1-sigmoid(x))
     * @param value
     * @return 
     */
    private double derivate_sigmoid(double value){
    
        return value*(1-value);
    }
    /**
     * Computes the TANH of vector m
     * tanh(x)=[2/(1+e^(-2*x))]-1
     * @param m
     * @return 
     */
    private Matrix tanh(Matrix m){
        Matrix tanh=new Matrix(m);
        
        for(int i=0;i<tanh.getRows();i++){
            tanh.setValue(i+1, 1, tanh(m.getValue(i+1, 1)));
        }
        return tanh;
    }
    /**
     * Computes tanh of value
     * @param value
     * @return tanh
     */
    private double tanh(double value){
        double tanh=(2/(1+Math.exp(-2*value)))-1;
        return tanh;
    }
    /**
     * Computes derivative of tanh vector m
     * f'(x)=1=f(2)^2
     * @param m
     * @return 
     */
    private Matrix derivative_tanh(Matrix m){
        Matrix der=new Matrix(m);
        for(int i=0;i<der.getRows();i++){
            der.setValue(i+1, 1, derivative_tanh(m.getValue(i+1, 1)));
        }
        return der;
    }
    /**
     * Computes derivative of value
     * f'(x)=1=f(2)^2
     * @param value
     * @return 
     */
    private double derivative_tanh(double value){
        double t=tanh(value);
        return 1-t*t;
    }
    /** 
     * SoftMax ActivationFunction function
     * S(x)= e^x/(SUM(e^x)); x is a vector of values
     * @param vector of output values
     * @return vector of SoftMax Values
     */
    private Matrix softMax(Matrix m){
        Matrix softmax=new Matrix(m);
        
        double sum=0;
        //Compute SUM(e^x)
        for(int i=0;i<softmax.getRows();i++){
            sum+=Math.exp(m.getValue(i+1, 1));
        }
        //Compute softmax of vector element and store it in softmax vector
        for(int i=0;i<softmax.getRows();i++){
            softmax.setValue(i+1, 1, Math.exp(m.getValue(i+1, 1))/sum);
        }
        return softmax;
    }
    private Matrix derivative_softmax(Matrix m){
       Matrix deriv=new Matrix(m);
        for(int i=0;i<deriv.getRows();i++){
            deriv.setValue(i+1,1,1);
        }
        return deriv;
    }
    
    private Matrix L2_reguralize(Matrix m){
    Matrix r=new Matrix(m);
    
    for(int i=0;i<r.getRows();i++){
        for(int j=0;j<r.getCols();j++){
            r.setValue(i+1, j+1, lambda*learning_rate*m.getValue(i+1, j+1));
        }
    }
    return r;
} 
    public void printLayerInfo(){
        
        int weightColumns=weights.getCols();
        int weightRows=weights.getRows();
        
        int biasColumns=bias.getCols();
        int biasRows=bias.getRows();
        
        int outputColumns=outputs.getCols();
        int outputRows=outputs.getRows();
        
        int errorColumns=errors.getCols();
        int errorRows=errors.getRows();
        
        System.out.println("Inputs: "+weightColumns+" Neurons: "+weightRows);
        System.out.println("Activation Function: "+activation);
        System.out.println("Init type: "+initialization);
        System.out.println("Weights Matrix");
        weights.show();
        System.out.println("Bias Matrix");
        bias.show();
        System.out.println("Outputs Matrix");
        outputs.show();
        System.out.println("Errors Matrix");
        errors.show();
    }
    
    
}    

