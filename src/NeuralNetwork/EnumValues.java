
package NeuralNetwork;

/**
 * 
 * @author Federico
 */
public class EnumValues {
   
    //Activation Functions List
    public enum ActivationFunction {
    SIGMOID, TANH, ArcTAN,  LINEAR, ELU, RELU, LEAKY_RELU, SOFTPLUS, SOFTMAX
}

    //Matrices Initalization method
    public enum InitializeMethod {
    ZEROS, RANDOM, GAUSSIAN_RANDOM
}
    
    public enum CostFunction {
        QUADRATIC, CROSS_ENTROPY
    }
}
