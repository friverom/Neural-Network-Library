
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
    ZEROS, RANDOM, GAUSSIAN_RANDOM, XAVIER, HE
}
    //Cost Function list for training the network
    public enum CostFunction {
        QUADRATIC, CROSS_ENTROPY
    }
    
    //Gradient Descent Methods
    public enum GradientDescent{
        SGD,NESTEROV
    }
    
    //Learning Algorithm
    public enum LearningMethod{
        ONLINE,MINI_BATCH,BATCH
    }
}
