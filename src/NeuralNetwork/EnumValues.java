/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NeuralNetwork;

/**
 *
 * @author Federico
 */
public class EnumValues {
    
    public enum ActivationFunction {
    SIGMOID, TANH, ArcTAN,  LINEAR, ELU, RELU, LEAKY_RELU, SOFTPLUS, SOFTMAX
}

    public enum InitializeMethod {
    ZEROS, RANDOM, GAUSSIAN_RANDOM
}
    
    public enum CostFunction {
        QUADRATIC, CROSS_ENTROPY
    }
}
