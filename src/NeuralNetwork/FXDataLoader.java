
package NeuralNetwork;

import Matrix.Matrix;
import java.io.File;
import java.util.List;

/**
 * 
 * @author Federico
 */
public interface FXDataLoader {
    
    /**
     * This method reads the Data File and return a List of Matrix type
     * with the data.
     * 
     * @param file
     * @return List<Matrix> Inputs data
     */
    public abstract List<Matrix> loadData(File file);
    
    
    
}
