
package NeuralNetwork;

import Matrix.Matrix;
import java.io.File;
import java.util.List;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

/**
 * 
 * @author Federico
 */
public interface FXDataLoader {
    
    /**
     * Reads the file and converts data to Matrix type and creates a List<Matrix>
     * 
     * @param file
     * @return List<Matrix> Inputs data
     */
    public abstract List<Matrix> loadData(File file);
    
}
