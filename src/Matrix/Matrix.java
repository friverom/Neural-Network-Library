package Matrix;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 *
 * 
 */
final public class Matrix implements Serializable{
    private final int M;             // number of rows
    private final int N;             // number of columns
    private final double[][] data;   // M-by-N array

    private static final long serialVersionUID = 2L;
    
    /**
     * 
     * @param M
     * @param N 
     */
    // create M-by-N matrix of 0's
    public Matrix(int M, int N) {
        this.M = M;
        this.N = N;
        data = new double[M][N];
    }

    // create matrix based on 2d array
    public Matrix(double[][] data) {
        M = data.length;
        N = data[0].length;
        this.data = new double[M][N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                    this.data[i][j] = data[i][j];
    }

    // copy constructor
    public Matrix(Matrix A) { this(A.data); }

    // create and return a random M-by-N matrix with values between -0.5 and 0.5
    public static Matrix random(int M, int N) {
        Matrix A = new Matrix(M, N);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                A.data[i][j] = (Math.random()-0.5)*2;
        return A;
    }
    
    // create and return a random M-by-N matrix with values between -1 and 1
    //normally distributed with mean=0 and sd=1
    public static Matrix random_gaussian(int M, int N) {
        Random random=new Random();
        Matrix A = new Matrix(M, N);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                A.data[i][j] = random.nextGaussian();
        return A;
    }

    // create and return a random M-by-N matrix with values between -1 and 1
    //normally distributed with mean=0 and sd=sqrt(1/n_inputs)
    public static Matrix xavier(int M, int N) {
        Random random=new Random();
        Matrix A = new Matrix(M, N);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                A.data[i][j] = random.nextGaussian()*Math.sqrt(1/M);
        return A;
    }
    
    // create and return a random M-by-N matrix with values between -1 and 1
    //normally distributed with mean=0 and sd=sqrt(2/n_inputs)
    public static Matrix he(int M, int N) {
        Random random=new Random();
        Matrix A = new Matrix(M, N);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                A.data[i][j] = random.nextGaussian()*Math.sqrt(2/M);
        return A;
    }
    // create and return the N-by-N identity matrix
    public static Matrix identity(int N) {
        Matrix I = new Matrix(N, N);
        for (int i = 0; i < N; i++)
            I.data[i][i] = 1;
        return I;
    }

    // swap rows i and j
    private void swap(int i, int j) {
        double[] temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }

    // create and return the transpose of the invoking matrix
    public Matrix transpose() {
        Matrix A = new Matrix(N, M);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                A.data[j][i] = this.data[i][j];
        return A;
    }

    // return C = A + B
    public Matrix plus(Matrix B) {
        Matrix A = this;
        if (B.M != A.M || B.N != A.N) throw new RuntimeException("Illegal matrix dimensions.");
        Matrix C = new Matrix(M, N);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                C.data[i][j] = A.data[i][j] + B.data[i][j];
        return C;
    }


    // return C = A - B
    public Matrix minus(Matrix B) {
        Matrix A = this;
        if (B.M != A.M || B.N != A.N) throw new RuntimeException("Illegal matrix dimensions.");
        Matrix C = new Matrix(M, N);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                C.data[i][j] = A.data[i][j] - B.data[i][j];
        return C;
    }

    // does A = B exactly?
    public boolean eq(Matrix B) {
        Matrix A = this;
        if (B.M != A.M || B.N != A.N) throw new RuntimeException("Illegal matrix dimensions.");
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                if (A.data[i][j] != B.data[i][j]) return false;
        return true;
    }

    //Hadamard product. Element wise multiplication
    public Matrix times_hadamard(Matrix B){
        Matrix A = this;
        if ((A.M != B.M) || (A.N != B.N)) throw new RuntimeException("Illegal matrix dimensions.");
        Matrix C = new Matrix(A.M,A.N);
        for(int i=0;i<A.M;i++){
            for(int j=0;j<A.N;j++){
                C.data[i][j]=A.data[i][j]*B.data[i][j];
            }
        }
        return C;
    }
    // return C = A * B
    public Matrix times(Matrix B) {
        Matrix A = this;
        if (A.N != B.M) throw new RuntimeException("Illegal matrix dimensions.");
        Matrix C = new Matrix(A.M, B.N);
        for (int i = 0; i < C.M; i++)
            for (int j = 0; j < C.N; j++)
                for (int k = 0; k < A.N; k++)
                    C.data[i][j] += (A.data[i][k] * B.data[k][j]);
        return C;
    }
    // return A*x multiply each elements by number
    public Matrix scale(double factor){
        Matrix A=this;
        for(int i=0;i<A.M;i++){
            for(int j=0;j<A.N;j++){
                A.data[i][j]=A.data[i][j]*factor;
            }
        }
        return A;
    }

    // return x = A^-1 b, assuming A is square and has full rank
    public Matrix solve(Matrix rhs) {
        if (M != N || rhs.M != N || rhs.N != 1)
            throw new RuntimeException("Illegal matrix dimensions.");

        // create copies of the data
        Matrix A = new Matrix(this);
        Matrix b = new Matrix(rhs);

        // Gaussian elimination with partial pivoting
        for (int i = 0; i < N; i++) {

            // find pivot row and swap
            int max = i;
            for (int j = i + 1; j < N; j++)
                if (Math.abs(A.data[j][i]) > Math.abs(A.data[max][i]))
                    max = j;
            A.swap(i, max);
            b.swap(i, max);

            // singular
            if (A.data[i][i] == 0.0) throw new RuntimeException("Matrix is singular.");

            // pivot within b
            for (int j = i + 1; j < N; j++)
                b.data[j][0] -= b.data[i][0] * A.data[j][i] / A.data[i][i];

            // pivot within A
            for (int j = i + 1; j < N; j++) {
                double m = A.data[j][i] / A.data[i][i];
                for (int k = i+1; k < N; k++) {
                    A.data[j][k] -= A.data[i][k] * m;
                }
                A.data[j][i] = 0.0;
            }
        }

        // back substitution
        Matrix x = new Matrix(N, 1);
        for (int j = N - 1; j >= 0; j--) {
            double t = 0.0;
            for (int k = j + 1; k < N; k++)
                t += A.data[j][k] * x.data[k][0];
            x.data[j][0] = (b.data[j][0] - t) / A.data[j][j];
        }
        return x;
   
    }

    public int getRows(){
        return this.M;
    }
    
    public int getCols(){
        return this.N;
    }
    public double getValue(int i,int j){
        return this.data[i-1][j-1];
    }
    
    public void setValue(int i, int j, double x){
        this.data[i-1][j-1]=x;
    }
    public double [][] getData(){
        return this.data;
    }
    public Matrix initMatrix(double value){
        Matrix M=this;
        for(int i=0;i<M.M;i++){
            for(int j=0;j<M.N;j++){
                M.setValue(i+1, j+1, value);
            }
        }
        return M;
       
    }
    // print matrix to standard output
    public void show() {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) 
                System.out.printf("%9.4f ", data[i][j]);
            System.out.println();
        }
    }
    
    //Return Matrix as String
    public String getMatrixAsString(){
        
        String s="";
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) 
                s+=String.format("%9.4f ", data[i][j]);
            s+="\n";
        }
        return s;
    }
    
    //Return vector from Matrix. Vector is a column
    public Matrix getVector(int index){
        
        Matrix vector=new Matrix(this.getRows(),1);
        
        for(int i=0;i<vector.getRows();i++){
            vector.setValue(i+1, 1, this.getValue(i+1, index));
        }
        return vector;
        
    }
    
    //Set vector in Matrix at index position. Vector is a Column
    public void setVector(Matrix vector,int index){
        
        for(int i=0;i<vector.getRows();i++){
            this.setValue(i+1, index, vector.getValue(i+1, 1));
        }
    }
    
    //Returns the Norm of a vector. (Length)
    public double getNorm(){
        
        double norm=0;
        
        for(int i=0;i<this.getRows();i++){
            norm+=Math.pow(this.getValue(i+1, 1), 2);
        }
        return Math.sqrt(norm);
    }
    
    //Returns the scalar multiplication of two vectors
    public double scalarMult(Matrix a){
        
        double scl=0;
        for(int i=0;i<a.getRows();i++){
            scl+=this.getValue(i+1, 1)*a.getValue(i+1, 1);
        }
        return scl;
    }
    
     /**
     * Returns the QR decomposition of a matrix.
     * The algorithm uses the Gram-Schmidt method
     * @param Matrix a
     * @return List. 1 element is the Q matrix and 2 is the R Matrix
     */
    public static List<Matrix> qr_decomposition(Matrix a) {

        Matrix q = new Matrix(a.getRows(), a.getCols());
        Matrix r = new Matrix(a.getRows(), a.getCols());

        for (int i = 0; i < a.getCols(); i++) {
            //Compute projections
            Matrix sum = new Matrix(q.getRows(),1);
            for (int j = 0; j < i; j++) {
                double factor=q.getVector(j+1).scalarMult(a.getVector(i+1));
                r.setValue(j+1, i+1, factor);
                sum=sum.plus(q.getVector(j+1).scale(factor));
            }

            Matrix qi = a.getVector(i + 1).minus(sum);
            double norm = qi.getNorm();
            q.setVector(qi.scale(1/norm), i + 1);
            r.setValue(i+1, i+1, norm);
        }
        
        List<Matrix> qr=new ArrayList<>();
        qr.add(q);
        qr.add(r);
        
        return qr;
 }   
    public double getSumItems(){
        
        double sum=0;
        for(int i=0;i<this.getRows();i++){
            for(int j=0;j<this.getCols();j++){
                sum+=this.getValue(i+1, j+1);
            }
        }
        return sum;
    }
    
    /**
     * Computes the eigenvalues and eigenvectors of a square and symetrical
     * matrix. The way eigenvalues and eigenvectors are computed is using the
     * QR algorithm. Matrix AP holds the QR decomposition and S will have
     * he eigenvectors.
     * 1.- A(k)=Q(k)R(k); S(k+1)=S(k)*Q(k)
     * 2.- A(k+1)=R(k)*Q(k)
     * 3.- iterate until Q --> Identity Matrix
     * @return  a list of Matrices. the first one is the eigenvalues on the
     * diagonal of the matrix and the second one is the eigenvectors in the
     * columns of the Matrix
     */
    public List<Matrix> eigenvalues(){
        
        Matrix ap=new Matrix(this);
        Matrix error=new Matrix(this.getRows(),this.getCols());
        Matrix s=Matrix.identity(this.getRows());
        
        List<Matrix> qr=new ArrayList<>();
        double err=1;
        int i=0;
        
        while(err>0.00001 && i<2000){
            qr=Matrix.qr_decomposition(ap);
            s=s.times(qr.get(0));
            err=(Math.abs(error.minus(qr.get(0)).getSumItems())/err)*100;
            error=qr.get(0);
            //System.out.print(err+", ");
            ap=qr.get(1).times(qr.get(0));
            i++;
        }
        
        List<Matrix> eigen = new ArrayList<>();
        eigen.add(ap);
        eigen.add(s);
        
        return eigen;
    }

}