package com.cpenarrieta;

/**
 * @date
 * @author sarbjit
 *
 */
public interface NeuralNetInterface extends CommonInterface {
	
	final double bias = 1.0; // The input for each neurons bias weight
	
	/**
	 * Return a bipolar sigmoid of the input X
	 * @param x The input
	 * @return f(x) = 2 / (1+e(-x)) - 1
	 */
	public double sigmoid(double x);
	
	/**
	 * This method implements a general sigmoid with asymptotes bounded by (a,b)
	 * @param x The input 
	 * @return f(x) = b_minus_a / (1 + e(-x)) - minus_a
	 */
	public double customSigmoid(double x);
	
	/**
	 * Initialize the weights to random values.
	 * For say 2 inputs, the input vector is [0] & [1]. We add [2] for the bias.
	 * Like wise for hidden units. For say 2 hidden units which are stored in an array.
	 * [0] & [1] are the hidden & [2] the bias.
	 * We also initialise the last weight change arrays. This is to implement the alpha term.
	 */
	public void initializeWeights();
	
	/**
	 * Initialize the weights to 0.
	 */
	public void zeroWeights();
}