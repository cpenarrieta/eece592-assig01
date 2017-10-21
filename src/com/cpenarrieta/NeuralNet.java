package com.cpenarrieta;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class NeuralNet implements NeuralNetInterface {

  final int numOutputs = 1;
  private Random rand = new Random();
  private double a;
  private double b;
  private int numInputs;
  private int numHidden;
  private double learningRate;
  private double momentumTerm;
  private double[] level1Weights;
  private double[] level2Weights;
  private double[] level1LastWeightsChange;
  private double[] level2LastWeightsChange;
  private double[] hiddenValues;
  private double[] hiddenValuesNoSigmoid;
  private int level1Lenght;
  private int level2Lenght;
  private double sigmoidBias;
  
	/**
	 * Constructor. (Cannot be declared in an interface, but your implementation will need one)
	 * @param argNumInputs The number of inputs in your input vector
	 * @param argNumHidden The number of hidden neurons in your hidden layer (Only a single hidden layer is supported).
	 * @param argLearningRate The learning rate coefficient
	 * @param argMomentumTerm The momentum coefficient
	 * @param argA Integer lower bound of sigmoid used by the output neuron only.
	 * @param arbB Integer upper bound of sigmoid used by the output neuron only.
	*/
	public NeuralNet (
	    int argNumInputs,
			int argNumHidden,
			double argLearningRate,
			double argMomentumTerm,
			double argA,
			double argB ) {
	  this.numInputs = argNumInputs;
	  this.numHidden = argNumHidden;
	  this.learningRate = argLearningRate;
	  this.momentumTerm = argMomentumTerm;
	  this.a = argA;
	  this.b = argB;
	  
	  level1Lenght = (numInputs + 1) * numHidden;
	  level2Lenght = (numHidden + 1) * numOutputs;
	  level1Weights = new double[level1Lenght];
	  level2Weights = new double[level2Lenght];
	  level1LastWeightsChange = new double[level1Lenght];
	  level2LastWeightsChange = new double[level2Lenght];
	  hiddenValues = new double[numHidden];
	  hiddenValuesNoSigmoid = new double[numHidden];
	  for (int i = 0; i < numHidden; i++) {
	    hiddenValues[i] = 0;
	    hiddenValuesNoSigmoid[i] = 0;
	  }
	  zeroWeights();
	  sigmoidBias = customSigmoid(bias);
	}
	 
	/**
   * @param X The input vector. An array of doubles.
   * @return The value returned by the Neural Net(NN) for this input vector
   */
	@Override
	public double outputFor(double[] X) {
	  for (int i = 0; i < numInputs + 1; i++) {
	    double tmpX = 0;
	    if (i != 0) {
	      tmpX = X[i - 1];
	    }
	    for (int j = 0; j < numHidden; j++) {
	      if (i == 0) {
	        hiddenValues[j] += (bias * level1Weights[j]);
	      } else {
	        hiddenValues[j] += (tmpX * level1Weights[(numHidden*(i)) + j]);	        
	      }
	    }
	  }
	  
	  for (int i = 0; i < numHidden; i++) {
	    hiddenValuesNoSigmoid[i] = hiddenValues[i];
	    hiddenValues[i] = customSigmoid(hiddenValues[i]);
	  }
	  
	  double sumResults = 0;
	  for (int i = 0; i < numHidden + 1; i++) {
	    if (i == 0) {
	      sumResults += (bias * level2Weights[0]);
	    } else {	      
	      sumResults += (hiddenValues[i - 1] * level2Weights[i]);
	    }
	  }
	  
	  return customSigmoid(sumResults);
	}
	
	public double gradientOfOutputWithRespectToInput(double finalOutput, double finalExpectedOutput, double hiddenValue, double input) {
	  return 0;
	}

	/**
   * This method will tell the Neural Net(NN) the output
   * value that should be mapped to the given input vector. I.e.
   * the desired correct output value for an input.
   * @param X The input vector
   * @param argValue The new value to learn
   * @return The error in the output for that input vector
   */
	@Override
	public double train(double[] X, double argValue) {
	  double output = outputFor(X);
	  double errorForOutput = calculateError(output, argValue);
	  
	  // level 2
	  double val;
	  double[] newLevel2Weights = new double[level2Lenght];
	  for (int i = 0; i < level2Lenght; i++) {
	    if (i == 0) {
	      val = sigmoidBias;
	    } else {
	      val = hiddenValues[i - 1];
	    }
	    double gradientWi = gradientWithRespecToWi(output, argValue, val);
	    double updatedWeight = newWeight(level2Weights[i], gradientWi);
	    newLevel2Weights[i] = updatedWeight;
	  }
	  
	  // level 1
	  double[] newLevel1Weights = new double[level1Lenght];
	  int tmp = numHidden;
    for (int i = 0; i < numInputs; i++) {
      for (int j = 0; j < numHidden; j++) {
        double gradientWi = gradientOfOutputWithRespectToInput(output, argValue, hiddenValues[j], X[i]);
        double updatedWeight = newWeight(level1Weights[tmp], gradientWi);
        newLevel1Weights[tmp] = updatedWeight;
        tmp++;
      }
    }
//	  int len = numInputs * numHidden;
//	  for (int i = 0; i < len; i++) {
////	    double gradientWi = gradientWithRespecToWi(output, argValue, hiddenValues[Math.round(i / numHidden)]);
//	    double gradientWi = gradientWithRespecToWi(hiddenValues[Math.round(i / numHidden)], argValue, X[Math.round(i / numHidden)]);
//      double updatedWeight = newWeight(level1Weights[i + numHidden], gradientWi);
//      newLevel1Weights[i + numHidden] = updatedWeight;
//	  }
	  
	  // level 1 - bias
	  for (int i = 0; i < numHidden; i++) {
//	    double gradientWi = gradientWithRespecToWi(hiddenValues[i], argValue, bias);
	    double gradientWi = gradientOfOutputWithRespectToInput(output, argValue, hiddenValues[i], bias);
      double updatedWeight = newWeight(level1Weights[i], gradientWi);
      newLevel1Weights[i] = updatedWeight;
	  }

	  //updating level 2
    for (int i = 0; i < level2Lenght; i++) {
      level2LastWeightsChange[i] = newLevel2Weights[i] - level2Weights[i];
      level2Weights[i] = newLevel2Weights[i];
    }
    
    //updating level 1
    for (int i = 0; i < level1Lenght; i++) {
      level1LastWeightsChange[i] = newLevel1Weights[i] - level1Weights[i];
      level1Weights[i] = newLevel1Weights[i];
    }
    
		return errorForOutput;
	}
	
	public void trainXORWithAcceptableError(double acceptableError) {
	  initializeWeights();

	  int numIterations = 0;
	  double error = 0;
	  double sumError = 0;
	  double avgError = 0;
	 	  
	  double[] input1 = { 0, 1 };
	  double target1 = 1;  
	  
	  double[] input2 = { 1, 0 };
	  double target2 = 1; 
	  
	  double[] input3 = { 1, 1 };
	  double target3 = 0;
  
	  double[] input4 = { 0, 0 };
	  double target4 = 0;
	  
	  do {
	    sumError = 0;
	    
	    error = train(input1, target1);
	    sumError += error;
	    
	    error = train(input2, target2);
	    sumError += error;
      
      error = train(input3, target3);
      sumError += error;
      
      error = train(input4, target4);
      sumError += error;
      
      numIterations++;
      avgError = sumError / 4;
	  } while (avgError >= acceptableError && numIterations < 10000000);
	  
	  System.out.println("final error: " + avgError);
	  System.out.println("============================");
	  
	  System.out.println("final Level 1 weights:");
	  for (int i = 0; i < level1Lenght; i++) {
	    System.out.println(level1Weights[i]);
	  }

	  System.out.println("============================");
	  System.out.println("final Level 2 weights:");
    for (int i = 0; i < level2Lenght; i++) {
      System.out.println(level2Weights[i]);
    }
    
    System.out.println("============================");
    System.out.println("iterations: " + numIterations);
	}

	/**
   * A method to write either a Look up Table(LUT) or weights of a Neural Net(NN) to a file.
   * @param argFile of type File.
   */
	@Override
	public void save(File argFile) {
	}

	/**
   * Loads the Look up Table(LUT) or neural net weights from file. The load must of course
   * have knowledge of how the data was written out by the save method.
   * You should raise an error in the case that an attempt is being
   * made to load data into an LUT or neural net whose structure does not match
   * the data in the file. (e.g. wrong number of hidden neurons).
   * @throws IOException
   */
	@Override
	public void load(String argFileName) throws IOException {
	}

	/**
   * Return a bipolar sigmoid of the input X
   * @param x The input
   * @return f(x) = 2 / (1+e(-x)) - 1
   */
	@Override
	public double sigmoid(double x) {
		return ((2) / (1 + Math.exp(-x))) - 1;
	}

	/**
   * This method implements a general sigmoid with asymptotes bounded by (a,b)
   * @param x The input 
   * @return f(x) = b_minus_a / (1 + e(-x)) - minus_a
   */
	@Override
	public double customSigmoid(double x) {
	  return ((b - a) / (1 + Math.exp(-x))) - (-a);
	}

	/**
   * Initialize the weights to random values.
   * For say 2 inputs, the input vector is [0] & [1]. We add [2] for the bias.
   * Like wise for hidden units. For say 2 hidden units which are stored in an array.
   * [0] & [1] are the hidden & [2] the bias.
   * We also initialise the last weight change arrays. This is to implement the alpha term.
   * 
   * For example, for numInputs=2 and numHidden=4
   * level1Weights=[biasWeight (BW),BW,BW,BW, Wx1,Wx1,Wx1,Wx1, Wx2,Wx2,Wx2,Wx2] array of 12 elements
   * level1Weights=[BW, Wx1, Wx2, Wx3, Wx4] array of 5 elements
   */
	@Override
	public void initializeWeights() {
	  for (int i = 0; i < level1Lenght; i++) {
	    double randomValue = getRandomValue();
	    level1LastWeightsChange[i] = randomValue - level1Weights[i]; 
      level1Weights[i] = randomValue;
    }
    
    for (int i = 0; i < level2Lenght; i++) {
      double randomValue = getRandomValue();
      level2LastWeightsChange[i] = randomValue - level2Weights[i]; 
      level2Weights[i] = randomValue;
    }
	}
	
	/**
   * function used for testing purpuse
   */
	public void predefineWeights(double[] level1, double[] level2) {
	  level1Weights = level1;
	  level2Weights = level2;
	}

	/**
   * Initialize the weights to 0.
   */
	@Override
	public void zeroWeights() {
	  for (int i = 0; i < level1Lenght; i++) {
	    level1Weights[i] = 0;
	    level1LastWeightsChange[i] = 0;
	  }
	  
	  for (int i = 0; i < level2Lenght; i++) {
      level2Weights[i] = 0;
      level2LastWeightsChange[i] = 0;
    }
	}
	
	public double getRandomValue() {
	  return rand.nextDouble() - 0.5;
	}
	
	public double calculateError(double output, double target) {
    return 0.5 * (Math.pow(target - output, 2));
  }
	
	public double gradientWithRespecToWi(double output, double target, double input) {
	  return -(target - output) * (output * (1 - output)) * input;
	}
	
	public double newWeight(double prev_weight, double gradientOfWeight) {
	  return prev_weight - (learningRate * gradientOfWeight);
	}
	
	public String printArray(double[] anArray) {
    String ret = "";
    for (int i = 0; i < anArray.length; i++) {
       if (i > 0) {
         ret += ", ";
       }
       ret += anArray[i];
    }
    return ret;
	}
}

//New weight = old weight + (momentum*delta weight) + (learning rate*error in output node*value of input node)
