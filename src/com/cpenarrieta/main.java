package com.cpenarrieta;

public class main {
  private static String printArray(double[] anArray) {
    String ret = "";
    for (int i = 0; i < anArray.length; i++) {
       if (i > 0) {
         ret += ", ";
       }
       ret += anArray[i];
    }
    return ret;
  }
  
  public double derivativeOfSigmoid(double x) {
    return 2 * (x + 1) * (1 - ((x + 1) / 2));
  }
  
  public static void main(String[] args) {
    NeuralNet neuralNet = new NeuralNet(2, 4, 0.2, 0.0, 0, 1);
    neuralNet.trainXORWithAcceptableError(0.05, false);
    
//    NeuralNet neuralNet = new NeuralNet(2, 4, 0.2, 0.0, -1, 1);
//    neuralNet.trainXORWithAcceptableError(0.05, true);
    
//    NeuralNet neuralNet = new NeuralNet(2, 4, 0.2, 0.5, -1, 1);
//    neuralNet.trainXORWithAcceptableError(0.05, true);

//    double[] level1 = {-0.10586955232379607, -0.16125740156369492, 0.34820579005232954, 3.801722391563003, 0.12449697278603704, -0.5987261751812982, -0.37816485267691496, 0.09809841972009699, -0.3679305464073409, -0.4172068806589305, -0.1964735179050258, -0.8996378053073606};
//    double[] level2 = {4.048625591757146, 3.9680837634876E7, 3.708691424425362E7, 2.618524836524375E7, 8397396.792110272};
//    neuralNet.predefineWeights(level1, level2);
//    
//    double[] input1 = { -1, 1 };
//    System.out.println("-1 1 -> " + neuralNet.outputFor(input1) + " | 1");
//    
//    double[] input2 = { 1, -1 };
//    System.out.println("1 -1 -> " + neuralNet.outputFor(input2) + " | 1");
//    
//    double[] input3 = { 1, 1 };
//    System.out.println("1 1 -> " + neuralNet.outputFor(input3) + " | -1");
//    
//    double[] input4 = { -1, -1 };
//    System.out.println("-1 -1 -> " + neuralNet.outputFor(input4) + " | -1");
  }
}

 