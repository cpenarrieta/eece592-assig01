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
  
  public static void main(String[] args) {
    NeuralNet neuralNet = new NeuralNet(2, 4, 0.2, 0.0, 0, 1);
    neuralNet.trainXORWithAcceptableError(0.05);
  }
}
